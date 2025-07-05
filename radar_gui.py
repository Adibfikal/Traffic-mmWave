import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTextEdit, QGroupBox, QGridLayout, QSplitter, QTabWidget, QComboBox,
                             QFileDialog, QSlider, QSpinBox, QCheckBox, QInputDialog, QDialog, QListWidget)
from PySide6.QtCore import QTimer, Signal, QThread, Slot, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from collections import deque
import time
import cv2
import os

from radar_interface import RadarInterface
from data_recorder import DataRecorder
from enhanced_hdbscan_tracker import EnhancedHDBSCANTracker
from enhanced_visualization import EnhancedTrackingVisualization

class WebcamThread(QThread):
    """Thread for webcam capture without blocking the GUI"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.is_running = False
        self.camera_index = 0
        
    def setup_camera(self, camera_index=0):
        """Initialize camera with specified index"""
        try:
            self.camera_index = camera_index
            
            # Try different backends for external cameras (Windows)
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.camera = cv2.VideoCapture(camera_index, backend)
                    
                    if not self.camera.isOpened():
                        if self.camera:
                            self.camera.release()
                        continue
                        
                    # Set camera properties for better performance
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
                    
                    # Test if we can actually read frames
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        backend_name = {
                            cv2.CAP_DSHOW: "DirectShow",
                            cv2.CAP_MSMF: "Microsoft Media Foundation", 
                            cv2.CAP_ANY: "Default"
                        }.get(backend, f"Backend {backend}")
                        
                        self.error_occurred.emit(f"Camera {camera_index} connected using {backend_name}")
                        return True
                    else:
                        # Can't read frames, try next backend
                        self.camera.release()
                        continue
                        
                except Exception as e:
                    if self.camera:
                        self.camera.release()
                    continue
            
            # If we get here, all backends failed
            raise Exception(f"Cannot access camera {camera_index} with any backend. Camera may be in use by another application.")
                
        except Exception as e:
            self.error_occurred.emit(f"Failed to setup camera: {str(e)}")
            return False
    
    def run(self):
        """Main thread loop for capturing frames"""
        self.is_running = True
        while self.is_running:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret:
                        # Convert BGR to RGB for Qt display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(frame_rgb)
                    else:
                        self.error_occurred.emit("Failed to read frame from camera")
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.error_occurred.emit(f"Error capturing frame: {str(e)}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop the thread and release camera"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.wait()

class RadarDataThread(QThread):
    """Thread for reading radar data without blocking the GUI"""
    data_ready = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.radar = None
        self.is_running = False
        
    def setup_radar(self, cli_port, data_port):
        """Initialize radar interface with specified ports"""
        try:
            self.radar = RadarInterface(cli_port, data_port)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to setup radar: {str(e)}")
            return False
    
    def run(self):
        """Main thread loop for reading radar data"""
        self.is_running = True
        while self.is_running:
            try:
                if self.radar and self.radar.is_connected():
                    data = self.radar.read_data()
                    if data is not None:
                        self.data_ready.emit(data)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.error_occurred.emit(f"Error reading data: {str(e)}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop the thread and close radar connection"""
        self.is_running = False
        if self.radar:
            self.radar.close()
        self.wait()

class RadarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IWR6843ISK Radar Visualization with Webcam")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.radar_thread = RadarDataThread()
        self.webcam_thread = WebcamThread()
        self.data_recorder = DataRecorder()
        
        # Initialize enhanced tracking system
        self.tracker = EnhancedHDBSCANTracker(
            min_cluster_size=3,
            min_samples=2,
            cluster_selection_epsilon=0.5,
            max_tracking_distance=2.0
        )
        
        # Initialize enhanced visualization (will be set up after GL widget creation)
        self.enhanced_visualization = None
        self.tracking_enabled = False
        self.tracking_results = None
        
        # Data storage
        self.point_cloud_data = None
        self.current_frame = None
        
        # Performance optimization variables
        self.last_visualization_update = 0
        self.visualization_update_interval = 50  # Start with 50ms (20 FPS)
        self.last_point_count = 0
        
        # FPS tracking
        self.frame_times = deque(maxlen=20)  # Track last 20 frame times
        self.last_fps_update = 0
        self.radar_fps = 0  # Track actual radar data reception rate
        self.viz_fps = 0    # Track visualization update rate
        
        self.dist = 70
        self.elev = 90
        self.azim = 270

        # Setup UI
        self.setup_ui()
        
        # Connect signals
        self.radar_thread.data_ready.connect(self.update_data)
        self.radar_thread.error_occurred.connect(self.handle_error)
        self.webcam_thread.frame_ready.connect(self.update_webcam_frame)
        self.webcam_thread.error_occurred.connect(self.handle_webcam_error)
        
        # Setup adaptive update timer for visualization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(self.visualization_update_interval)
        
        # IMPORTANT: Two separate systems:
        # 1. RadarDataThread receives data at full 20Hz (independent)
        # 2. This timer only affects GUI visualization refresh rate (adaptive)
        
        # Reduce range enforcement frequency for better performance
        self.range_timer = QTimer()
        self.range_timer.timeout.connect(self.enforce_2d_ranges)
        self.range_timer.start(500)  # Reduced from 100ms to 500ms
        
        # Ensure origin markers are positioned correctly after UI setup
        QTimer.singleShot(100, self.refresh_origin_markers)  # Call after 100ms delay
        
    def scan_available_cameras(self):
        """Scan for available cameras and return a list of camera indices and names"""
        available_cameras = []
        consecutive_failures = 0
        max_consecutive_failures = 2  # Stop after 2 consecutive failures
        
        # Suppress OpenCV error messages temporarily
        old_opencv_log_level = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
        
        try:
            # Test camera indices starting from 0
            for i in range(10):  # Maximum 10 cameras
                camera_found = False
                
                # Update status for user feedback
                self.log_status(f"Testing camera {i}...")
                QApplication.processEvents()  # Keep UI responsive
                
                # Try different backends for better compatibility
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                
                for backend in backends:
                    cap = None
                    try:
                        cap = cv2.VideoCapture(i, backend)
                        # Optimized settings for faster detection
                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 300)  # Shorter timeout
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Lower resolution for testing
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        
                        if cap.isOpened():
                            # Try to read a frame to confirm the camera works
                            ret, _ = cap.read()
                            if ret:
                                # Get actual camera resolution (after successful test)
                                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                
                                backend_name = {
                                    cv2.CAP_DSHOW: "DS",
                                    cv2.CAP_MSMF: "MSMF", 
                                    cv2.CAP_ANY: "Default"
                                }.get(backend, "Unknown")
                                
                                camera_name = f"Camera {i} ({int(width)}x{int(height)}, {backend_name})"
                                available_cameras.append((i, camera_name))
                                camera_found = True
                                consecutive_failures = 0  # Reset failure counter
                                self.log_status(f"✓ Found: {camera_name}")
                                break  # Found working backend, stop trying others
                                
                    except Exception as e:
                        # Continue to next backend
                        pass
                    finally:
                        if cap is not None:
                            cap.release()
                
                # Track consecutive failures to know when to stop
                if not camera_found:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        # Stop scanning after consecutive failures
                        self.log_status(f"Stopping scan after {max_consecutive_failures} consecutive failures")
                        break
                        
        finally:
            # Restore original OpenCV log level
            if old_opencv_log_level:
                os.environ['OPENCV_LOG_LEVEL'] = old_opencv_log_level
            else:
                os.environ.pop('OPENCV_LOG_LEVEL', None)
                
        return available_cameras
        
    def refresh_camera_list(self):
        """Refresh the camera dropdown with available cameras"""
        # Show scanning status
        self.refresh_cameras_btn.setText("Scanning...")
        self.refresh_cameras_btn.setEnabled(False)
        self.camera_dropdown.clear()
        self.camera_dropdown.addItem("Scanning for cameras...", -1)
        
        # Force UI update
        QApplication.processEvents()
        
        # Scan for available cameras
        cameras = self.scan_available_cameras()
        
        # Clear scanning indicator
        self.camera_dropdown.clear()
        
        if cameras:
            for camera_index, camera_name in cameras:
                self.camera_dropdown.addItem(camera_name, camera_index)
            self.log_status(f"Found {len(cameras)} camera(s)")
        else:
            self.camera_dropdown.addItem("No cameras found", -1)
            self.log_status("No cameras detected")
        
        # Restore button
        self.refresh_cameras_btn.setText("Refresh Cameras")
        self.refresh_cameras_btn.setEnabled(True)
            
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right panel - Visualization
        viz_panel = self.create_visualization_panel()
        main_layout.addWidget(viz_panel, 3)
        
    def create_control_panel(self):
        """Create the control panel with tabs for main controls and object tracking"""
        panel = QGroupBox("Controls")
        main_layout = QVBoxLayout()
        
        # Create tab widget for control panel
        self.control_tab_widget = QTabWidget()
        
        # Main Controls Tab
        main_controls_tab = self.create_main_controls_tab()
        self.control_tab_widget.addTab(main_controls_tab, "Main Controls")
        
        # Object Tracking Tab
        tracking_tab = self.create_tracking_tab()
        self.control_tab_widget.addTab(tracking_tab, "Object Tracking")
        
        main_layout.addWidget(self.control_tab_widget)
        panel.setLayout(main_layout)
        return panel
    
    def create_main_controls_tab(self):
        """Create the main controls tab with port, webcam, recording, and status"""
        main_tab = QWidget()
        layout = QVBoxLayout()
        
        # Port configuration
        port_group = QGroupBox("Port Configuration")
        port_layout = QGridLayout()
        
        port_layout.addWidget(QLabel("CLI Port:"), 0, 0)
        self.cli_port_input = QLineEdit("COM3")  # Enhanced COM Port for CLI
        port_layout.addWidget(self.cli_port_input, 0, 1)
        
        port_layout.addWidget(QLabel("Data Port:"), 1, 0)
        self.data_port_input = QLineEdit("COM4")  # Standard COM Port for Data
        port_layout.addWidget(self.data_port_input, 1, 1)
        
        port_group.setLayout(port_layout)
        layout.addWidget(port_group)
        
        # Webcam configuration
        webcam_group = QGroupBox("Webcam Configuration")
        webcam_layout = QGridLayout()
        
        webcam_layout.addWidget(QLabel("Camera Selection:"), 0, 0)
        self.camera_dropdown = QComboBox()
        self.camera_dropdown.setMinimumWidth(200)
        webcam_layout.addWidget(self.camera_dropdown, 0, 1)
        
        # Refresh cameras button
        self.refresh_cameras_btn = QPushButton("Refresh Cameras")
        self.refresh_cameras_btn.clicked.connect(self.refresh_camera_list)
        webcam_layout.addWidget(self.refresh_cameras_btn, 0, 2)
        
        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.toggle_webcam)
        webcam_layout.addWidget(self.webcam_btn, 1, 0, 1, 3)
        
        webcam_group.setLayout(webcam_layout)
        layout.addWidget(webcam_group)
        
        # Recording configuration
        recording_group = QGroupBox("Recording & Playback")
        recording_layout = QGridLayout()
        
        # Recording mode selection
        recording_layout.addWidget(QLabel("Record Mode:"), 0, 0)
        self.recording_mode_combo = QComboBox()
        self.recording_mode_combo.addItems(["Both (Radar + Camera)", "Point Cloud Only", "Camera Only"])
        self.recording_mode_combo.currentTextChanged.connect(self.on_recording_mode_changed)
        recording_layout.addWidget(self.recording_mode_combo, 0, 1, 1, 2)
        
        # Recording controls
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        recording_layout.addWidget(self.record_btn, 1, 0, 1, 2)
        
        self.save_recording_btn = QPushButton("💾 Save Recording")
        self.save_recording_btn.clicked.connect(self.save_recording)
        self.save_recording_btn.setEnabled(False)
        recording_layout.addWidget(self.save_recording_btn, 1, 2)
        
        # Playback controls
        self.load_recording_btn = QPushButton("📁 Load Recording")
        self.load_recording_btn.clicked.connect(self.load_recording)
        recording_layout.addWidget(self.load_recording_btn, 2, 0)
        
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        recording_layout.addWidget(self.play_btn, 2, 1)
        
        self.stop_playback_btn = QPushButton("⏹️ Stop")
        self.stop_playback_btn.clicked.connect(self.stop_playback)
        self.stop_playback_btn.setEnabled(False)
        recording_layout.addWidget(self.stop_playback_btn, 2, 2)
        
        # Playback speed
        recording_layout.addWidget(QLabel("Speed:"), 3, 0)
        self.playback_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_speed_slider.setMinimum(1)
        self.playback_speed_slider.setMaximum(50)  # 0.1x to 5.0x speed
        self.playback_speed_slider.setValue(10)  # 1.0x speed
        self.playback_speed_slider.valueChanged.connect(self.on_playback_speed_changed)
        recording_layout.addWidget(self.playback_speed_slider, 3, 1)
        
        self.playback_speed_label = QLabel("1.0x")
        recording_layout.addWidget(self.playback_speed_label, 3, 2)
        
        recording_group.setLayout(recording_layout)
        layout.addWidget(recording_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        button_layout.addWidget(self.connect_btn)
        
        self.config_btn = QPushButton("Send Configuration")
        self.config_btn.clicked.connect(self.send_configuration)
        self.config_btn.setEnabled(False)
        button_layout.addWidget(self.config_btn)
        
        self.start_btn = QPushButton("Start Sensor")
        self.start_btn.clicked.connect(self.toggle_sensor)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        # Debug mode checkbox
        self.debug_checkbox = QPushButton("Debug Mode: OFF")
        self.debug_checkbox.setCheckable(True)
        self.debug_checkbox.toggled.connect(self.toggle_debug_mode)
        button_layout.addWidget(self.debug_checkbox)
        
        layout.addLayout(button_layout)
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.points_label = QLabel("Points: 0")
        self.fps_label = QLabel("FPS: 0")
        self.webcam_status_label = QLabel("Webcam: Disconnected")
        self.recording_status_label = QLabel("Recording: Stopped")
        self.playback_status_label = QLabel("Playback: No file loaded")
        self.performance_status_label = QLabel("Performance: Normal")
        self.tracking_status_label = QLabel("Tracking: Disabled")
        self.tracks_label = QLabel("Tracks: 0")
        
        stats_layout.addWidget(self.points_label)
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.webcam_status_label)
        stats_layout.addWidget(self.recording_status_label)
        stats_layout.addWidget(self.playback_status_label)
        stats_layout.addWidget(self.performance_status_label)
        stats_layout.addWidget(self.tracking_status_label)
        stats_layout.addWidget(self.tracks_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        main_tab.setLayout(layout)
        return main_tab
    
    def create_tracking_tab(self):
        """Create the object tracking tab with tracking controls and parameters"""
        tracking_tab = QWidget()
        layout = QVBoxLayout()
        
        # Tracking controls
        tracking_group = QGroupBox("Object Tracking Controls")
        tracking_layout = QGridLayout()
        
        self.tracking_btn = QPushButton("Enable Tracking")
        self.tracking_btn.setCheckable(True)
        self.tracking_btn.toggled.connect(self.toggle_tracking)
        tracking_layout.addWidget(self.tracking_btn, 0, 0, 1, 2)
        
        # Tracking parameters
        tracking_layout.addWidget(QLabel("Min Cluster Size:"), 1, 0)
        self.min_cluster_size_spin = QSpinBox()
        self.min_cluster_size_spin.setMinimum(2)
        self.min_cluster_size_spin.setMaximum(20)
        self.min_cluster_size_spin.setValue(3)
        self.min_cluster_size_spin.valueChanged.connect(self.update_tracking_params)
        tracking_layout.addWidget(self.min_cluster_size_spin, 1, 1)
        
        tracking_layout.addWidget(QLabel("Max Track Distance:"), 2, 0)
        self.max_track_distance_spin = QSpinBox()
        self.max_track_distance_spin.setMinimum(1)
        self.max_track_distance_spin.setMaximum(10)
        self.max_track_distance_spin.setValue(2)
        self.max_track_distance_spin.valueChanged.connect(self.update_tracking_params)
        tracking_layout.addWidget(self.max_track_distance_spin, 2, 1)
        
        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)
        
        # Advanced tracking parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QGridLayout()
        
        advanced_layout.addWidget(QLabel("Min Samples:"), 0, 0)
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setMinimum(1)
        self.min_samples_spin.setMaximum(20)
        self.min_samples_spin.setValue(2)
        self.min_samples_spin.valueChanged.connect(self.update_tracking_params)
        advanced_layout.addWidget(self.min_samples_spin, 0, 1)
        
        advanced_layout.addWidget(QLabel("Cluster Epsilon:"), 1, 0)
        self.cluster_epsilon_spin = QSpinBox()
        self.cluster_epsilon_spin.setMinimum(1)
        self.cluster_epsilon_spin.setMaximum(50)
        self.cluster_epsilon_spin.setValue(5)  # Represents 0.5 (value/10)
        self.cluster_epsilon_spin.setSuffix(" (×0.1)")
        self.cluster_epsilon_spin.valueChanged.connect(self.update_tracking_params)
        advanced_layout.addWidget(self.cluster_epsilon_spin, 1, 1)
        
        advanced_layout.addWidget(QLabel("Min Confidence:"), 2, 0)
        self.min_confidence_spin = QSpinBox()
        self.min_confidence_spin.setMinimum(1)
        self.min_confidence_spin.setMaximum(100)
        self.min_confidence_spin.setValue(30)  # Represents 0.3 (value/100)
        self.min_confidence_spin.setSuffix(" (%)")
        self.min_confidence_spin.valueChanged.connect(self.update_tracking_params)
        advanced_layout.addWidget(self.min_confidence_spin, 2, 1)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Tracking information
        info_group = QGroupBox("Tracking Information")
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            "HDBSCAN + Particle Filter Tracking:\n\n"
            "• Min Cluster Size: Minimum points to form a cluster\n"
            "• Max Track Distance: Maximum distance for data association\n"
            "• Min Samples: Minimum samples for HDBSCAN core points\n"
            "• Cluster Epsilon: Distance threshold for cluster selection\n"
            "• Min Confidence: Minimum confidence for valid detections\n\n"
            "Tracked objects appear as colored markers in the 3D visualization."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        tracking_tab.setLayout(layout)
        return tracking_tab
        
    def create_visualization_panel(self):
        """Create the visualization panel with tabs for radar and combined view"""
        panel = QGroupBox("Visualization")
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Radar tab (3D only)
        radar_tab = self.create_radar_tab()
        self.tab_widget.addTab(radar_tab, "Radar Data (3D)")
        
        # Combined view tab (2D + Webcam)
        combined_tab = self.create_combined_tab()
        self.tab_widget.addTab(combined_tab, "Combined View (2D + Camera)")
        
        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel
    
    def create_radar_tab(self):
        """Create the radar visualization tab with 3D view only"""
        radar_widget = QWidget()
        layout = QVBoxLayout()
        
        # === 3D View Only ===
        self.plot_widget = gl.GLViewWidget()
        self.plot_widget.setCameraPosition(distance=50)
        
        # Add grid
        grid = gl.GLGridItem()
        grid.scale(5, 5, 1)
        self.plot_widget.addItem(grid)
        
        # Add axes
        axis = gl.GLAxisItem()
        self.plot_widget.addItem(axis)
        
        # Create scatter plot for point cloud
        self.point_scatter = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            color=(1, 1, 1, 0.5),
            size=0.1
        )
        self.plot_widget.addItem(self.point_scatter)
        
        # Create scatter plot for tracked objects (centroids)
        self.track_scatter = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            color=(1, 0, 0, 1.0),  # Red for tracked objects
            size=0.3
        )
        self.plot_widget.addItem(self.track_scatter)
        
        # Add 3D view directly to layout (no splitter)
        layout.addWidget(self.plot_widget)
        radar_widget.setLayout(layout)
        return radar_widget
    

    
    def create_combined_tab(self):
        """Create the combined view tab with 3D radar visualization and webcam feed"""
        combined_widget = QWidget()
        layout = QHBoxLayout()
        
        # Left side - 3D Radar Visualization  
        radar_side = QWidget()
        radar_layout = QVBoxLayout()
        radar_layout.addWidget(QLabel("3D Radar View (Bird's Eye)"))
        
        # Create 3D plot for combined view
        self.setup_3d_radar_plot_combined()
        radar_layout.addWidget(self.plot_3d_combined)
        radar_side.setLayout(radar_layout)
        
        # Right side - Webcam Feed
        webcam_side = QWidget()
        webcam_layout = QVBoxLayout()
        webcam_layout.addWidget(QLabel("Camera Feed"))
        
        self.webcam_label_combined = QLabel("Webcam feed will appear here")
        self.webcam_label_combined.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label_combined.setMinimumSize(400, 300)
        self.webcam_label_combined.setStyleSheet("border: 2px solid gray; background-color: black; color: white; font-size: 14px;")
        
        webcam_layout.addWidget(self.webcam_label_combined)
        webcam_side.setLayout(webcam_layout)
        
        # Add both sides to main layout (65% radar, 35% webcam)
        layout.addWidget(radar_side, 65)  # Radar takes 65% of space
        layout.addWidget(webcam_side, 35)  # Webcam takes 35% of space
        
        combined_widget.setLayout(layout)
        return combined_widget
    
    def setup_3d_radar_plot_combined(self):
        """Setup a 3D radar plot for combined view with bird's eye perspective"""
        # Create the 3D plot widget
        self.plot_3d_combined = gl.GLViewWidget()
        
        # Set the camera position for bird's eye view (z=0 plane, azimuth rotated 90 degrees counter clockwise)
        # Position camera above the scene looking down, with y=0 at bottom
        
        self.plot_3d_combined.setCameraPosition(
            distance=self.dist,
            elevation=self.elev,  # Look straight down (90 degrees from horizontal)
            azimuth=self.azim,    # 90 degrees counter clockwise rotation around z-axis
           
        )
        
        # Add grid (keeping existing style)
        grid = gl.GLGridItem()
        grid.scale(5, 5, 1)
        self.plot_3d_combined.addItem(grid)
        
        # Add axes (keeping existing style)
        axis = gl.GLAxisItem()
        self.plot_3d_combined.addItem(axis)
        
        # Create scatter plot for point cloud (keeping existing style)
        self.point_scatter_3d_combined = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            color=(1, 1, 1, 0.5),
            size=0.1
        )
        self.plot_3d_combined.addItem(self.point_scatter_3d_combined)
        
        # Create scatter plot for tracked objects in combined view
        self.track_scatter_3d_combined = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            color=(1, 0, 0, 1.0),  # Red for tracked objects
            size=0.3
        )
        self.plot_3d_combined.addItem(self.track_scatter_3d_combined)
        
    def refresh_origin_markers(self):
        """Refresh 3D plot view settings (keeping bird's eye perspective)"""
        if hasattr(self, 'plot_3d_combined'):
            # Ensure the camera maintains the bird's eye view
            self.plot_3d_combined.setCameraPosition(
                distance=self.dist,
                elevation=self.elev,  # Look straight down (90 degrees from horizontal)
                azimuth=self.azim     # 90 degrees counter clockwise rotation around z-axis
            )
    def toggle_connection(self):
        """Connect or disconnect from the radar"""
        if self.connect_btn.text() == "Connect":
            cli_port = self.cli_port_input.text()
            data_port = self.data_port_input.text()
            
            if self.radar_thread.setup_radar(cli_port, data_port):
                self.radar_thread.start()
                self.connect_btn.setText("Disconnect")
                self.config_btn.setEnabled(True)
                self.log_status(f"Connected to CLI: {cli_port}, Data: {data_port}")
            else:
                self.log_status("Failed to connect to radar")
        else:
            self.radar_thread.stop()
            self.connect_btn.setText("Connect")
            self.config_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.log_status("Disconnected from radar")
            
    def send_configuration(self):
        """Send configuration to the radar"""
        if self.radar_thread.radar:
            result = self.radar_thread.radar.send_config()
            
            # Handle the new return format (tuple)
            if isinstance(result, tuple):
                success, response = result
            else:
                # Fallback for old format
                success = result
                response = None
            
            if success:
                self.start_btn.setEnabled(True)
                self.log_status("Configuration sent successfully")
            else:
                self.log_status("Failed to send configuration")
                if response:
                    self.log_status(f"Error details: {response}")
                
    def toggle_sensor(self):
        """Start or stop the sensor"""
        if self.start_btn.text() == "Start Sensor":
            if self.radar_thread.radar:
                result = self.radar_thread.radar.start_sensor()
                
                # Handle the new return format (tuple)
                if isinstance(result, tuple):
                    success, response = result
                else:
                    # Fallback for old format
                    success = True
                    response = None
                
                if success:
                    self.start_btn.setText("Stop Sensor")
                    self.log_status("Sensor started")
                else:
                    self.log_status("Failed to start sensor")
                    if response:
                        self.log_status(f"Error: {response}")
        else:
            if self.radar_thread.radar:
                result = self.radar_thread.radar.stop_sensor()
                
                # Handle the new return format (tuple)
                if isinstance(result, tuple):
                    success, response = result
                else:
                    # Fallback for old format
                    success = True
                    response = None
                
                if success:
                    self.start_btn.setText("Start Sensor")
                    self.log_status("Sensor stopped")
                else:
                    self.log_status("Failed to stop sensor")
                    if response:
                        self.log_status(f"Error: {response}")
                
    def toggle_debug_mode(self, checked):
        """Toggle debug mode for point cloud data"""
        if checked:
            self.debug_checkbox.setText("Debug Mode: ON")
            self.log_status("Debug mode enabled - Point cloud data will be printed to terminal")
            print("\n=== Point Cloud Debug Mode Enabled ===")
            print("Format: Frame# | Point# | X(m) | Y(m) | Z(m) | Doppler(m/s)")
            print("=" * 60)
        else:
            self.debug_checkbox.setText("Debug Mode: OFF")
            self.log_status("Debug mode disabled")
            print("\n=== Point Cloud Debug Mode Disabled ===\n")
    
    def toggle_tracking(self, checked):
        """Toggle object tracking on/off"""
        self.tracking_enabled = checked
        if checked:
            self.tracking_btn.setText("Disable Tracking")
            self.log_status("Object tracking enabled - HDBSCAN clustering + Particle Filter")
            print("\n=== Object Tracking Enabled ===")
            print("Using HDBSCAN clustering with Particle Filter tracking")
        else:
            self.tracking_btn.setText("Enable Tracking")
            self.log_status("Object tracking disabled")
            print("\n=== Object Tracking Disabled ===")
            # Clear tracking results
            self.tracking_results = None
            
    def update_tracking_params(self):
        """Update tracking algorithm parameters"""
        if hasattr(self, 'tracker'):
            # Update enhanced tracker parameters using the new API
            params = {
                'min_cluster_size': self.min_cluster_size_spin.value(),
                'max_tracking_distance': float(self.max_track_distance_spin.value())
            }
            
            # Add advanced parameters if they exist
            if hasattr(self, 'min_samples_spin'):
                params['min_samples'] = self.min_samples_spin.value()
            if hasattr(self, 'cluster_epsilon_spin'):
                params['cluster_selection_epsilon'] = self.cluster_epsilon_spin.value() / 10.0
            
            self.tracker.update_parameters(**params)
            
            self.log_status(f"Updated tracking params: min_cluster_size={params['min_cluster_size']}, "
                          f"max_distance={params['max_tracking_distance']}")
            print(f"TRACKING: Updated parameters - {params}")
    

    
    def toggle_webcam(self):
        """Start or stop webcam capture"""
        if self.webcam_btn.text() == "Start Webcam":
            # Get camera index from dropdown data
            camera_index = self.camera_dropdown.currentData()
            
            if camera_index is None or camera_index == -1:
                self.log_status("No valid camera selected")
                return
                
            if self.webcam_thread.setup_camera(camera_index):
                self.webcam_thread.start()
                self.webcam_btn.setText("Stop Webcam")
                self.webcam_status_label.setText("Webcam: Connected")
                self.log_status(f"Webcam started ({self.camera_dropdown.currentText()})")
            else:
                self.log_status("Failed to start webcam")
        else:
            self.webcam_thread.stop()
            self.webcam_btn.setText("Start Webcam")
            self.webcam_status_label.setText("Webcam: Disconnected")
            self.webcam_label_combined.setText("Webcam feed will appear here")
            self.log_status("Webcam stopped")
    
    @Slot(np.ndarray)
    def update_webcam_frame(self, frame):
        """Update webcam display with new frame"""
        self.current_frame = frame
        
        # Record camera frame if recording is active (non-blocking)
        if self.data_recorder.is_recording:
            # The frame from webcam is already in RGB format, but OpenCV expects BGR
            # So we need to convert RGB to BGR for proper recording
            # Use a copy to avoid blocking the UI thread
            frame_copy = frame.copy()
            frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            # Use threaded recording to avoid blocking UI
            self.data_recorder.add_camera_frame_async(frame_bgr)
        
        # Only update displays if playback is NOT active to avoid conflicts
        if not self.data_recorder.is_playing:
            # Convert numpy array to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            
            # Update the combined view webcam display
            scaled_pixmap_combined = pixmap.scaled(
                self.webcam_label_combined.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.webcam_label_combined.setPixmap(scaled_pixmap_combined)
    
    def _update_webcam_displays_with_frame(self, frame_rgb):
        """Update webcam displays directly with a frame (used for playback)"""
        # Convert numpy array to QImage - frame should already be in RGB format
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        
        # Update the combined view webcam display
        scaled_pixmap_combined = pixmap.scaled(
            self.webcam_label_combined.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.webcam_label_combined.setPixmap(scaled_pixmap_combined)

    @Slot(str)
    def handle_webcam_error(self, error_msg):
        """Handle webcam errors"""
        self.log_status(f"Webcam error: {error_msg}")
        self.webcam_status_label.setText("Webcam: Error")
                
    @Slot(dict)
    def update_data(self, data):
        """Update with new radar data"""
        
        # FPS tracking for performance monitoring
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # DEBUG: Log all received data to understand what's happening
        frame_num = data.get('header', {}).get('frameNumber', 'Unknown')
        pointcloud_data = data.get('pointCloud', [])
        # Handle both numpy arrays and lists properly
        if pointcloud_data is not None:
            if hasattr(pointcloud_data, 'shape'):  # numpy array
                num_points = pointcloud_data.shape[0]
            else:  # list
                num_points = len(pointcloud_data)
        else:
            num_points = 0
        
        # DEBUG: Always log frame reception during recording
        if self.data_recorder.is_recording:
            if num_points > 0:  # Only log frames with actual data during recording
                print(f"DEBUG: [RECORDING] Received frame {frame_num} with {num_points} points")
        
        # Log frames with point cloud data for monitoring
        if num_points > 0:
            pass  # Removed verbose logging
        
        # Record radar data if recording is active
        if self.data_recorder.is_recording:
            if num_points > 0:  # Only log when we have data to record
                print(f"DEBUG: [RECORDING] Processing frame {frame_num} with {num_points} points")
            
            # Format data for recording
            radar_data = {
                "error": 0,
                "frameNum": data.get('header', {}).get('frameNumber', 0),
                "pointCloud": data.get('pointCloud', []),
                "numDetectedPoints": len(data.get('pointCloud', [])),
                "numDetectedTracks": 0,
                "trackData": [],
                "trackIndexes": []
            }
            
            self.data_recorder.add_radar_frame(radar_data)
            
            # Log recording stats periodically
            stats = self.data_recorder.get_recording_stats()
            if stats['frame_count'] % 20 == 0:  # Every 20 frames
                print(f"DEBUG: [RECORDING] Stats - {stats['frame_count']} total frames, {stats['radar_frames']} radar frames")
        
        # Extract point cloud
        if 'pointCloud' in data:
            # Ensure point cloud data is a numpy array for consistent handling
            if data['pointCloud'] is not None and len(data['pointCloud']) > 0:
                self.point_cloud_data = np.array(data['pointCloud'])
                
                # Process tracking if enabled
                if self.tracking_enabled and self.point_cloud_data is not None:
                    try:
                        self.tracking_results = self.tracker.process_frame(self.point_cloud_data, current_time)
                        
                        # Log tracking info periodically
                        if self.tracking_results['frame_stats']['frame_count'] % 20 == 0:
                            n_tracks = self.tracking_results['frame_stats']['active_tracks']
                            n_detections = len(self.tracking_results['detections'])
                            proc_time = self.tracking_results['processing_time'] * 1000
                            print(f"TRACKING: Frame {self.tracking_results['frame_stats']['frame_count']}: "
                                  f"{n_tracks} tracks, {n_detections} detections, {proc_time:.1f}ms")
                                  
                    except Exception as e:
                        print(f"TRACKING ERROR: {e}")
                        self.tracking_results = None
            else:
                self.point_cloud_data = None
                if self.tracking_enabled:
                    # Process empty frame for tracking (predictions only)
                    try:
                        self.tracking_results = self.tracker.process_frame(np.array([]), current_time)
                    except Exception as e:
                        print(f"TRACKING ERROR (empty frame): {e}")
                        self.tracking_results = None
            
            # Debug mode: print point cloud data to terminal
            if self.debug_checkbox.isChecked() and self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
                frame_num = data.get('header', {}).get('frameNumber', 'N/A')
                point_count = len(self.point_cloud_data)
                
                # Limit debug output for large point clouds to prevent performance issues
                if point_count > 50:
                    print(f"\nFrame {frame_num}: {point_count} points (showing first 10 for performance)")
                    print("-" * 60)
                    display_points = self.point_cloud_data[:10]  # Only show first 10 points
                else:
                    print(f"\nFrame {frame_num}: {point_count} points")
                    print("-" * 60)
                    display_points = self.point_cloud_data
                
                for i, point in enumerate(display_points):
                    if len(point) >= 4:
                        # Include doppler if available
                        print(f"Point {i:3d}: X={point[0]:7.3f}m  Y={point[1]:7.3f}m  Z={point[2]:7.3f}m  Doppler={point[3]:6.3f}m/s")
                    else:
                        # Fallback for old format without doppler
                        print(f"Point {i:3d}: X={point[0]:7.3f}m  Y={point[1]:7.3f}m  Z={point[2]:7.3f}m")
                
                if point_count > 50:
                    print(f"... and {point_count - 10} more points")
                print(f"Total points: {point_count}")
            
            # Update statistics (including FPS)
            num_points = len(self.point_cloud_data) if self.point_cloud_data is not None else 0
            self.points_label.setText(f"Points: {num_points}")
            
            # Update tracking statistics
            if self.tracking_enabled and self.tracking_results:
                self.tracking_status_label.setText("Tracking: Active")
                n_tracks = self.tracking_results.get('frame_stats', {}).get('active_tracks', 0)
                n_detections = len(self.tracking_results.get('detections', []))
                self.tracks_label.setText(f"Tracks: {n_tracks} | Detections: {n_detections}")
            else:
                self.tracking_status_label.setText("Tracking: Disabled" if not self.tracking_enabled else "Tracking: No Data")
                self.tracks_label.setText("Tracks: 0")
            
            # Calculate and update FPS (less frequently for performance)
            if current_time - self.last_fps_update > 1.0:  # Update FPS every second
                if len(self.frame_times) > 1:
                    time_span = self.frame_times[-1] - self.frame_times[0]
                    if time_span > 0:
                        # This tracks RADAR DATA RECEPTION rate (should be ~20 Hz)
                        self.radar_fps = (len(self.frame_times) - 1) / time_span
                        # Visualization rate is separate and adaptive
                        self.viz_fps = 1000 / self.visualization_update_interval
                        
                        self.fps_label.setText(f"Radar: {self.radar_fps:.1f} FPS | Viz: {self.viz_fps:.1f} FPS")
                    else:
                        self.fps_label.setText("FPS: Calculating...")
                else:
                    self.fps_label.setText("Radar: 0 FPS | Viz: 0 FPS")
                self.last_fps_update = current_time

    def adjust_update_rate_for_performance(self, point_count):
        """
        Dynamically adjust VISUALIZATION update rate based on point cloud size for better performance.
        
        IMPORTANT: This ONLY affects how often the GUI plots are refreshed, NOT radar data reception.
        Radar data continues to be received at the full 20Hz rate in the background thread.
        The data is stored immediately and the latest data is used for visualization.
        """
        if point_count <= 50:
            # Small point clouds: 20 FPS (50ms)
            new_interval = 50
            performance_status = "Performance: Normal (20 FPS)"
        elif point_count <= 100:
            # Medium point clouds: 15 FPS (67ms)
            new_interval = 67
            performance_status = "Performance: Optimized (15 FPS)"
        elif point_count <= 170:
            # Large point clouds: 10 FPS (100ms)
            new_interval = 100
            performance_status = "Performance: Adaptive (10 FPS)"
        else:
            # Very large point clouds: 5 FPS (200ms)
            new_interval = 200
            performance_status = "Performance: High Load (5 FPS)"
            
        # Update performance status
        self.performance_status_label.setText(performance_status)
            
        # Only change timer if interval changed significantly
        if abs(new_interval - self.visualization_update_interval) > 10:
            self.visualization_update_interval = new_interval
            self.update_timer.stop()
            self.update_timer.start(new_interval)
            
            fps = 1000 / new_interval
            self.log_status(f"Adjusted VISUALIZATION to {fps:.1f} FPS for {point_count} points (data reception unaffected)")

    def update_visualization(self):
        """
        Update both 3D and 2D visualizations with performance optimizations.
        
        NOTE: This method only handles GUI rendering. Radar data reception happens 
        independently in RadarDataThread at full 20Hz rate. This method reads the 
        latest received data and updates the display at an adaptive rate.
        """
        current_time = time.time() * 1000  # milliseconds
        
        # Handle playback if active
        if self.data_recorder.is_playing:
            playback_frame = self.data_recorder.get_next_playback_frame()
            if playback_frame:
                # Handle radar data from playback
                if "frameData" in playback_frame:
                    frame_data = playback_frame["frameData"]
                    if frame_data.get("pointCloud"):
                        self.point_cloud_data = np.array(frame_data["pointCloud"])
                    
                # Handle camera data from playback - UPDATE DISPLAYS DIRECTLY
                if "cameraFrame" in playback_frame and "decoded_image" in playback_frame["cameraFrame"]:
                    camera_frame = playback_frame["cameraFrame"]["decoded_image"]
                    if camera_frame is not None:
                        # Update webcam displays directly to avoid conflict with live webcam
                        self._update_webcam_displays_with_frame(camera_frame)
                
                self.update_playback_status()
        
        # Update recording statistics (less frequently for performance)
        if self.data_recorder.is_recording and current_time - self.last_visualization_update > 200:
            stats = self.data_recorder.get_recording_stats()
            status = f"Recording: {stats['recording_mode']} mode - {stats['frame_count']} frames"
            if stats.get('duration_seconds'):
                status += f" ({stats['duration_seconds']:.1f}s)"
            self.recording_status_label.setText(status)
        else:
            if not self.data_recorder.is_recording:
                self.recording_status_label.setText("Recording: Stopped")
            
        # Update 3D and 2D point cloud visualization
        if self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
            # Convert to numpy array if it's a list (handles both live and playback data)
            if not isinstance(self.point_cloud_data, np.ndarray):
                points_array = np.array(self.point_cloud_data)
            else:
                points_array = self.point_cloud_data
            
            # Get current point count for performance adjustment
            current_point_count = len(points_array)
            
            # Adjust update rate based on point count (only check periodically)
            if abs(current_point_count - self.last_point_count) > 20:
                self.adjust_update_rate_for_performance(current_point_count)
                self.last_point_count = current_point_count
            
            # Extract only x,y,z for 3D visualization
            if len(points_array.shape) >= 2 and points_array.shape[1] >= 3:
                points_3d = points_array[:, :3]
            elif len(points_array.shape) >= 2 and points_array.shape[1] >= 2:
                # If only x,y available, add z=0
                points_3d = np.column_stack([points_array[:, :2], np.zeros(points_array.shape[0])])
            else:
                # Fallback: assume it's already in correct format
                points_3d = points_array
                
            # Update 3D visualization
            self.point_scatter.setData(pos=points_3d)
            
            # Update 3D point cloud for combined view  
            if hasattr(self, 'point_scatter_3d_combined'):
                self.point_scatter_3d_combined.setData(pos=points_3d)
        else:
            # No point cloud data, clear displays (but only occasionally for performance)
            if current_time - self.last_visualization_update > 500:  # Only clear every 500ms
                self.point_scatter.setData(pos=np.array([[0, 0, 0]]))
                
                # Clear 3D scatter plot for combined view
                if hasattr(self, 'point_scatter_3d_combined'):
                    self.point_scatter_3d_combined.setData(pos=np.array([[0, 0, 0]]))
        
        # Update enhanced tracking visualization if tracking is enabled
        if self.tracking_enabled and self.tracking_results and 'tracks' in self.tracking_results:
            tracks = self.tracking_results['tracks']
            if tracks:
                # Extract track positions and motion states
                track_positions = np.array([track['position'] for track in tracks])
                
                # Create colors based on motion state (as requested by user)
                motion_state_colors = {
                    'stationary': [1, 0, 0, 1],         # Red for stationary
                    'constant_velocity': [0, 1, 0, 1],  # Green for constant velocity
                    'maneuvering': [0, 0, 1, 1]         # Blue for maneuvering
                }
                
                # Create color array based on motion states
                track_colors = []
                for track in tracks:
                    motion_state = track.get('motion_state', 'constant_velocity')
                    color = motion_state_colors.get(motion_state, [1, 1, 1, 1])  # Default white
                    track_colors.append(color)
                
                track_colors = np.array(track_colors)
                
                # Use larger size for better visibility (as requested by user)
                track_size = 0.5
                
                # Update tracking scatter plots with enhanced visualization
                if hasattr(self, 'track_scatter'):
                    self.track_scatter.setData(pos=track_positions, color=track_colors, size=track_size)
                if hasattr(self, 'track_scatter_3d_combined'):
                    self.track_scatter_3d_combined.setData(pos=track_positions, color=track_colors, size=track_size)
            else:
                # Clear tracking displays when no tracks
                if hasattr(self, 'track_scatter'):
                    self.track_scatter.setData(pos=np.array([[0, 0, 0]]), color=np.array([[1, 0, 0, 0]]), size=0.1)
                if hasattr(self, 'track_scatter_3d_combined'):
                    self.track_scatter_3d_combined.setData(pos=np.array([[0, 0, 0]]), color=np.array([[1, 0, 0, 0]]), size=0.1)
        else:
            # Clear tracking displays when tracking disabled
            if hasattr(self, 'track_scatter'):
                self.track_scatter.setData(pos=np.array([[0, 0, 0]]), color=np.array([[1, 0, 0, 0]]), size=0.1)
            if hasattr(self, 'track_scatter_3d_combined'):
                self.track_scatter.setData(pos=np.array([[0, 0, 0]]), color=np.array([[1, 0, 0, 0]]), size=0.1)
        
        self.last_visualization_update = current_time
        
    def enforce_2d_ranges(self):
        """Enforce the correct camera view for combined 3D plot (optimized for performance)"""
        try:
            # Ensure the 3D combined view maintains bird's eye perspective
            if hasattr(self, 'plot_3d_combined'):
                # Refresh the camera position to maintain bird's eye view
                self.refresh_origin_markers()
                    
        except Exception:
            # Ignore any errors during view enforcement to prevent crashes
            pass

    def handle_error(self, error_msg):
        """Handle error messages from the radar thread"""
        self.log_status(f"ERROR: {error_msg}")
        
    def log_status(self, message):
        """Log a status message"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
    
    # Recording and Playback Methods
    def on_recording_mode_changed(self, mode_text):
        """Handle recording mode change"""
        mode_map = {
            "Both (Radar + Camera)": "both",
            "Point Cloud Only": "pointcloud", 
            "Camera Only": "camera"
        }
        mode = mode_map.get(mode_text, "both")
        self.data_recorder.set_recording_mode(mode)
        self.log_status(f"Recording mode set to: {mode}")
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.data_recorder.is_recording:
            # Get config commands if radar is connected
            config_commands = []
            if self.radar_thread.radar:
                try:
                    # Try to get config from file
                    config_file = 'xwr68xx_config.cfg'
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('%'):
                                    config_commands.append(line)
                except Exception:
                    pass
            
            print("DEBUG: ========== STARTING RECORDING ==========")
            print(f"DEBUG: Recording mode: {self.data_recorder.recording_mode}")
            print(f"DEBUG: Radar connected: {self.radar_thread.radar is not None}")
            print(f"DEBUG: Radar thread running: {self.radar_thread.is_running}")
            
            success, message = self.data_recorder.start_recording(config_commands)
            if success:
                self.record_btn.setText("⏹️ Stop Recording")
                self.save_recording_btn.setEnabled(False)
                self.log_status(message)
                print("DEBUG: Recording started successfully")
            else:
                self.log_status(f"Failed to start recording: {message}")
                print(f"DEBUG: Recording start failed: {message}")
        else:
            print("DEBUG: ========== STOPPING RECORDING ==========")
            
            # Log final stats before stopping
            stats = self.data_recorder.get_recording_stats()
            print(f"DEBUG: Final recording stats before stop:")
            print(f"DEBUG: - Frame count: {stats['frame_count']}")
            print(f"DEBUG: - Radar frames: {stats['radar_frames']}")
            print(f"DEBUG: - Duration: {stats.get('duration_seconds', 0):.2f}s")
            
            success, message = self.data_recorder.stop_recording()
            if success:
                self.record_btn.setText("🔴 Start Recording")
                self.save_recording_btn.setEnabled(True)
                self.log_status(message)
                print("DEBUG: Recording stopped successfully")
            else:
                self.log_status(f"Failed to stop recording: {message}")
                print(f"DEBUG: Recording stop failed: {message}")
    
    def save_recording(self):
        """Save current recording session"""
        if not self.data_recorder.session_folder:
            self.log_status("No recording session to save")
            return
        
        # Get custom name from user
        from PySide6.QtWidgets import QInputDialog
        custom_name, ok = QInputDialog.getText(
            self, 
            "Save Recording",
            "Enter a name for this recording:",
            text=f"recording_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        if ok and custom_name.strip():
            success, message = self.data_recorder.save_recording(custom_name.strip())
            self.log_status(message)
            if success:
                self.save_recording_btn.setEnabled(False)
        elif ok:
            # User clicked OK but didn't enter a name, use automatic name
            success, message = self.data_recorder.save_recording()
            self.log_status(message)
            if success:
                self.save_recording_btn.setEnabled(False)
    
    def load_recording(self):
        """Load recording session for playback"""
        # First try to use folder dialog
        recordings_dir = "recordings"
        if not os.path.exists(recordings_dir):
            # If no recordings folder, create it and inform user
            os.makedirs(recordings_dir, exist_ok=True)
            self.log_status("No recordings found. The 'recordings' folder has been created.")
            return
        
        # Get list of available recordings
        recordings = self.data_recorder.list_recordings()
        
        if not recordings:
            self.log_status("No recordings found in the recordings folder.")
            return
        
        # Show recordings in a selection dialog
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QLabel, QPushButton, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Recording")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Available Recordings:"))
        
        # Create list widget
        list_widget = QListWidget()
        for recording in recordings:
            info = recording["info"]
            name = recording["name"]
            mode = info.get("recording_mode", "unknown")
            duration = info.get("duration_seconds", 0)
            size_mb = recording["size_mb"]
            
            # Format display text
            display_text = f"{name} ({mode}) - {duration:.1f}s, {size_mb:.1f}MB"
            list_widget.addItem(display_text)
            list_widget.item(list_widget.count() - 1).setData(256, recording["path"])  # Store path as user data
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(load_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Connect signals
        def load_selected():
            current_item = list_widget.currentItem()
            if current_item:
                session_path = current_item.data(256)
                success, message = self.data_recorder.load_recording(session_path)
                self.log_status(message)
                if success:
                    self.play_btn.setEnabled(True)
                    self.update_playback_status()
                dialog.accept()
        
        def on_double_click():
            load_selected()
        
        load_button.clicked.connect(load_selected)
        cancel_button.clicked.connect(dialog.reject)
        list_widget.itemDoubleClicked.connect(on_double_click)
        
        dialog.exec()
    
    def toggle_playback(self):
        """Start or stop playback"""
        if not self.data_recorder.is_playing:
            speed = self.playback_speed_slider.value() / 10.0
            success, message = self.data_recorder.start_playback(speed)
            if success:
                self.play_btn.setText("⏸️ Pause")
                self.stop_playback_btn.setEnabled(True)
                self.log_status(message)
                if self.webcam_thread.is_running:
                    self.log_status("Note: Webcam display will show playback video during playback")
            else:
                self.log_status(f"Failed to start playback: {message}")
        else:
            success, message = self.data_recorder.stop_playback()
            if success:
                self.play_btn.setText("▶️ Play")
                self.stop_playback_btn.setEnabled(False)
                self.log_status(message)
    
    def stop_playback(self):
        """Stop playback"""
        success, message = self.data_recorder.stop_playback()
        if success:
            self.play_btn.setText("▶️ Play")
            self.stop_playback_btn.setEnabled(False)
            self.log_status(message)
            
            # Clear playback displays and restore to live webcam if running
            if self.webcam_thread.is_running:
                # Webcam will resume updating displays automatically since playback is stopped
                self.log_status("Switched back to live webcam feed")
            else:
                # No webcam running, clear displays
                self.webcam_label_combined.setText("Webcam feed will appear here")
    
    def on_playback_speed_changed(self, value):
        """Handle playback speed change"""
        speed = value / 10.0
        self.playback_speed_label.setText(f"{speed:.1f}x")
        if self.data_recorder.is_playing:
            self.data_recorder.playback_speed = speed
    
    def update_playback_status(self):
        """Update playback status in statistics"""
        info = self.data_recorder.get_playback_info()
        if info["loaded"]:
            session_info = info.get("session_info", {})
            mode = session_info.get("recording_mode", "unknown")
            status = f"Loaded: {info['total_frames']} frames ({mode})"
            if info["is_playing"]:
                status += f" (Playing {info['progress']:.1f}%)"
            self.playback_status_label.setText(f"Playback: {status}")
        else:
            self.playback_status_label.setText("Playback: No recording loaded")
        
    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.radar_thread.stop()
        self.webcam_thread.stop()
        
        # Stop timers
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'range_timer'):
            self.range_timer.stop()
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = RadarGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 