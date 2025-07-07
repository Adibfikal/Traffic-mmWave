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

# Import refactored modules
from threads import WebcamThread, RadarDataThread
from visualization import BoundingBoxManager, TrailManager, VisualizationCoordinator
# Tracking disabled - using basic point cloud visualization only
from data_recorder import DataRecorder
from radar_interface import RadarInterface
from gtrack_algorithm import GTRACKProcessor
from gtrack_no_snr import GTRACKNoSNR


class RadarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IWR6843ISK Radar Visualization with Webcam")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.radar_thread = RadarDataThread()
        self.webcam_thread = WebcamThread()
        self.data_recorder = DataRecorder()
        
        # GTRACK processor (No-SNR version for X,Y only data)
        self.gtrack_processor = GTRACKNoSNR(max_tracks=12)  # Realistic vehicle count
        self.gtrack_enabled = False
        self.gtrack_results = None
        
        # Track visualization data
        self.track_history = {}  # Store track position history for trails
        self.track_trail_length = 20  # Number of points in trail
        
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
        """Create the control panel with tabs for controls and status/statistics"""
        panel = QGroupBox("Control Panel")
        main_layout = QVBoxLayout()
        
        # Create tab widget for control panel
        self.control_tab_widget = QTabWidget()
        
        # Create controls tab
        controls_tab = self.create_controls_tab()
        self.control_tab_widget.addTab(controls_tab, "Controls")
        
        # Create status & statistics tab
        status_tab = self.create_status_statistics_tab()
        self.control_tab_widget.addTab(status_tab, "Status & Statistics")
        
        main_layout.addWidget(self.control_tab_widget)
        panel.setLayout(main_layout)
        return panel
    
    def create_controls_tab(self):
        """Create the controls tab with main controls"""
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
        
        # GTRACK tracking configuration
        tracking_group = QGroupBox("GTRACK Vehicle Tracking")
        tracking_layout = QGridLayout()
        
        # Enable/disable tracking
        self.gtrack_enable_checkbox = QCheckBox("Enable GTRACK Tracking")
        self.gtrack_enable_checkbox.toggled.connect(self.toggle_gtrack)
        tracking_layout.addWidget(self.gtrack_enable_checkbox, 0, 0, 1, 2)
        
        # Max tracks setting
        tracking_layout.addWidget(QLabel("Max Vehicles:"), 1, 0)
        self.max_tracks_spinbox = QSpinBox()
        self.max_tracks_spinbox.setMinimum(1)
        self.max_tracks_spinbox.setMaximum(50)
        self.max_tracks_spinbox.setValue(25)
        self.max_tracks_spinbox.valueChanged.connect(self.on_max_tracks_changed)
        tracking_layout.addWidget(self.max_tracks_spinbox, 1, 1)
        
        # Clustering sensitivity
        tracking_layout.addWidget(QLabel("Clustering:"), 2, 0)
        self.clustering_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.clustering_sensitivity_slider.setMinimum(1)
        self.clustering_sensitivity_slider.setMaximum(10)
        self.clustering_sensitivity_slider.setValue(5)
        self.clustering_sensitivity_slider.valueChanged.connect(self.on_clustering_changed)
        tracking_layout.addWidget(self.clustering_sensitivity_slider, 2, 1)
        
        # Reset tracking button
        self.reset_tracking_btn = QPushButton("Reset All Tracks")
        self.reset_tracking_btn.clicked.connect(self.reset_gtrack)
        self.reset_tracking_btn.setEnabled(False)
        tracking_layout.addWidget(self.reset_tracking_btn, 3, 0, 1, 2)
        
        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)
        
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
        
        main_tab.setLayout(layout)
        return main_tab
        
    def create_status_statistics_tab(self):
        """Create the status and statistics tab"""
        status_tab = QWidget()
        layout = QVBoxLayout()
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
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
        status_tab.setLayout(layout)
        return status_tab
    
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
        
        # Create track trails (line plots for track history)
        self.track_trails = {}  # Dictionary to store trail line plots for each track
        
        # Create bounding boxes for tracks
        self.track_boxes = {}  # Dictionary to store bounding box line plots for each track
        
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
        
        # Create track trails for combined view
        self.track_trails_combined = {}  # Dictionary to store trail line plots for each track
        
        # Create bounding boxes for combined view
        self.track_boxes_combined = {}  # Dictionary to store bounding box line plots for each track
        
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
    
    # GTRACK methods
    def toggle_gtrack(self, enabled):
        """Toggle GTRACK tracking on/off"""
        self.gtrack_enabled = enabled
        self.reset_tracking_btn.setEnabled(enabled)
        
        if enabled:
            self.tracking_status_label.setText("Tracking: GTRACK Enabled")
            self.log_status("GTRACK vehicle tracking enabled")
        else:
            self.tracking_status_label.setText("Tracking: Disabled")
            self.tracks_label.setText("Tracks: 0")
            self.gtrack_results = None
            self.log_status("GTRACK tracking disabled")
    
    def on_max_tracks_changed(self, value):
        """Handle changes to maximum tracks setting"""
        self.gtrack_processor = GTRACKProcessor(max_tracks=value)
        self.log_status(f"Maximum tracks set to: {value}")
    
    def on_clustering_changed(self, value):
        """Handle changes to clustering sensitivity"""
        # Adjust clustering parameters based on slider value (1-10)
        sensitivity = value / 5.0  # Convert to 0.2-2.0 range
        
        # Update clustering parameters for no-SNR version
        base_eps_map = {'near_range': 1.2, 'mid_range': 1.8, 'far_range': 2.5}
        for range_level in self.gtrack_processor.clustering_params:
            if range_level in base_eps_map:
                base_eps = base_eps_map[range_level]
                self.gtrack_processor.clustering_params[range_level]['eps'] = base_eps * sensitivity
        
        self.log_status(f"Clustering sensitivity: {value}/10")
    
    def reset_gtrack(self):
        """Reset all GTRACK tracks"""
        self.gtrack_processor = GTRACKNoSNR(max_tracks=min(self.max_tracks_spinbox.value(), 12))
        self.gtrack_results = None
        
        # Clear visualization data
        self.track_history = {}
        self._clear_track_visualization()
        
        self.tracks_label.setText("Tracks: 0")
        self.log_status("All GTRACK tracks reset")
    
    def _update_track_visualization(self, tracks):
        """Update track visualization with centroids, bounding boxes, and trails"""
        print(f"DEBUG: [VIZ] _update_track_visualization called with {len(tracks)} tracks")
        
        # Extract track positions for centroids
        track_positions = []
        current_track_ids = set()
        
        for track in tracks:
            track_id = track['id']
            x, y = track['position']
            state = track['state']
            conf = track['confidence']
            track_positions.append([x, y, 0])  # Add z=0 for 3D visualization
            current_track_ids.add(track_id)
            
            print(f"DEBUG: [VIZ] Track {track_id}: pos=({x:.2f},{y:.2f}), state={state}, conf={conf:.2f}")
            
            # Update track history for trails
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append([x, y, 0])
            
            # Limit trail length
            if len(self.track_history[track_id]) > self.track_trail_length:
                self.track_history[track_id] = self.track_history[track_id][-self.track_trail_length:]
        
        print(f"DEBUG: [VIZ] Processed {len(tracks)} tracks, sample positions: {track_positions[:3]}")
        
        # Update centroids
        if track_positions:
            track_positions = np.array(track_positions)
            print(f"DEBUG: [VIZ] Setting track_scatter data with {len(track_positions)} positions")
            print(f"DEBUG: [VIZ] Position array shape: {track_positions.shape}")
            print(f"DEBUG: [VIZ] Position range X: {np.min(track_positions[:,0]):.2f} to {np.max(track_positions[:,0]):.2f}")
            print(f"DEBUG: [VIZ] Position range Y: {np.min(track_positions[:,1]):.2f} to {np.max(track_positions[:,1]):.2f}")
            
            # Update scatter plots with more visible settings
            self.track_scatter.setData(pos=track_positions, color=(1, 0, 0, 1.0), size=1.0)  # Larger size
            if hasattr(self, 'track_scatter_3d_combined'):
                self.track_scatter_3d_combined.setData(pos=track_positions, color=(1, 0, 0, 1.0), size=1.0)
                print(f"DEBUG: [VIZ] Updated combined view track centroids")
            
            print(f"DEBUG: [VIZ] Successfully updated {len(track_positions)} track centroids")
        else:
            print(f"DEBUG: [VIZ] No track positions to display")
        
        # Update trails
        print(f"DEBUG: [VIZ] Updating trails for {len(current_track_ids)} tracks")
        self._update_track_trails(current_track_ids)
        
        # Update bounding boxes
        print(f"DEBUG: [VIZ] Updating bounding boxes for {len(tracks)} tracks")
        self._update_track_bounding_boxes(tracks)
        
        # Clean up old tracks
        self._cleanup_old_tracks(current_track_ids)
    
    def _update_track_trails(self, current_track_ids):
        """Update track trails (movement history)"""
        for track_id in current_track_ids:
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                trail_points = np.array(self.track_history[track_id])
                
                # Create or update trail line plot for main view
                if track_id not in self.track_trails:
                    trail_line = gl.GLLinePlotItem(
                        pos=trail_points,
                        color=(0, 1, 0, 0.8),  # Green trails
                        width=2.0,
                        antialias=True
                    )
                    self.plot_widget.addItem(trail_line)
                    self.track_trails[track_id] = trail_line
                    print(f"DEBUG: [VIZ] Created trail for track {track_id}")
                else:
                    self.track_trails[track_id].setData(pos=trail_points)
                
                # Create or update trail line plot for combined view
                if hasattr(self, 'plot_3d_combined'):
                    if track_id not in self.track_trails_combined:
                        trail_line_combined = gl.GLLinePlotItem(
                            pos=trail_points,
                            color=(0, 1, 0, 0.8),  # Green trails
                            width=2.0,
                            antialias=True
                        )
                        self.plot_3d_combined.addItem(trail_line_combined)
                        self.track_trails_combined[track_id] = trail_line_combined
                    else:
                        self.track_trails_combined[track_id].setData(pos=trail_points)
    
    def _update_track_bounding_boxes(self, tracks):
        """Update bounding boxes around tracks"""
        for track in tracks:
            track_id = track['id']
            x, y = track['position']
            vx, vy = track['velocity']
            
            # Calculate bounding box based on velocity and uncertainty
            # Larger box for faster moving objects
            speed = np.sqrt(vx**2 + vy**2)
            box_size = max(2.0, min(8.0, speed * 0.5 + 2.0))  # 2-8 meter box
            
            # Create box vertices
            box_vertices = np.array([
                [x - box_size/2, y - box_size/2, 0],
                [x + box_size/2, y - box_size/2, 0],
                [x + box_size/2, y + box_size/2, 0],
                [x - box_size/2, y + box_size/2, 0],
                [x - box_size/2, y - box_size/2, 0]  # Close the box
            ])
            
            # Create or update bounding box for main view
            if track_id not in self.track_boxes:
                box_line = gl.GLLinePlotItem(
                    pos=box_vertices,
                    color=(1, 1, 0, 0.8),  # Yellow bounding boxes
                    width=2.0,
                    antialias=True
                )
                self.plot_widget.addItem(box_line)
                self.track_boxes[track_id] = box_line
                print(f"DEBUG: [VIZ] Created bounding box for track {track_id}")
            else:
                self.track_boxes[track_id].setData(pos=box_vertices)
            
            # Create or update bounding box for combined view
            if hasattr(self, 'plot_3d_combined'):
                if track_id not in self.track_boxes_combined:
                    box_line_combined = gl.GLLinePlotItem(
                        pos=box_vertices,
                        color=(1, 1, 0, 0.8),  # Yellow bounding boxes
                        width=2.0,
                        antialias=True
                    )
                    self.plot_3d_combined.addItem(box_line_combined)
                    self.track_boxes_combined[track_id] = box_line_combined
                else:
                    self.track_boxes_combined[track_id].setData(pos=box_vertices)
    
    def _cleanup_old_tracks(self, current_track_ids):
        """Remove visualization elements for tracks that no longer exist"""
        # Clean up trails
        old_trail_ids = set(self.track_trails.keys()) - current_track_ids
        for track_id in old_trail_ids:
            self.plot_widget.removeItem(self.track_trails[track_id])
            del self.track_trails[track_id]
            print(f"DEBUG: [VIZ] Removed trail for old track {track_id}")
            
        old_trail_combined_ids = set(self.track_trails_combined.keys()) - current_track_ids
        for track_id in old_trail_combined_ids:
            if hasattr(self, 'plot_3d_combined'):
                self.plot_3d_combined.removeItem(self.track_trails_combined[track_id])
            del self.track_trails_combined[track_id]
        
        # Clean up bounding boxes
        old_box_ids = set(self.track_boxes.keys()) - current_track_ids
        for track_id in old_box_ids:
            self.plot_widget.removeItem(self.track_boxes[track_id])
            del self.track_boxes[track_id]
            print(f"DEBUG: [VIZ] Removed bounding box for old track {track_id}")
            
        old_box_combined_ids = set(self.track_boxes_combined.keys()) - current_track_ids
        for track_id in old_box_combined_ids:
            if hasattr(self, 'plot_3d_combined'):
                self.plot_3d_combined.removeItem(self.track_boxes_combined[track_id])
            del self.track_boxes_combined[track_id]
        
        # Clean up history
        old_history_ids = set(self.track_history.keys()) - current_track_ids
        for track_id in old_history_ids:
            del self.track_history[track_id]
    
    def _clear_track_visualization(self):
        """Clear all track visualization elements"""
        # Only clear if there are actually items to clear (avoid spam)
        items_to_clear = (len(self.track_trails) + len(self.track_boxes) + 
                         len(self.track_trails_combined) + len(self.track_boxes_combined) +
                         len(self.track_history))
        
        if items_to_clear > 0:
            print("DEBUG: [VIZ] Clearing all track visualization")
        
        # Clear centroids
        self.track_scatter.setData(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 0.0), size=0.1)
        if hasattr(self, 'track_scatter_3d_combined'):
            self.track_scatter_3d_combined.setData(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 0.0), size=0.1)
        
        # Clear all trails
        for trail_line in self.track_trails.values():
            self.plot_widget.removeItem(trail_line)
        self.track_trails.clear()
        
        for trail_line in self.track_trails_combined.values():
            if hasattr(self, 'plot_3d_combined'):
                self.plot_3d_combined.removeItem(trail_line)
        self.track_trails_combined.clear()
        
        # Clear all bounding boxes
        for box_line in self.track_boxes.values():
            self.plot_widget.removeItem(box_line)
        self.track_boxes.clear()
        
        for box_line in self.track_boxes_combined.values():
            if hasattr(self, 'plot_3d_combined'):
                self.plot_3d_combined.removeItem(box_line)
        self.track_boxes_combined.clear()
        
        # Clear history
        self.track_history.clear()
                
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
                
                # Process GTRACK tracking if enabled
                if self.gtrack_enabled and self.point_cloud_data is not None:
                    try:
                        # DEBUG: Log input data
                        frame_num = data.get('header', {}).get('frameNumber', 0)
                        print(f"DEBUG: [GTRACK INPUT] Frame {frame_num} - Processing {len(data['pointCloud'])} points")
                        
                        # Format data for GTRACK processing
                        frame_data = {
                            'timestamp': current_time * 1000,  # GTRACK expects milliseconds
                            'frameData': {
                                'error': 0,
                                'frameNum': frame_num,
                                'pointCloud': data['pointCloud']
                            }
                        }
                        
                        self.gtrack_results = self.gtrack_processor.process_frame(frame_data)
                        
                        # DEBUG: Always log GTRACK results to see what's happening
                        n_tracks = self.gtrack_results['frame_stats']['active_tracks']
                        n_detections = self.gtrack_results['frame_stats']['valid_detections']
                        total_detections = self.gtrack_results['frame_stats']['total_detections']
                        proc_time = self.gtrack_results['processing_time'] * 1000
                        
                        print(f"DEBUG: [GTRACK OUTPUT] Frame {frame_num}: "
                              f"{total_detections} raw -> {n_detections} valid -> {n_tracks} tracks, {proc_time:.1f}ms")
                        
                        # DEBUG: Log individual tracks
                        if self.gtrack_results.get('tracks'):
                            for track in self.gtrack_results['tracks']:
                                pos = track['position']
                                vel = track['velocity']
                                conf = track['confidence']
                                print(f"DEBUG: [TRACK] ID:{track['id']} pos:({pos[0]:.2f},{pos[1]:.2f}) "
                                      f"vel:({vel[0]:.2f},{vel[1]:.2f}) conf:{conf:.2f}")
                        else:
                            print("DEBUG: [GTRACK] No tracks detected")
                                  
                    except Exception as e:
                        print(f"DEBUG: [GTRACK ERROR] {e}")
                        import traceback
                        traceback.print_exc()
                        self.gtrack_results = None
            else:
                self.point_cloud_data = None
                if self.gtrack_enabled:
                    # Process empty frame for GTRACK (predictions only)
                    try:
                        frame_data = {
                            'timestamp': current_time * 1000,
                            'frameData': {
                                'error': 0,
                                'frameNum': data.get('header', {}).get('frameNumber', 0),
                                'pointCloud': []
                            }
                        }
                        self.gtrack_results = self.gtrack_processor.process_frame(frame_data)
                    except Exception as e:
                        print(f"GTRACK ERROR (empty frame): {e}")
                        self.gtrack_results = None
            
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
            
                    # Update GTRACK tracking statistics
        if self.gtrack_enabled and self.gtrack_results:
            self.tracking_status_label.setText("Tracking: GTRACK Active")
            n_tracks = self.gtrack_results.get('frame_stats', {}).get('active_tracks', 0)
            n_detections = self.gtrack_results.get('frame_stats', {}).get('valid_detections', 0)
            visible_tracks = self.gtrack_results.get('frame_stats', {}).get('visible_tracks', 0)
            total_result_tracks = len(self.gtrack_results.get('tracks', []))
            self.tracks_label.setText(f"Tracks: {n_tracks} active, {visible_tracks} visible, {total_result_tracks} in results | Detections: {n_detections}")
            
            print(f"DEBUG: [MAIN] GTRACK results received - Active:{n_tracks}, Visible:{visible_tracks}, Result tracks:{total_result_tracks}")
        else:
            self.tracking_status_label.setText("Tracking: Disabled" if not self.gtrack_enabled else "Tracking: No Data")
            self.tracks_label.setText("Tracks: 0")
            print(f"DEBUG: [MAIN] No GTRACK results - Enabled:{self.gtrack_enabled}, Results:{self.gtrack_results is not None}")
            
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
                        
                        # *** CRITICAL FIX: Process GTRACK during playback ***
                        if self.gtrack_enabled and self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
                            try:
                                print(f"DEBUG: [PLAYBACK GTRACK] Processing {len(self.point_cloud_data)} points from playback")
                                
                                # Format data for GTRACK processing (same as live data)
                                gtrack_frame_data = {
                                    'timestamp': current_time,  # Use current time for playback
                                    'frameData': {
                                        'error': 0,
                                        'frameNum': frame_data.get('frameNum', 0),
                                        'pointCloud': frame_data["pointCloud"]
                                    }
                                }
                                
                                self.gtrack_results = self.gtrack_processor.process_frame(gtrack_frame_data)
                                
                                # DEBUG: Log playback GTRACK results
                                n_tracks = self.gtrack_results['frame_stats']['active_tracks']
                                n_detections = self.gtrack_results['frame_stats']['valid_detections']
                                visible_tracks = self.gtrack_results['frame_stats'].get('visible_tracks', 0)
                                tracks_in_result = len(self.gtrack_results.get('tracks', []))
                                
                                print(f"DEBUG: [PLAYBACK GTRACK] Active:{n_tracks}, Visible:{visible_tracks}, Results:{tracks_in_result}, Detections:{n_detections}")
                                
                            except Exception as e:
                                print(f"DEBUG: [PLAYBACK GTRACK ERROR] {e}")
                                import traceback
                                traceback.print_exc()
                                self.gtrack_results = None
                
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
            
            # Update GTRACK tracks visualization with bounding boxes and trails
            if self.gtrack_enabled and self.gtrack_results and self.gtrack_results.get('tracks'):
                print(f"DEBUG: [VIZ] Updating visualization for {len(self.gtrack_results['tracks'])} tracks")
                self._update_track_visualization(self.gtrack_results['tracks'])
            else:
                # DEBUG: Detailed diagnosis of why tracks aren't showing
                if not self.gtrack_enabled:
                    print("DEBUG: [VIZ] GTRACK disabled - clearing visualization")
                elif not self.gtrack_results:
                    print("DEBUG: [VIZ] No GTRACK results - clearing visualization")
                elif not self.gtrack_results.get('tracks'):
                    stats = self.gtrack_results.get('frame_stats', {})
                    print(f"DEBUG: [VIZ] GTRACK results but no tracks - active:{stats.get('active_tracks', 0)}, detections:{stats.get('valid_detections', 0)} - clearing visualization")
                else:
                    print("DEBUG: [VIZ] Unknown condition - clearing track visualization")
                self._clear_track_visualization()
        else:
            # No point cloud data, clear displays (but only occasionally for performance)
            if current_time - self.last_visualization_update > 500:  # Only clear every 500ms
                self.point_scatter.setData(pos=np.array([[0, 0, 0]]))
                
                # Clear 3D scatter plot for combined view
                if hasattr(self, 'point_scatter_3d_combined'):
                    self.point_scatter_3d_combined.setData(pos=np.array([[0, 0, 0]]))
        
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