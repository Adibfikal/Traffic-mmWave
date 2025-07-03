import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTextEdit, QGroupBox, QGridLayout, QSplitter, QTabWidget, QComboBox)
from PySide6.QtCore import QTimer, Signal, QThread, Slot, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from collections import deque
import time
import cv2
import os

from radar_interface import RadarInterface
from tracking import ObjectTracker

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
        self.tracker = ObjectTracker()
        
        # Data storage
        self.point_cloud_data = None
        self.tracked_objects = {}
        self.object_trails = {}  # Store trails for each tracked object
        self.max_trail_length = 50
        self.current_frame = None
        
        # Display mode state - True for point cloud, False for tracked targets
        self.show_point_cloud_mode = True
        
        # Setup UI
        self.setup_ui()
        
        # Connect signals
        self.radar_thread.data_ready.connect(self.update_data)
        self.radar_thread.error_occurred.connect(self.handle_error)
        self.webcam_thread.frame_ready.connect(self.update_webcam_frame)
        self.webcam_thread.error_occurred.connect(self.handle_webcam_error)
        
        # Setup update timer for visualization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(50)  # 20 FPS update rate
        
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
        """Create the control panel with port selection and controls"""
        panel = QGroupBox("Controls")
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
        
        # Display mode toggle
        self.display_mode_btn = QPushButton("Mode: Point Cloud")
        self.display_mode_btn.setCheckable(True)
        self.display_mode_btn.setChecked(True)  # Start with point cloud mode
        self.display_mode_btn.toggled.connect(self.toggle_display_mode)
        button_layout.addWidget(self.display_mode_btn)
        
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
        self.objects_label = QLabel("Tracked Objects: 0")
        self.fps_label = QLabel("FPS: 0")
        self.webcam_status_label = QLabel("Webcam: Disconnected")
        
        stats_layout.addWidget(self.points_label)
        stats_layout.addWidget(self.objects_label)
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.webcam_status_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def create_visualization_panel(self):
        """Create the visualization panel with tabs for radar and webcam"""
        panel = QGroupBox("Visualization")
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Radar tab
        radar_tab = self.create_radar_tab()
        self.tab_widget.addTab(radar_tab, "Radar Data")
        
        # Webcam tab
        webcam_tab = self.create_webcam_tab()
        self.tab_widget.addTab(webcam_tab, "Webcam Feed")
        
        # Combined view tab
        combined_tab = self.create_combined_tab()
        self.tab_widget.addTab(combined_tab, "Combined View")
        
        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel
    
    def create_radar_tab(self):
        """Create the radar visualization tab"""
        radar_widget = QWidget()
        layout = QVBoxLayout()
        
        # Create a splitter to hold both views
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # === 3D View ===
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
        
        # Dictionary to store scatter plots for tracked objects
        self.object_scatters = {}
        
        # Dictionary to store line plots for trails
        self.trail_lines = {}
        
        # === 2D Bird's Eye View ===
        self.plot_2d = pg.PlotWidget()
        self.plot_2d.setLabel('left', 'Y Distance', units='m')
        self.plot_2d.setLabel('bottom', 'X Distance', units='m')
        self.plot_2d.setTitle("Bird's Eye View (2D)")
        self.plot_2d.setAspectLocked(True)
        self.plot_2d.showGrid(x=True, y=True)
        
        # Set range with y starting at 0
        self.plot_2d.setXRange(-10, 10)
        self.plot_2d.setYRange(0, 20)
        
        # Add origin marker
        origin = pg.ScatterPlotItem([0], [0], pen='w', brush='w', size=10, symbol='o')
        self.plot_2d.addItem(origin)
        
        # Create scatter plot for 2D point cloud
        self.point_scatter_2d = pg.ScatterPlotItem(
            [], [], 
            pen=None, 
            brush=(255, 255, 255, 100),
            size=5
        )
        self.plot_2d.addItem(self.point_scatter_2d)
        
        # Dictionaries for 2D tracked objects and trails
        self.object_scatters_2d = {}
        self.trail_lines_2d = {}
        
        # Add both widgets to splitter
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.plot_2d)
        splitter.setSizes([400, 400])  # Equal sizes
        
        layout.addWidget(splitter)
        radar_widget.setLayout(layout)
        return radar_widget
    
    def create_webcam_tab(self):
        """Create the webcam feed tab"""
        webcam_widget = QWidget()
        layout = QVBoxLayout()
        
        # Webcam display label
        self.webcam_label = QLabel("Webcam feed will appear here")
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        
        layout.addWidget(self.webcam_label)
        webcam_widget.setLayout(layout)
        return webcam_widget
    
    def create_combined_tab(self):
        """Create the combined view tab showing both radar and webcam"""
        combined_widget = QWidget()
        layout = QHBoxLayout()
        
        # Left side - Radar (2D view)
        radar_side = QWidget()
        radar_layout = QVBoxLayout()
        radar_layout.addWidget(QLabel("Radar Bird's Eye View"))
        
        # Create a smaller 2D plot for combined view
        self.plot_2d_combined = pg.PlotWidget()
        self.plot_2d_combined.setLabel('left', 'Y Distance', units='m')
        self.plot_2d_combined.setLabel('bottom', 'X Distance', units='m')
        self.plot_2d_combined.setAspectLocked(True)
        self.plot_2d_combined.showGrid(x=True, y=True)
        self.plot_2d_combined.setXRange(-10, 10)
        self.plot_2d_combined.setYRange(0, 20)
        
        # Add origin marker for combined view
        origin_combined = pg.ScatterPlotItem([0], [0], pen='w', brush='w', size=10, symbol='o')
        self.plot_2d_combined.addItem(origin_combined)
        
        # Create scatter plot for combined 2D view
        self.point_scatter_2d_combined = pg.ScatterPlotItem(
            [], [], 
            pen=None, 
            brush=(255, 255, 255, 100),
            size=5
        )
        self.plot_2d_combined.addItem(self.point_scatter_2d_combined)
        
        radar_layout.addWidget(self.plot_2d_combined)
        radar_side.setLayout(radar_layout)
        
        # Right side - Webcam
        webcam_side = QWidget()
        webcam_layout = QVBoxLayout()
        webcam_layout.addWidget(QLabel("Webcam Feed"))
        
        self.webcam_label_combined = QLabel("Webcam feed will appear here")
        self.webcam_label_combined.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label_combined.setMinimumSize(320, 240)
        self.webcam_label_combined.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        
        webcam_layout.addWidget(self.webcam_label_combined)
        webcam_side.setLayout(webcam_layout)
        
        # Add both sides to main layout
        layout.addWidget(radar_side, 2)  # Radar takes 2/3 of space
        layout.addWidget(webcam_side, 1)  # Webcam takes 1/3 of space
        
        combined_widget.setLayout(layout)
        return combined_widget
        
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
    
    def toggle_display_mode(self, checked):
        """Toggle between point cloud and tracked targets display"""
        if checked:
            self.show_point_cloud_mode = True
            self.display_mode_btn.setText("Mode: Point Cloud")
            self.log_status("Display mode: Point Cloud only")
        else:
            self.show_point_cloud_mode = False
            self.display_mode_btn.setText("Mode: Tracked Targets")
            self.log_status("Display mode: Tracked Targets only")
    
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
            self.webcam_label.setText("Webcam feed will appear here")
            self.webcam_label_combined.setText("Webcam feed will appear here")
            self.log_status("Webcam stopped")
    
    @Slot(np.ndarray)
    def update_webcam_frame(self, frame):
        """Update webcam display with new frame"""
        self.current_frame = frame
        
        # Convert numpy array to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.webcam_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Update both webcam displays
        self.webcam_label.setPixmap(scaled_pixmap)
        
        # For combined view, scale to smaller size
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
        # Extract point cloud
        if 'pointCloud' in data:
            self.point_cloud_data = data['pointCloud']
            
            # Debug mode: print point cloud data to terminal
            if self.debug_checkbox.isChecked() and self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
                frame_num = data.get('header', {}).get('frameNumber', 'N/A')
                print(f"\nFrame {frame_num}:")
                print("-" * 60)
                
                for i, point in enumerate(self.point_cloud_data):
                    if len(point) >= 4:
                        # Include doppler if available
                        print(f"Point {i:3d}: X={point[0]:7.3f}m  Y={point[1]:7.3f}m  Z={point[2]:7.3f}m  Doppler={point[3]:6.3f}m/s")
                    else:
                        # Fallback for old format without doppler
                        print(f"Point {i:3d}: X={point[0]:7.3f}m  Y={point[1]:7.3f}m  Z={point[2]:7.3f}m")
                
                print(f"Total points: {len(self.point_cloud_data)}")
            
            # Update tracker with new point cloud (need to handle 4D points)
            if self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
                # Extract only x,y,z for tracker (it expects 3D points)
                points_3d = self.point_cloud_data[:, :3] if self.point_cloud_data.shape[1] >= 3 else self.point_cloud_data
                
                # Check if we have enough points for clustering (HDBSCAN needs at least 2)
                if len(points_3d) >= 2:
                    tracked_objects = self.tracker.update(points_3d)
                    self.tracked_objects = tracked_objects
                else:
                    # For single points, just treat as untracked objects
                    self.tracked_objects = {}
            else:
                tracked_objects = self.tracker.update(None)
                self.tracked_objects = tracked_objects
            
            # Update trails
            self.update_trails()
            
            # Update statistics
            num_points = len(self.point_cloud_data) if self.point_cloud_data is not None else 0
            if self.show_point_cloud_mode:
                self.points_label.setText(f"Points: {num_points} (displayed)")
                self.objects_label.setText(f"Tracked Objects: {len(self.tracked_objects)} (hidden)")
            else:
                self.points_label.setText(f"Points: {num_points} (hidden)")
                self.objects_label.setText(f"Tracked Objects: {len(self.tracked_objects)} (displayed)")
            
    def update_trails(self):
        """Update trail history for tracked objects"""
        current_ids = set(self.tracked_objects.keys())
        
        # Remove trails for objects that no longer exist
        for obj_id in list(self.object_trails.keys()):
            if obj_id not in current_ids:
                del self.object_trails[obj_id]
                
        # Update trails for existing objects
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_id not in self.object_trails:
                self.object_trails[obj_id] = deque(maxlen=self.max_trail_length)
                
            position = obj_data['position']
            self.object_trails[obj_id].append(position.copy())
            
    def update_visualization(self):
        """Update both 3D and 2D visualizations"""
        # Update 3D point cloud - only show if in point cloud mode
        if self.show_point_cloud_mode and self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
            # Extract only x,y,z for 3D visualization
            if self.point_cloud_data.shape[1] >= 3:
                points_3d = self.point_cloud_data[:, :3]
            else:
                points_3d = self.point_cloud_data
                
            self.point_scatter.setData(pos=points_3d)
            
            # Update 2D point cloud (X-Y projection)
            x_points = points_3d[:, 0]
            y_points = points_3d[:, 1]
            self.point_scatter_2d.setData(x_points, y_points)
            
            # Update combined view
            if hasattr(self, 'point_scatter_2d_combined'):
                self.point_scatter_2d_combined.setData(x_points, y_points)
        elif not self.show_point_cloud_mode:
            # Hide point cloud data when in tracked targets mode
            self.point_scatter.setData(pos=np.array([[0, 0, 0]]))
            self.point_scatter_2d.setData([], [])
            
            # Also hide in combined view
            if hasattr(self, 'point_scatter_2d_combined'):
                self.point_scatter_2d_combined.setData([], [])
        
        # Update tracked objects and trails - only show if in tracked targets mode
        if not self.show_point_cloud_mode:
            current_ids = set(self.tracked_objects.keys())
            existing_ids = set(self.object_scatters.keys())
            
            # Remove visualizations for objects that no longer exist
            for obj_id in existing_ids - current_ids:
                # Remove from 3D view
                if obj_id in self.object_scatters:
                    self.plot_widget.removeItem(self.object_scatters[obj_id])
                    del self.object_scatters[obj_id]
                    
                if obj_id in self.trail_lines:
                    self.plot_widget.removeItem(self.trail_lines[obj_id])
                    del self.trail_lines[obj_id]
                    
                # Remove from 2D view
                if obj_id in self.object_scatters_2d:
                    self.plot_2d.removeItem(self.object_scatters_2d[obj_id])
                    del self.object_scatters_2d[obj_id]
                    
                if obj_id in self.trail_lines_2d:
                    self.plot_2d.removeItem(self.trail_lines_2d[obj_id])
                    del self.trail_lines_2d[obj_id]
            
            # Update or create visualizations for current objects
            for obj_id, obj_data in self.tracked_objects.items():
                color = self.get_object_color(obj_id)
                color_rgb = [int(c * 255) for c in color]
                
                # === Update 3D visualization ===
                if obj_id not in self.object_scatters:
                    scatter = gl.GLScatterPlotItem(
                        pos=np.array([obj_data['position']]),
                        color=color + (1,),
                        size=0.5
                    )
                    self.plot_widget.addItem(scatter)
                    self.object_scatters[obj_id] = scatter
                else:
                    self.object_scatters[obj_id].setData(
                        pos=np.array([obj_data['position']])
                    )
                
                # Update 3D trail
                if obj_id in self.object_trails and len(self.object_trails[obj_id]) > 1:
                    trail_points = np.array(list(self.object_trails[obj_id]))
                    
                    if obj_id not in self.trail_lines:
                        line = gl.GLLinePlotItem(
                            pos=trail_points,
                            color=color + (0.5,),
                            width=2,
                            antialias=True
                        )
                        self.plot_widget.addItem(line)
                        self.trail_lines[obj_id] = line
                    else:
                        self.trail_lines[obj_id].setData(pos=trail_points)
                        
                # === Update 2D visualization ===
                obj_x = obj_data['position'][0]
                obj_y = obj_data['position'][1]
                
                # Update 2D object position
                if obj_id not in self.object_scatters_2d:
                    scatter_2d = pg.ScatterPlotItem(
                        [obj_x], [obj_y],
                        pen=pg.mkPen(color=color_rgb, width=2),
                        brush=pg.mkBrush(color_rgb + [255]),
                        size=10,
                        symbol='o'
                    )
                    self.plot_2d.addItem(scatter_2d)
                    self.object_scatters_2d[obj_id] = scatter_2d
                else:
                    self.object_scatters_2d[obj_id].setData([obj_x], [obj_y])
                
                # Update 2D trail
                if obj_id in self.object_trails and len(self.object_trails[obj_id]) > 1:
                    trail_points = np.array(list(self.object_trails[obj_id]))
                    trail_x = trail_points[:, 0]
                    trail_y = trail_points[:, 1]
                    
                    if obj_id not in self.trail_lines_2d:
                        line_2d = self.plot_2d.plot(
                            trail_x, trail_y,
                            pen=pg.mkPen(color=color_rgb + [128], width=2)
                        )
                        self.trail_lines_2d[obj_id] = line_2d
                    else:
                        self.trail_lines_2d[obj_id].setData(trail_x, trail_y)
        else:
            # Hide tracked objects when in point cloud mode
            current_ids = set(self.tracked_objects.keys())
            existing_ids = set(self.object_scatters.keys())
            
            # Hide all tracked object visualizations
            for obj_id in existing_ids:
                # Hide from 3D view
                if obj_id in self.object_scatters:
                    self.plot_widget.removeItem(self.object_scatters[obj_id])
                    del self.object_scatters[obj_id]
                    
                if obj_id in self.trail_lines:
                    self.plot_widget.removeItem(self.trail_lines[obj_id])
                    del self.trail_lines[obj_id]
                    
                # Hide from 2D view
                if obj_id in self.object_scatters_2d:
                    self.plot_2d.removeItem(self.object_scatters_2d[obj_id])
                    del self.object_scatters_2d[obj_id]
                    
                if obj_id in self.trail_lines_2d:
                    self.plot_2d.removeItem(self.trail_lines_2d[obj_id])
                    del self.trail_lines_2d[obj_id]
                    
    def get_object_color(self, obj_id):
        """Get a consistent color for an object based on its ID"""
        # Generate a color based on object ID
        np.random.seed(obj_id)
        return tuple(np.random.rand(3))
        
    def handle_error(self, error_msg):
        """Handle error messages from the radar thread"""
        self.log_status(f"ERROR: {error_msg}")
        
    def log_status(self, message):
        """Log a status message"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        
    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.radar_thread.stop()
        self.webcam_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = RadarGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 