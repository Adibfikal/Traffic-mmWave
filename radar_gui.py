import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTextEdit, QGroupBox, QGridLayout, QSplitter)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, pyqtSlot, Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from collections import deque
import time

from radar_interface import RadarInterface
from tracking import ObjectTracker

class RadarDataThread(QThread):
    """Thread for reading radar data without blocking the GUI"""
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
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
        self.setWindowTitle("IWR6843ISK Radar Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.radar_thread = RadarDataThread()
        self.tracker = ObjectTracker()
        
        # Data storage
        self.point_cloud_data = None
        self.tracked_objects = {}
        self.object_trails = {}  # Store trails for each tracked object
        self.max_trail_length = 50
        
        # Setup UI
        self.setup_ui()
        
        # Connect signals
        self.radar_thread.data_ready.connect(self.update_data)
        self.radar_thread.error_occurred.connect(self.handle_error)
        
        # Setup update timer for visualization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(50)  # 20 FPS update rate
        
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
        self.cli_port_input = QLineEdit("COM4")
        port_layout.addWidget(self.cli_port_input, 0, 1)
        
        port_layout.addWidget(QLabel("Data Port:"), 1, 0)
        self.data_port_input = QLineEdit("COM3")
        port_layout.addWidget(self.data_port_input, 1, 1)
        
        port_group.setLayout(port_layout)
        layout.addWidget(port_group)
        
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
        self.objects_label = QLabel("Tracked Objects: 0")
        self.fps_label = QLabel("FPS: 0")
        
        stats_layout.addWidget(self.points_label)
        stats_layout.addWidget(self.objects_label)
        stats_layout.addWidget(self.fps_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def create_visualization_panel(self):
        """Create the visualization panel with 3D and 2D views"""
        panel = QGroupBox("Radar Visualization")
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
        panel.setLayout(layout)
        return panel
        
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
                
                # Show debug information if enabled
                if self.debug_checkbox.isChecked() and response:
                    self.log_status("=== DEBUG: Configuration Responses ===")
                    if isinstance(response, list):
                        for item in response:
                            if isinstance(item, dict):
                                self.log_status(f"CMD: {item['command']}")
                                self.log_status(f"RSP: {item['response']}")
                            else:
                                self.log_status(f"RSP: {item}")
                    else:
                        self.log_status(f"Response: {response}")
                    self.log_status("=== END DEBUG ===")
            else:
                self.log_status("Failed to send configuration")
                if self.debug_checkbox.isChecked() and response:
                    self.log_status(f"DEBUG: Error details: {response}")
                
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
                    
                    # Show debug information if enabled
                    if self.debug_checkbox.isChecked() and response:
                        self.log_status(f"DEBUG: Start sensor response: {response}")
                else:
                    self.log_status("Failed to start sensor")
                    if self.debug_checkbox.isChecked() and response:
                        self.log_status(f"DEBUG: Error: {response}")
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
                    
                    # Show debug information if enabled
                    if self.debug_checkbox.isChecked() and response:
                        self.log_status(f"DEBUG: Stop sensor response: {response}")
                else:
                    self.log_status("Failed to stop sensor")
                    if self.debug_checkbox.isChecked() and response:
                        self.log_status(f"DEBUG: Error: {response}")
                
    def toggle_debug_mode(self, checked):
        """Toggle debug mode"""
        if checked:
            self.debug_checkbox.setText("Debug Mode: ON")
            self.log_status("Debug mode enabled")
        else:
            self.debug_checkbox.setText("Debug Mode: OFF")
            self.log_status("Debug mode disabled")
                
    @pyqtSlot(dict)
    def update_data(self, data):
        """Update with new radar data"""
        # Extract point cloud
        if 'pointCloud' in data:
            self.point_cloud_data = data['pointCloud']
            
            # Update tracker with new point cloud
            tracked_objects = self.tracker.update(self.point_cloud_data)
            self.tracked_objects = tracked_objects
            
            # Update trails
            self.update_trails()
            
            # Update statistics
            num_points = len(self.point_cloud_data) if self.point_cloud_data is not None else 0
            self.points_label.setText(f"Points: {num_points}")
            self.objects_label.setText(f"Tracked Objects: {len(self.tracked_objects)}")
            
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
        # Update 3D point cloud
        if self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
            self.point_scatter.setData(pos=self.point_cloud_data)
            
            # Update 2D point cloud (X-Y projection)
            x_points = self.point_cloud_data[:, 0]
            y_points = self.point_cloud_data[:, 1]
            self.point_scatter_2d.setData(x_points, y_points)
        
        # Update tracked objects and trails
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
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = RadarGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 