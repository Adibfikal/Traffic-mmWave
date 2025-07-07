"""
Base GUI utilities for radar tracking application.
This module provides common functionality shared between PyQt6 and PySide6 versions.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional

# Import our extracted modules
from threads import WebcamThread, RadarDataThread
try:
    from visualization import VisualizationCoordinator
except ImportError:
    VisualizationCoordinator = None

try:
    from data_recorder import DataRecorder
except ImportError:
    DataRecorder = None


class BaseRadarGUI:
    """Simple base class for radar GUI applications with common utilities"""
    
    def __init__(self):
        """Initialize common GUI components and state"""
        # Initialize threads only if needed
        self.radar_thread = None
        self.webcam_thread = None
        self.data_recorder = None
        
        # Common state variables
        self.point_cloud_data = None
        self.current_frame = None
        self.last_visualization_update = 0
        self.visualization_update_interval = 50
        self.frame_times = deque(maxlen=20)
        self.last_fps_update = 0
        
    def initialize_threads(self):
        """Initialize thread objects"""
        self.radar_thread = RadarDataThread()
        self.webcam_thread = WebcamThread()
        if DataRecorder:
            self.data_recorder = DataRecorder()
            
    def log_status(self, message: str):
        """Log a status message - should be overridden by subclasses"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def handle_error(self, error_msg: str):
        """Handle error messages - should be overridden by subclasses"""
        self.log_status(f"ERROR: {error_msg}")
        
    def cleanup(self):
        """Clean up resources when closing"""
        if self.radar_thread:
            self.radar_thread.stop()
        if self.webcam_thread:
            self.webcam_thread.stop() 