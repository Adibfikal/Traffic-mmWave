"""
Radar data acquisition thread for non-blocking radar data reading.
This module handles radar interface setup and data streaming.
"""

import time

# Try to import Qt components, starting with PyQt6, then falling back to PySide6
try:
    from PyQt6.QtCore import pyqtSignal as Signal
except ImportError:
    try:
        from PySide6.QtCore import Signal
    except ImportError:
        raise ImportError("Neither PyQt6 nor PySide6 is installed")

from .base_thread import BaseThread
from radar_interface import RadarInterface


class RadarDataThread(BaseThread):
    """Thread for reading radar data without blocking the GUI"""
    data_ready = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.radar = None
        
    def setup(self, cli_port: str, data_port: str) -> tuple[bool, str]:
        """Initialize radar interface with specified ports"""
        return self.setup_radar(cli_port, data_port)
        
    def setup_radar(self, cli_port: str, data_port: str) -> tuple[bool, str]:
        """Initialize radar interface with specified ports"""
        try:
            self.radar = RadarInterface(cli_port, data_port)
            return True, f"Radar interface initialized with CLI: {cli_port}, Data: {data_port}"
        except Exception as e:
            error_msg = f"Failed to setup radar: {str(e)}"
            self.error_occurred.emit(error_msg)
            return False, error_msg
    
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
    
    def cleanup(self):
        """Clean up radar resources"""
        if self.radar:
            self.radar.close()
            self.radar = None
    
    def send_config(self, config_file: str = 'xwr68xx_config.cfg'):
        """Send radar configuration commands from file"""
        if not self.radar:
            return False, "Radar not initialized"
        return self.radar.send_config(config_file)
    
    def start_sensor(self):
        """Start the radar sensor"""
        if not self.radar:
            return False, "Radar not initialized"
        return self.radar.start_sensor()
    
    def stop_sensor(self):
        """Stop the radar sensor"""
        if not self.radar:
            return False, "Radar not initialized"
        return self.radar.stop_sensor()
    
    def is_connected(self) -> bool:
        """Check if radar is connected"""
        return self.radar is not None and self.radar.is_connected() 