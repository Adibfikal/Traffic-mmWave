"""
Base thread classes for Qt applications that work with both PyQt6 and PySide6.
This module provides thread abstractions for webcam and radar data handling.
"""

import time
import numpy as np

# Try to import Qt components, starting with PyQt6, then falling back to PySide6
try:
    from PyQt6.QtCore import QThread, pyqtSignal as Signal
except ImportError:
    try:
        from PySide6.QtCore import QThread, Signal
    except ImportError:
        raise ImportError("Neither PyQt6 nor PySide6 is installed")


class BaseThread(QThread):
    """Base class for all threads providing common functionality"""
    
    def __init__(self):
        super().__init__()
        self.is_running = False
    
    def setup(self, *args, **kwargs):
        """Setup the thread with given parameters. Returns (success, message)"""
        # Default implementation - subclasses should override
        return True, "Setup complete"
    
    def cleanup(self):
        """Clean up resources when stopping"""
        # Default implementation - subclasses should override
        pass
    
    def stop(self):
        """Stop the thread and clean up resources"""
        self.is_running = False
        self.cleanup()
        self.wait() 