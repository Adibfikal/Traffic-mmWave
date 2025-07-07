"""
Webcam capture thread for non-blocking camera frame acquisition.
This module handles camera detection, configuration, and frame streaming.
"""

import time
import cv2
import numpy as np
import os
from typing import List, Tuple

# Try to import Qt components, starting with PyQt6, then falling back to PySide6
try:
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtWidgets import QApplication
except ImportError:
    try:
        from PySide6.QtCore import Signal
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise ImportError("Neither PyQt6 nor PySide6 is installed")

from .base_thread import BaseThread


class WebcamThread(BaseThread):
    """Thread for webcam capture without blocking the GUI"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.camera_index = 0
        
    def setup(self, camera_index: int = 0) -> tuple[bool, str]:
        """Initialize camera with specified index"""
        return self.setup_camera(camera_index)
        
    def setup_camera(self, camera_index: int = 0) -> tuple[bool, str]:
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
                        
                        message = f"Camera {camera_index} connected using {backend_name}"
                        self.error_occurred.emit(message)
                        return True, message
                    else:
                        # Can't read frames, try next backend
                        self.camera.release()
                        continue
                        
                except Exception as e:
                    if self.camera:
                        self.camera.release()
                    continue
            
            # If we get here, all backends failed
            error_msg = f"Cannot access camera {camera_index} with any backend. Camera may be in use by another application."
            raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Failed to setup camera: {str(e)}"
            self.error_occurred.emit(error_msg)
            return False, error_msg
    
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
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
    
    @staticmethod
    def scan_available_cameras(status_callback=None) -> List[Tuple[int, str]]:
        """
        Scan for available cameras and return a list of camera indices and names.
        
        Args:
            status_callback: Optional callback function to report scanning status
            
        Returns:
            List of tuples (camera_index, camera_name)
        """
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
                if status_callback:
                    status_callback(f"Testing camera {i}...")
                
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
                                
                                if status_callback:
                                    status_callback(f"✓ Found: {camera_name}")
                                break  # Found working backend, stop trying others
                                
                    except Exception:
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
                        if status_callback:
                            status_callback(f"Stopping scan after {max_consecutive_failures} consecutive failures")
                        break
                        
        finally:
            # Restore original OpenCV log level
            if old_opencv_log_level:
                os.environ['OPENCV_LOG_LEVEL'] = old_opencv_log_level
            else:
                os.environ.pop('OPENCV_LOG_LEVEL', None)
                
        return available_cameras 