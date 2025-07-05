import json
import time
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
import shutil
import threading
from queue import Queue


class DataRecorder:
    """
    Handles recording and playback of radar point cloud data and camera frames.
    Saves video as MP4 files and radar data as JSON in organized folders.
    Supports recording:
    - Point cloud data only
    - Camera frames only  
    - Both point cloud and camera data
    """
    
    def __init__(self):
        self.is_recording = False
        self.recording_mode = "both"  # "pointcloud", "camera", "both"
        self.session_folder = None
        self.radar_data = []
        self.start_time = None
        self.frame_count = 0
        
        # Video recording
        self.video_writer = None
        self.video_fps = 30
        self.video_codec = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
        self.video_frame_size = None
        
        # Async video recording
        self.video_queue = Queue()
        self.video_thread = None
        self.video_thread_running = False
        
        # Playback state
        self.is_playing = False
        self.playback_data = None
        self.playback_video = None
        self.playback_index = 0
        self.playback_start_time = None
        self.playback_speed = 1.0
        self.playback_session_info = None
        
    def set_recording_mode(self, mode: str):
        """Set recording mode: 'pointcloud', 'camera', or 'both'"""
        if mode in ["pointcloud", "camera", "both"]:
            self.recording_mode = mode
        else:
            raise ValueError("Mode must be 'pointcloud', 'camera', or 'both'")
    
    def set_video_settings(self, fps: int = 30, codec: str = 'mp4v'):
        """Configure video recording settings"""
        self.video_fps = max(1, min(60, fps))
        self.video_codec = cv2.VideoWriter.fourcc(*codec)  # type: ignore
    
    def _create_session_folder(self):
        """Create a new session folder for recording"""
        # Create recordings directory if it doesn't exist
        recordings_dir = "recordings"
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
        
        # Create session folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = os.path.join(recordings_dir, f"session_{timestamp}")
        os.makedirs(self.session_folder, exist_ok=True)
        
        return self.session_folder
    
    def start_recording(self, config_commands: Optional[List[str]] = None):
        """Start recording session"""
        if self.is_recording:
            return False, "Already recording"
        
        # Create session folder
        session_path = self._create_session_folder()
        
        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0
        self.radar_data = []
        
        # Create session info
        session_info = {
            "recording_mode": self.recording_mode,
            "start_time": datetime.now().isoformat(),
            "config_commands": config_commands or [],
            "device": "xWR6843ISK",
            "video_fps": self.video_fps if self.recording_mode in ["camera", "both"] else None
        }
        
        # Save session info
        with open(os.path.join(session_path, "session_info.json"), 'w') as f:
            json.dump(session_info, f, indent=2)
        
        return True, f"Started recording in '{self.recording_mode}' mode\nSession: {os.path.basename(session_path)}"
    
    def stop_recording(self):
        """Stop recording session"""
        if not self.is_recording:
            return False, "Not currently recording"
        
        self.is_recording = False
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Stop async video recording thread
        if self.video_thread_running:
            self.video_thread_running = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=5.0)  # Wait up to 5 seconds
        
        # Close video writer if open
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Save radar data if any
        if self.radar_data and self.session_folder:
            radar_file = os.path.join(self.session_folder, "radar_data.json")
            with open(radar_file, 'w') as f:
                json.dump(self.radar_data, f, indent=2)
        
        # Update session info with end time and statistics
        if self.session_folder:
            session_info_file = os.path.join(self.session_folder, "session_info.json")
            if os.path.exists(session_info_file):
                with open(session_info_file, 'r') as f:
                    session_info = json.load(f)
                
                session_info.update({
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "total_frames": self.frame_count,
                    "radar_frames": len(self.radar_data),
                    "has_video": os.path.exists(os.path.join(self.session_folder, "camera_video.mp4"))
                })
                
                with open(session_info_file, 'w') as f:
                    json.dump(session_info, f, indent=2)
        
        if self.session_folder is not None:
            session_name = os.path.basename(self.session_folder)
        else:
            session_name = "unknown"
        return True, f"Recording stopped: {session_name}\n{self.frame_count} frames in {duration:.2f}s"
    
    def add_radar_frame(self, radar_data: Dict[str, Any]):
        """Add radar frame data to recording"""
        if not self.is_recording or self.recording_mode not in ["pointcloud", "both"]:
            return
        
        num_points = radar_data.get('numDetectedPoints', 0) or 0
        if num_points > 0:  # Only log when we have actual data
            if self.frame_count % 20 == 0:  # Reduce frequency
                print(f"DEBUG: Recording radar frame {self.frame_count + 1} with {num_points} points")
        
        # Format frame data with timestamp
        frame_entry = {
            "timestamp": time.time() * 1000,  # milliseconds
            "frameData": {
                "error": radar_data.get("error", 0),
                "frameNum": radar_data.get("frameNum", self.frame_count),
                "pointCloud": self._format_point_cloud(radar_data.get("pointCloud", [])),
                "numDetectedPoints": radar_data.get("numDetectedPoints", 0),
                "numDetectedTracks": radar_data.get("numDetectedTracks", 0),
                "trackData": radar_data.get("trackData", []),
                "trackIndexes": radar_data.get("trackIndexes", [])
            }
        }
        
        self.radar_data.append(frame_entry)
        self.frame_count += 1
    
    def add_camera_frame(self, camera_frame: np.ndarray):
        """Add camera frame to video recording"""
        if not self.is_recording or self.recording_mode not in ["camera", "both"]:
            return
        
        if self.session_folder is None:
            return
        
        # Initialize video writer on first frame
        if self.video_writer is None:
            height, width = camera_frame.shape[:2]
            self.video_frame_size = (width, height)
            
            video_path = os.path.join(self.session_folder, "camera_video.mp4")
            self.video_writer = cv2.VideoWriter(
                video_path, 
                self.video_codec, 
                self.video_fps, 
                self.video_frame_size
            )
            
            if not self.video_writer.isOpened():
                print(f"Warning: Could not open video writer for {video_path}")
                return
        
        # Write frame to video
        if self.video_writer and self.video_writer.isOpened():
            # Ensure frame is in correct format for OpenCV video writer
            if len(camera_frame.shape) == 3 and camera_frame.shape[2] == 3:
                # The frame should already be in BGR format from the GUI conversion
                # Just ensure it's uint8 type
                frame_bgr = camera_frame.astype(np.uint8)
                
                # Resize frame if necessary to match video writer size
                if frame_bgr.shape[:2][::-1] != self.video_frame_size:
                    frame_bgr = cv2.resize(frame_bgr, self.video_frame_size)
                
                self.video_writer.write(frame_bgr)
    
    def add_camera_frame_async(self, camera_frame: np.ndarray):
        """Add camera frame to recording queue for async processing"""
        if not self.is_recording or self.recording_mode not in ["camera", "both"]:
            return
        
        if self.session_folder is None:
            return
        
        # Start video recording thread if not already running
        if not self.video_thread_running:
            self._start_video_thread()
        
        # Add frame to queue (non-blocking)
        try:
            self.video_queue.put(camera_frame, block=False)
        except:
            # Queue is full, skip this frame to avoid blocking
            pass
    
    def _start_video_thread(self):
        """Start the video recording thread"""
        if self.video_thread_running:
            return
        
        self.video_thread_running = True
        self.video_thread = threading.Thread(target=self._video_recording_worker, daemon=True)
        self.video_thread.start()
    
    def _video_recording_worker(self):
        """Worker thread for async video recording"""
        while self.video_thread_running or not self.video_queue.empty():
            try:
                # Get frame from queue with timeout
                frame = self.video_queue.get(timeout=1.0)
                
                # Initialize video writer on first frame
                if self.video_writer is None:
                    height, width = frame.shape[:2]
                    self.video_frame_size = (width, height)
                    
                    video_path = os.path.join(self.session_folder, "camera_video.mp4")
                    self.video_writer = cv2.VideoWriter(
                        video_path, 
                        self.video_codec, 
                        self.video_fps, 
                        self.video_frame_size
                    )
                
                # Write frame to video
                if self.video_writer and self.video_writer.isOpened():
                    # Ensure frame is in correct format
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = frame.astype(np.uint8)
                        
                        # Resize if necessary
                        if frame_bgr.shape[:2][::-1] != self.video_frame_size:
                            frame_bgr = cv2.resize(frame_bgr, self.video_frame_size)
                        
                        self.video_writer.write(frame_bgr)
                
                self.video_queue.task_done()
                
            except:
                # Timeout or queue empty
                continue
        
        self.video_thread_running = False
    
    def _format_point_cloud(self, point_cloud: Any) -> List[List[float]]:
        """Format point cloud data to match replay format"""
        # Handle both numpy arrays and lists
        if point_cloud is None:
            return []
        
        # Convert numpy array to list if needed
        if hasattr(point_cloud, 'tolist'):
            point_cloud = point_cloud.tolist()
        
        # Check if empty after conversion
        try:
            if len(point_cloud) == 0:
                return []
        except (TypeError, AttributeError):
            return []
        
        formatted_points = []
        for point in point_cloud:
            if len(point) >= 3:  # At minimum need x, y, z
                # Ensure we have 7 elements like in replay_example.json
                formatted_point = list(point[:7])  # Take first 7 elements
                
                # Pad with defaults if needed
                while len(formatted_point) < 7:
                    if len(formatted_point) == 3:  # velocity
                        formatted_point.append(0.0)
                    elif len(formatted_point) == 4:  # intensity  
                        formatted_point.append(10.0)
                    elif len(formatted_point) == 5:  # range
                        formatted_point.append(65.0)
                    else:  # trackIndex
                        formatted_point.append(255.0)
                
                formatted_points.append(formatted_point)
        
        return formatted_points
    
    def save_recording(self, custom_name: str = None) -> tuple[bool, str]:
        """Move/rename current recording session"""
        if self.is_recording:
            return False, "Cannot save while recording is active"
        
        if not self.session_folder or not os.path.exists(self.session_folder):
            return False, "No recording session to save"
        
        if custom_name:
            # Create new name for the session folder
            recordings_dir = os.path.dirname(self.session_folder)
            new_folder = os.path.join(recordings_dir, custom_name)
            
            try:
                if os.path.exists(new_folder):
                    return False, f"Recording '{custom_name}' already exists"
                
                shutil.move(self.session_folder, new_folder)
                self.session_folder = new_folder
                
                return True, f"Recording saved as '{custom_name}'"
            except Exception as e:
                return False, f"Failed to rename recording: {str(e)}"
        else:
            # Already saved in session folder
            folder_name = os.path.basename(self.session_folder)
            return True, f"Recording available as '{folder_name}'"
    
    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all available recordings"""
        recordings = []
        recordings_dir = "recordings"
        
        if not os.path.exists(recordings_dir):
            return recordings
        
        for item in os.listdir(recordings_dir):
            session_path = os.path.join(recordings_dir, item)
            if os.path.isdir(session_path):
                session_info_file = os.path.join(session_path, "session_info.json")
                
                if os.path.exists(session_info_file):
                    try:
                        with open(session_info_file, 'r') as f:
                            info = json.load(f)
                        
                        # Calculate folder size
                        total_size = 0
                        for dirpath, dirnames, filenames in os.walk(session_path):
                            for filename in filenames:
                                filepath = os.path.join(dirpath, filename)
                                total_size += os.path.getsize(filepath)
                        
                        recordings.append({
                            "name": item,
                            "path": session_path,
                            "info": info,
                            "size_mb": total_size / (1024 * 1024)
                        })
                    except Exception as e:
                        print(f"Error reading session info for {item}: {e}")
        
        # Sort by creation time (newest first)
        recordings.sort(key=lambda x: x["info"].get("start_time", ""), reverse=True)
        return recordings
    
    def load_recording(self, session_path: str) -> tuple[bool, str]:
        """Load recording session for playback"""
        try:
            if not os.path.exists(session_path):
                return False, f"Recording not found: {session_path}"
            
            # Load session info
            session_info_file = os.path.join(session_path, "session_info.json")
            if not os.path.exists(session_info_file):
                return False, "Invalid recording: missing session info"
            
            with open(session_info_file, 'r') as f:
                self.playback_session_info = json.load(f)
            
            # Load radar data if available
            radar_file = os.path.join(session_path, "radar_data.json")
            if os.path.exists(radar_file):
                with open(radar_file, 'r') as f:
                    self.playback_data = json.load(f)
            else:
                self.playback_data = []
            
            # Load video if available
            video_file = os.path.join(session_path, "camera_video.mp4")
            if os.path.exists(video_file):
                self.playback_video = cv2.VideoCapture(video_file)
                if not self.playback_video.isOpened():
                    self.playback_video = None
                    return False, "Could not open video file"
            else:
                self.playback_video = None
            
            self.playback_index = 0
            self.playback_start_time = None
            
            recording_name = os.path.basename(session_path)
            mode = self.playback_session_info.get("recording_mode", "unknown")
            frame_count = len(self.playback_data)
            
            return True, f"Loaded '{recording_name}' ({mode}, {frame_count} frames)"
            
        except Exception as e:
            return False, f"Failed to load recording: {str(e)}"
    
    def start_playback(self, speed: float = 1.0):
        """Start playback of loaded recording"""
        if not self.playback_session_info:
            return False, "No recording loaded"
        
        self.is_playing = True
        self.playback_index = 0
        self.playback_start_time = time.time()
        self.playback_speed = max(0.1, min(10.0, speed))
        
        # Reset video to beginning
        if self.playback_video:
            self.playback_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return True, f"Started playback at {self.playback_speed}x speed"
    
    def stop_playback(self):
        """Stop playback"""
        self.is_playing = False
        return True, "Playback stopped"
    
    def get_next_playback_frame(self) -> Optional[Dict[str, Any]]:
        """Get next frame for playback based on timing"""
        if not self.is_playing or not self.playback_session_info:
            return None
        
        result = {}
        
        # Handle radar data
        if self.playback_data and self.playback_index < len(self.playback_data):
            radar_frame = self.playback_data[self.playback_index]
            
            # Check timing for radar frame
            if self.playback_start_time is None:
                self.playback_start_time = time.time()
                if self.playback_data:
                    first_timestamp = self.playback_data[0]["timestamp"]
                    current_timestamp = radar_frame["timestamp"]
                    self.playback_start_time -= (current_timestamp - first_timestamp) / 1000.0
            
            current_time = time.time()
            elapsed_real_time = current_time - self.playback_start_time
            
            if self.playback_data:
                first_timestamp = self.playback_data[0]["timestamp"]
                frame_relative_time = (radar_frame["timestamp"] - first_timestamp) / 1000.0
                expected_time = frame_relative_time / self.playback_speed
                
                if elapsed_real_time >= expected_time:
                    result.update(radar_frame)
                    self.playback_index += 1
        
        # Handle video data
        if self.playback_video and self.playback_video.isOpened():
            ret, frame = self.playback_video.read()
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result["cameraFrame"] = {
                    "decoded_image": frame_rgb,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": frame.shape[2]
                }
        
        # Check if playback is complete
        if (not self.playback_data or self.playback_index >= len(self.playback_data)) and \
           (not self.playback_video or not self.playback_video.isOpened()):
            self.is_playing = False
        
        return result if result else None
    
    def get_playback_info(self) -> Dict[str, Any]:
        """Get information about current playback state"""
        if not self.playback_session_info:
            return {"loaded": False}
        
        total_frames = len(self.playback_data) if self.playback_data else 0
        
        info = {
            "loaded": True,
            "total_frames": total_frames,
            "current_frame": self.playback_index,
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed,
            "progress": (self.playback_index / total_frames * 100) if total_frames > 0 else 0,
            "session_info": self.playback_session_info
        }
        
        return info
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get statistics about current recording session"""
        stats = {
            "is_recording": self.is_recording,
            "recording_mode": self.recording_mode,
            "frame_count": self.frame_count
        }
        
        if self.start_time:
            duration = time.time() - self.start_time
            stats["duration_seconds"] = duration
            stats["fps"] = self.frame_count / duration if duration > 0 else 0
        
        if self.session_folder:
            stats["session_folder"] = os.path.basename(self.session_folder)
            stats["radar_frames"] = len(self.radar_data)
            stats["has_video"] = self.video_writer is not None
        
        return stats 