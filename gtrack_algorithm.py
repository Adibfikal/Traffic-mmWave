"""
GTRACK-Style Multi-Object Tracking Algorithm
Implements Texas Instruments GTRACK-like tracking for vehicle detection
with x,y coordinate data from mmWave radar recordings.

Key Features:
- Dynamic DBSCAN clustering with SNR-based parameters
- Unscented Kalman Filter for motion prediction
- Probability matrix data association (Hungarian algorithm)
- Track lifecycle management with state machine
- Motion pattern analysis for false target rejection
- Optimized for up to 25 vehicle tracks
"""

import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class TrackState(Enum):
    """Track state enumeration similar to GTRACK"""
    DETECTION = "detection"      # Initial detection
    ACTIVE = "active"           # Confirmed active track
    COASTED = "coasted"         # Temporarily lost but predicted
    FREE = "free"               # Available for new assignment

class MotionModel(Enum):
    """Motion models for different vehicle behaviors"""
    CONSTANT_VELOCITY = "cv"     # Standard highway/street driving
    CONSTANT_ACCELERATION = "ca" # Acceleration/deceleration
    COORDINATED_TURN = "ct"      # Turning vehicles

@dataclass
class Detection:
    """Individual detection from radar data"""
    x: float
    y: float
    snr: float
    timestamp: float
    cluster_id: int = -1

@dataclass
class Track:
    """Vehicle track with state estimation and history"""
    id: int
    state: TrackState
    motion_model: MotionModel
    
    # Kalman filter state [x, y, vx, vy, ax, ay]
    x_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    P_cov: np.ndarray = field(default_factory=lambda: np.eye(6) * 10.0)
    
    # Track management
    detection_count: int = 0
    coast_count: int = 0
    last_update_time: float = 0.0
    
    # Track history for motion analysis
    position_history: deque = field(default_factory=lambda: deque(maxlen=20))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Track quality metrics
    confidence: float = 0.0
    range_rate: float = 0.0
    
    def __post_init__(self):
        """Initialize track with default covariance matrix"""
        if np.allclose(self.P_cov, np.eye(6) * 10.0):
            # Position uncertainty
            self.P_cov[0:2, 0:2] = np.eye(2) * 2.0  # 2m position uncertainty
            # Velocity uncertainty  
            self.P_cov[2:4, 2:4] = np.eye(2) * 5.0  # 5m/s velocity uncertainty
            # Acceleration uncertainty
            self.P_cov[4:6, 4:6] = np.eye(2) * 2.0  # 2m/s² acceleration uncertainty

class GTRACKProcessor:
    """Main GTRACK-style tracking processor"""
    
    def __init__(self, max_tracks: int = 25):
        """
        Initialize GTRACK processor
        
        Args:
            max_tracks: Maximum number of simultaneous tracks (default: 25 for vehicles)
        """
        self.max_tracks = max_tracks
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        
        # GTRACK Parameters - optimized for vehicles
        self.dt = 0.05  # Typical radar frame rate (20 FPS)
        
        # Dynamic DBSCAN parameters by SNR level (more sensitive for real-time)
        self.dbscan_params = {
            'high_snr': {'eps': 2.0, 'min_samples': 2},    # Strong vehicle returns
            'medium_snr': {'eps': 2.5, 'min_samples': 2},  # Medium vehicle returns  
            'low_snr': {'eps': 3.5, 'min_samples': 2}      # Weak/distant vehicles (reduced from 4)
        }
        
        # SNR thresholds (more lenient for better detection)
        self.snr_thresholds = {
            'high': 150,    # Strong vehicle reflections (reduced from 200)
            'medium': 80,   # Medium strength (reduced from 100)
            'low': 30       # Minimum detectable (reduced from 50)
        }
        
        # Track management thresholds (more responsive)
        self.confirmation_threshold = 2    # Detections needed to confirm track (reduced from 3)
        self.deletion_threshold = 8       # Coast cycles before deletion (increased from 5)
        self.max_coast_distance = 10.0   # Max distance for track association (meters)
        
        # Vehicle-specific motion limits
        self.max_vehicle_speed = 60.0     # m/s (216 km/h reasonable max)
        self.max_vehicle_accel = 10.0     # m/s² (reasonable vehicle acceleration)
        
        # Association gate parameters
        self.gate_probability = 0.95      # Statistical gate probability
        self.gate_threshold = 9.21        # Chi-squared threshold for 95% confidence
        
    def process_frame(self, frame_data: Dict) -> Dict[str, Any]:
        """
        Process a single radar frame
        
        Args:
            frame_data: Frame data from recording with timestamp and pointCloud
            
        Returns:
            Dictionary with tracking results
        """
        timestamp = frame_data['timestamp'] / 1000.0  # Convert to seconds
        point_cloud = frame_data['frameData']['pointCloud']
        
        # Extract x,y coordinates and SNR values
        detections = self._extract_detections(point_cloud, timestamp)
        
        # Step 1: Dynamic DBSCAN Clustering
        clustered_detections = self._dynamic_dbscan_clustering(detections)
        
        # Step 2: Kalman Prediction for all tracks
        self._predict_tracks(timestamp)
        
        # Step 3: Data Association using probability matrix
        associations = self._associate_detections_to_tracks(clustered_detections)
        
        # Step 4: Update tracks with associated detections
        self._update_tracks(associations, timestamp)
        
        # Step 5: Create new tracks from unassociated detections
        self._create_new_tracks(associations['unassociated_detections'], timestamp)
        
        # Step 6: Track management (promote, demote, delete)
        self._manage_tracks(timestamp)
        
        # Step 7: Motion pattern analysis for false target rejection
        self._motion_pattern_analysis()
        
        # Return tracking results
        return self._generate_tracking_results(timestamp, detections, clustered_detections)
    
    def _extract_detections(self, point_cloud: List[List[float]], timestamp: float) -> List[Detection]:
        """Extract Detection objects from point cloud data"""
        detections = []
        
        print(f"DEBUG: [EXTRACT] Processing {len(point_cloud)} raw points")
        
        for i, point in enumerate(point_cloud):
            if len(point) >= 7:  # Ensure we have all required fields
                x, y = point[0], point[1]
                # Use the higher SNR value (index 5 seems to be the main SNR)
                snr = point[5] if point[5] > 0 else point[4]
                
                # DEBUG: Log first few points to see data format
                if i < 3:
                    print(f"DEBUG: [EXTRACT] Point {i}: x={x:.2f}, y={y:.2f}, snr={snr:.1f}")
                
                # Filter out obvious noise and invalid detections
                if (abs(x) < 100 and abs(y) < 100 and  # Reasonable detection range
                    snr >= self.snr_thresholds['low']):   # Minimum SNR threshold
                    
                    detections.append(Detection(x=x, y=y, snr=snr, timestamp=timestamp))
                elif i < 3:
                    print(f"DEBUG: [EXTRACT] Point {i} FILTERED: range_check={abs(x) < 100 and abs(y) < 100}, snr_check={snr >= self.snr_thresholds['low']}")
        
        print(f"DEBUG: [EXTRACT] {len(point_cloud)} raw -> {len(detections)} valid detections")
        return detections
    
    def _dynamic_dbscan_clustering(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply dynamic DBSCAN clustering based on SNR levels
        Similar to GTRACK's adaptive clustering
        """
        if not detections:
            print("DEBUG: [CLUSTER] No detections to cluster")
            return detections
        
        # Group detections by SNR level
        snr_groups = {'high': [], 'medium': [], 'low': []}
        
        for det in detections:
            if det.snr >= self.snr_thresholds['high']:
                snr_groups['high'].append(det)
            elif det.snr >= self.snr_thresholds['medium']:
                snr_groups['medium'].append(det)
            else:
                snr_groups['low'].append(det)
        
        print(f"DEBUG: [CLUSTER] SNR groups - High:{len(snr_groups['high'])}, Med:{len(snr_groups['medium'])}, Low:{len(snr_groups['low'])}")
        
        cluster_id = 0
        clustered_detections = []
        
        # Apply DBSCAN with different parameters for each SNR group
        for snr_level, group_detections in snr_groups.items():
            if not group_detections:
                continue
                
            # Create position matrix for clustering
            positions = np.array([[d.x, d.y] for d in group_detections])
            
            # Apply DBSCAN with SNR-appropriate parameters
            snr_key_map = {'high': 'high_snr', 'medium': 'medium_snr', 'low': 'low_snr'}
            params = self.dbscan_params[snr_key_map[snr_level]]
            clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            labels = clustering.fit_predict(positions)
            
            # Count clusters found
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise (-1)
            n_noise = list(labels).count(-1)
            
            print(f"DEBUG: [CLUSTER] {snr_level} SNR: {len(group_detections)} points -> {n_clusters} clusters, {n_noise} noise")
            
            # Assign cluster IDs to detections
            for i, det in enumerate(group_detections):
                if labels[i] != -1:  # Not noise
                    det.cluster_id = cluster_id + labels[i]
                else:
                    det.cluster_id = -1  # Mark as noise
                clustered_detections.append(det)
            
            # Update cluster ID counter
            if len(labels) > 0 and len(unique_labels) > 0:
                max_label = max([label for label in labels if label >= 0], default=-1)
                if max_label >= 0:
                    cluster_id += max_label + 1
        
        valid_clusters = len([d for d in clustered_detections if d.cluster_id != -1])
        print(f"DEBUG: [CLUSTER] Total result: {valid_clusters} valid clustered detections")
        return clustered_detections
    
    def _predict_tracks(self, timestamp: float):
        """Predict track states using Kalman filter"""
        dt = timestamp - getattr(self, '_last_timestamp', timestamp)
        if dt <= 0 or dt > 1.0:  # Handle first frame or large gaps
            dt = self.dt
        
        for track in self.tracks.values():
            if track.state != TrackState.FREE:
                self._kalman_predict(track, dt)
        
        self._last_timestamp = timestamp
    
    def _kalman_predict(self, track: Track, dt: float):
        """
        Kalman filter prediction step for vehicle motion
        State vector: [x, y, vx, vy, ax, ay]
        """
        # State transition matrix for constant acceleration model
        F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0],
            [0, 0, 0,  1,  0,         dt],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1]
        ])
        
        # Process noise covariance (vehicle motion uncertainty)
        q = 1.0  # Process noise intensity
        Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0,       dt**2/2, 0],
            [0,       dt**4/4, 0,       dt**3/2, 0,       dt**2/2],
            [dt**3/2, 0,       dt**2,   0,       dt,      0],
            [0,       dt**3/2, 0,       dt**2,   0,       dt],
            [dt**2/2, 0,       dt,      0,       1,       0],
            [0,       dt**2/2, 0,       dt,      0,       1]
        ]) * q
        
        # Predict state and covariance
        track.x_state = F @ track.x_state
        track.P_cov = F @ track.P_cov @ F.T + Q
        
        # Apply vehicle motion constraints
        self._apply_motion_constraints(track)
    
    def _apply_motion_constraints(self, track: Track):
        """Apply realistic vehicle motion constraints"""
        # Limit maximum speed
        speed = np.linalg.norm(track.x_state[2:4])
        if speed > self.max_vehicle_speed:
            track.x_state[2:4] = track.x_state[2:4] * (self.max_vehicle_speed / speed)
        
        # Limit maximum acceleration
        accel = np.linalg.norm(track.x_state[4:6])
        if accel > self.max_vehicle_accel:
            track.x_state[4:6] = track.x_state[4:6] * (self.max_vehicle_accel / accel)
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Associate detections to tracks using probability matrix and Hungarian algorithm
        This implements GTRACK's data association approach
        """
        # Filter valid detections (not noise)
        valid_detections = [d for d in detections if d.cluster_id != -1]
        
        if not valid_detections or not self.tracks:
            return {
                'associations': [],
                'unassociated_detections': valid_detections,
                'unassociated_tracks': list(self.tracks.keys())
            }
        
        active_tracks = [t for t in self.tracks.values() if t.state != TrackState.FREE]
        
        if not active_tracks:
            return {
                'associations': [],
                'unassociated_detections': valid_detections,
                'unassociated_tracks': []
            }
        
        # Build cost matrix for Hungarian algorithm
        cost_matrix = self._build_association_cost_matrix(valid_detections, active_tracks)
        
        # Solve assignment problem
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Process assignments
        associations = []
        used_detections = set()
        used_tracks = set()
        
        for t_idx, d_idx in zip(track_indices, detection_indices):
            cost = cost_matrix[t_idx, d_idx]
            
            # Check if assignment is within gate
            if cost < self.gate_threshold:
                track = active_tracks[t_idx]
                detection = valid_detections[d_idx]
                
                associations.append({
                    'track_id': track.id,
                    'detection': detection,
                    'cost': cost
                })
                
                used_detections.add(d_idx)
                used_tracks.add(track.id)
        
        # Find unassociated detections and tracks
        unassociated_detections = [valid_detections[i] for i in range(len(valid_detections)) 
                                  if i not in used_detections]
        unassociated_tracks = [t.id for t in active_tracks if t.id not in used_tracks]
        
        return {
            'associations': associations,
            'unassociated_detections': unassociated_detections,
            'unassociated_tracks': unassociated_tracks
        }
    
    def _build_association_cost_matrix(self, detections: List[Detection], tracks: List[Track]) -> np.ndarray:
        """
        Build cost matrix for data association
        Uses Mahalanobis distance considering uncertainty
        """
        cost_matrix = np.full((len(tracks), len(detections)), np.inf)
        
        for t_idx, track in enumerate(tracks):
            track_pos = track.x_state[:2]
            
            for d_idx, detection in enumerate(detections):
                det_pos = np.array([detection.x, detection.y])
                
                # Calculate innovation (residual)
                innovation = det_pos - track_pos
                
                # Measurement covariance (observation uncertainty)
                R = np.eye(2) * (2.0 + (255 - detection.snr) / 255.0 * 3.0)  # SNR-based uncertainty
                
                # Innovation covariance
                H = np.array([[1, 0, 0, 0, 0, 0],    # Measurement matrix
                             [0, 1, 0, 0, 0, 0]])
                
                S = H @ track.P_cov @ H.T + R
                
                # Mahalanobis distance (accounts for uncertainty)
                try:
                    mahalanobis_dist = innovation.T @ np.linalg.inv(S) @ innovation
                    cost_matrix[t_idx, d_idx] = mahalanobis_dist
                except np.linalg.LinAlgError:
                    # Fallback to Euclidean distance if matrix inversion fails
                    euclidean_dist = np.linalg.norm(innovation)
                    cost_matrix[t_idx, d_idx] = euclidean_dist**2
        
        return cost_matrix
    
    def _update_tracks(self, associations: Dict[str, Any], timestamp: float):
        """Update tracks with associated detections using Kalman filter"""
        for assoc in associations['associations']:
            track_id = assoc['track_id']
            detection = assoc['detection']
            
            if track_id in self.tracks:
                track = self.tracks[track_id]
                self._kalman_update(track, detection)
                track.last_update_time = timestamp
                track.detection_count += 1
                track.coast_count = 0
                
                # Update track history
                track.position_history.append([detection.x, detection.y])
                
                # Calculate velocity from position history
                if len(track.position_history) >= 2:
                    dt = self.dt
                    vel = [(track.position_history[-1][i] - track.position_history[-2][i]) / dt 
                          for i in range(2)]
                    track.velocity_history.append(vel)
        
        # Update unassociated tracks (increase coast count)
        for track_id in associations['unassociated_tracks']:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.coast_count += 1
                if track.state == TrackState.ACTIVE:
                    track.state = TrackState.COASTED
    
    def _kalman_update(self, track: Track, detection: Detection):
        """Kalman filter update step with detection measurement"""
        # Measurement vector [x, y]
        z = np.array([detection.x, detection.y])
        
        # Measurement matrix (we observe position only)
        H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0]])
        
        # Measurement noise covariance (SNR-based)
        measurement_noise = 2.0 + (255 - detection.snr) / 255.0 * 3.0
        R = np.eye(2) * measurement_noise
        
        # Innovation
        innovation = z - H @ track.x_state
        
        # Innovation covariance
        S = H @ track.P_cov @ H.T + R
        
        # Kalman gain
        K = track.P_cov @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        track.x_state = track.x_state + K @ innovation
        track.P_cov = (np.eye(6) - K @ H) @ track.P_cov
        
        # Update confidence based on innovation magnitude
        innovation_norm = np.linalg.norm(innovation)
        track.confidence = max(0.1, min(1.0, 1.0 - innovation_norm / 10.0))
    
    def _create_new_tracks(self, unassociated_detections: List[Detection], timestamp: float):
        """Create new tracks from unassociated detections"""
        if len(self.tracks) >= self.max_tracks:
            return  # Don't exceed maximum track limit
        
        for detection in unassociated_detections:
            if len(self.tracks) >= self.max_tracks:
                break
            
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            
            # Initialize track state with detection
            x_state = np.array([detection.x, detection.y, 0.0, 0.0, 0.0, 0.0])
            
            track = Track(
                id=track_id,
                state=TrackState.DETECTION,
                motion_model=MotionModel.CONSTANT_VELOCITY,
                x_state=x_state,
                last_update_time=timestamp,
                detection_count=1
            )
            
            # Add to position history
            track.position_history.append([detection.x, detection.y])
            
            self.tracks[track_id] = track
    
    def _manage_tracks(self, timestamp: float):
        """Manage track lifecycle (confirmation, deletion)"""
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            # Promote detection to active track
            if (track.state == TrackState.DETECTION and 
                track.detection_count >= self.confirmation_threshold):
                track.state = TrackState.ACTIVE
            
            # Delete tracks that have been coasting too long
            if (track.coast_count >= self.deletion_threshold or
                (track.state == TrackState.DETECTION and 
                 timestamp - track.last_update_time > 2.0)):  # 2 second timeout for detections
                tracks_to_delete.append(track_id)
        
        # Remove deleted tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
    
    def _motion_pattern_analysis(self):
        """
        Analyze motion patterns to reject false targets
        Similar to GTRACK's motion pattern analysis
        """
        for track in self.tracks.values():
            if len(track.position_history) < 5:
                continue  # Need sufficient history
            
            # Check for unrealistic motion patterns
            positions = np.array(list(track.position_history))
            
            # Check for excessive jitter (possible multipath)
            if len(positions) >= 3:
                # Calculate position variance
                pos_var = np.var(positions, axis=0)
                if np.mean(pos_var) > 25.0:  # High variance indicates jitter
                    track.confidence *= 0.8  # Reduce confidence
            
            # Check for unrealistic speeds
            if len(track.velocity_history) >= 2:
                velocities = np.array(list(track.velocity_history))
                speeds = np.linalg.norm(velocities, axis=1)
                
                # Penalize tracks with impossible speeds
                max_speed = np.max(speeds)
                if max_speed > self.max_vehicle_speed:
                    track.confidence *= 0.5
    
    def _generate_tracking_results(self, timestamp: float, detections: List[Detection], 
                                 clustered_detections: List[Detection]) -> Dict[str, Any]:
        """Generate comprehensive tracking results"""
        # Include ALL tracks (DETECTION, ACTIVE, COASTED) for visualization
        visible_tracks = [t for t in self.tracks.values() if t.state != TrackState.FREE]
        active_tracks = [t for t in self.tracks.values() if t.state == TrackState.ACTIVE]
        
        tracks_info = []
        for track in visible_tracks:  # Show all visible tracks, not just active ones
            tracks_info.append({
                'id': track.id,
                'position': track.x_state[:2].tolist(),
                'velocity': track.x_state[2:4].tolist(),
                'acceleration': track.x_state[4:6].tolist(),
                'confidence': track.confidence,
                'state': track.state.value,
                'motion_model': track.motion_model.value,
                'detection_count': track.detection_count,
                'coast_count': track.coast_count
            })
        
        # DEBUG: Log track states for diagnosis
        if len(self.tracks) > 0:
            state_counts = {}
            for track in self.tracks.values():
                state = track.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
            print(f"DEBUG: [TRACKS] States: {state_counts}, Visible: {len(visible_tracks)}, Active: {len(active_tracks)}")
        
        return {
            'timestamp': timestamp,
            'frame_stats': {
                'total_detections': len(detections),
                'valid_detections': len([d for d in clustered_detections if d.cluster_id != -1]),
                'active_tracks': len(active_tracks),  # Keep active count for stats
                'total_tracks': len(self.tracks),
                'visible_tracks': len(visible_tracks)  # Add visible track count
            },
            'tracks': tracks_info,
            'detections': [[d.x, d.y] for d in clustered_detections if d.cluster_id != -1],
            'processing_time': 0.0  # Will be calculated by caller
        }

# Utility functions for processing recordings
def process_recording_file(filepath: str, processor: GTRACKProcessor) -> List[Dict[str, Any]]:
    """Process a complete recording file with GTRACK algorithm"""
    with open(filepath, 'r') as f:
        frame_data = json.load(f)
    
    results = []
    
    for frame in frame_data:
        start_time = time.time()
        result = processor.process_frame(frame)
        result['processing_time'] = time.time() - start_time
        results.append(result)
    
    return results

def analyze_recording_folder(recordings_path: str = "recordings") -> Dict[str, Any]:
    """Analyze all recordings in the recordings folder"""
    import os
    import glob
    
    recording_files = glob.glob(os.path.join(recordings_path, "*/radar_data.json"))
    
    analysis_results = {}
    
    for recording_file in recording_files:
        recording_name = os.path.basename(os.path.dirname(recording_file))
        print(f"Processing recording: {recording_name}")
        
        processor = GTRACKProcessor(max_tracks=25)
        results = process_recording_file(recording_file, processor)
        
        # Calculate summary statistics
        total_frames = len(results)
        avg_processing_time = np.mean([r['processing_time'] for r in results])
        max_tracks = max([r['frame_stats']['active_tracks'] for r in results])
        
        analysis_results[recording_name] = {
            'total_frames': total_frames,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_simultaneous_tracks': max_tracks,
            'results': results
        }
        
        print(f"  - Processed {total_frames} frames")
        print(f"  - Average processing time: {avg_processing_time*1000:.2f}ms")
        print(f"  - Max simultaneous tracks: {max_tracks}")
    
    return analysis_results

if __name__ == "__main__":
    # Example usage for testing
    print("GTRACK-Style Vehicle Tracking Algorithm")
    print("======================================")
    
    # Test with a single recording
    processor = GTRACKProcessor(max_tracks=25)
    
    # This would process all recordings
    # results = analyze_recording_folder()
    print("Algorithm ready for processing recordings.") 