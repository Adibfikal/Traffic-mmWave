"""
GTRACK Algorithm Implementation - No SNR Version
Optimized for vehicle tracking using only X,Y coordinate data
Replaces SNR-based clustering with distance and density-based methods
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

@dataclass
class Detection:
    """Point detection from radar"""
    x: float
    y: float
    timestamp: float
    cluster_id: int = -1
    range_val: float = field(init=False)
    
    def __post_init__(self):
        self.range_val = np.sqrt(self.x**2 + self.y**2)

@dataclass
class ClusterCentroid:
    """Cluster centroid for vehicle detection"""
    x: float
    y: float
    timestamp: float
    cluster_id: int
    point_count: int
    range_val: float = field(init=False)
    
    def __post_init__(self):
        self.range_val = np.sqrt(self.x**2 + self.y**2)

class TrackState(Enum):
    """Track lifecycle states"""
    DETECTION = "detection"      # Initial detection
    ACTIVE = "active"           # Confirmed track  
    COASTED = "coasted"         # Predicted state (no detection)
    LOST = "lost"               # Track ended

class MotionModel(Enum):
    """Motion models for different target types"""
    CONSTANT_VELOCITY = "cv"
    CONSTANT_ACCELERATION = "ca"
    STATIONARY = "stationary"

@dataclass
class Track:
    """Individual track state"""
    id: int
    x_state: np.ndarray  # [x, y, vx, vy, ax, ay]
    P_covariance: np.ndarray  # State covariance
    state: TrackState = TrackState.DETECTION
    motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY
    
    # Track management
    detection_count: int = 0
    coast_count: int = 0
    confidence: float = 0.0
    
    # Motion analysis
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)
    last_detection_time: float = 0.0

class GTRACKNoSNR:
    """GTRACK Algorithm without SNR dependency"""
    
    def __init__(self, max_tracks: int = 12):  # Realistic max for 8-10 vehicles
        self.max_tracks = max_tracks
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        
        # Distance-based clustering parameters (ULTRA restrictive for realistic vehicle count)
        self.clustering_params = {
            'near_range': {'eps': 3.5, 'min_samples': 12},   # <20m range - need 12+ points for vehicle
            'mid_range': {'eps': 4.5, 'min_samples': 15},    # 20-40m range - need 15+ points for vehicle  
            'far_range': {'eps': 6.5, 'min_samples': 18}     # >40m range - need 18+ points for vehicle
        }
        
        # Motion-based filtering (realistic vehicle constraints)
        self.max_vehicle_speed = 25.0  # 25 m/s max vehicle speed (~90 km/h) - more realistic
        self.max_acceleration = 5.0    # 5 m/s² max vehicle acceleration - more realistic
        
        # Position filtering (realistic road/traffic boundaries - narrower focus)
        self.position_limits = {
            'min_x': -15.0, 'max_x': 15.0,  # Narrower road width (30m total)
            'min_y': 8.0,   'max_y': 60.0   # Skip very close detections (likely noise)
        }
        
        # Track management (EXTREMELY conservative - realistic vehicle tracking)
        self.confirmation_threshold = 8     # Need 8 consistent detections to confirm track
        self.deletion_threshold = 12       # Allow longer coasting before deletion
        self.max_association_distance = 6.0  # Slightly relaxed association distance
        
        # Additional constraints for realistic vehicle detection
        self.max_clusters_per_frame = 10   # Limit clusters per frame to realistic vehicle count
        self.min_cluster_points = 10       # Minimum points in cluster to consider (vehicles are big)
        
        # Kalman filter parameters
        self.process_noise = 2.0
        self.measurement_noise = 1.0
        self.dt = 0.05  # 50ms frame interval
        
        # Frame statistics
        self.frame_count = 0
        self.total_frames = 0
        
    def process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main GTRACK processing pipeline"""
        start_time = time.time()
        
        self.frame_count += 1
        self.total_frames += 1
        
        # Extract timestamp and point cloud
        timestamp = frame_data.get('timestamp', time.time() * 1000) / 1000.0
        point_cloud = frame_data.get('frameData', {}).get('pointCloud', [])
        
        # Step 1: Extract and filter detections (X,Y only)
        detections = self._extract_detections(point_cloud, timestamp)
        
        # Step 2: Distance-based clustering (replace SNR clustering)
        clustered_detections = self._distance_based_clustering(detections)
        
        # Step 3: Extract cluster centroids (1 per cluster = 1 per vehicle)
        cluster_centroids = self._extract_cluster_centroids(clustered_detections, timestamp)
        
        # Step 4: Kalman prediction for existing tracks
        for track in self.tracks.values():
            self._kalman_predict(track, timestamp)
        
        # Step 5: Data association using centroids (not individual points)
        associated_centroids = self._associate_centroids(cluster_centroids, timestamp)
        
        # Step 6: Update associated tracks with centroids
        for track_id, centroid in associated_centroids.items():
            self._kalman_update_centroid(self.tracks[track_id], centroid, timestamp)
            
        # Step 7: Create new tracks from unassociated centroids
        unassociated_centroids = [c for c in cluster_centroids if 
                                c.cluster_id not in [cent.cluster_id for cent in associated_centroids.values()]]
        self._create_new_tracks_from_centroids(unassociated_centroids, timestamp)
        
        # Step 8: Track management (confirm, coast, delete)
        self._manage_track_lifecycle(timestamp)
        
        # Step 9: Motion analysis and false target rejection
        self._analyze_motion_patterns()
        
        # Generate results
        processing_time = time.time() - start_time
        results = self._generate_tracking_results(timestamp, detections, clustered_detections, cluster_centroids)
        results['processing_time'] = processing_time
        
        return results
    
    def _extract_detections(self, point_cloud: List[List[float]], timestamp: float) -> List[Detection]:
        """Extract valid detections from point cloud - X,Y only"""
        detections = []
        
        for point in point_cloud:
            if len(point) >= 2:  # Only need x,y coordinates
                x, y = point[0], point[1]
                
                # Position-based filtering (replace SNR filtering)
                if (self.position_limits['min_x'] <= x <= self.position_limits['max_x'] and
                    self.position_limits['min_y'] <= y <= self.position_limits['max_y']):
                    
                    detections.append(Detection(x=x, y=y, timestamp=timestamp))
        
        return detections
    
    def _distance_based_clustering(self, detections: List[Detection]) -> List[Detection]:
        """
        Distance-based clustering (replaces SNR-based clustering)
        Uses different parameters based on range instead of SNR
        """
        if not detections:
            return detections
        
        # Group detections by range instead of SNR
        range_groups = {'near': [], 'mid': [], 'far': []}
        
        for det in detections:
            if det.range_val < 20.0:
                range_groups['near'].append(det)
            elif det.range_val < 40.0:
                range_groups['mid'].append(det)
            else:
                range_groups['far'].append(det)
        
        cluster_id = 0
        clustered_detections = []
        total_valid_clusters = 0  # Track total clusters across all ranges
        
        # Apply DBSCAN with range-appropriate parameters
        for range_level, group_detections in range_groups.items():
            if len(group_detections) < 2:
                # Too few points, mark as noise
                for det in group_detections:
                    det.cluster_id = -1
                clustered_detections.extend(group_detections)
                continue
            
            # Prepare data for DBSCAN
            positions = np.array([[det.x, det.y] for det in group_detections])
            
            # Get appropriate parameters for this range
            if range_level == 'near':
                params = self.clustering_params['near_range']
            elif range_level == 'mid':
                params = self.clustering_params['mid_range']
            else:
                params = self.clustering_params['far_range']
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            cluster_labels = clustering.fit_predict(positions)
            
            # Check cluster quality and apply limits
            cluster_sizes = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:
                    if label not in cluster_sizes:
                        cluster_sizes[label] = 0
                    cluster_sizes[label] += 1
            
            # Filter clusters by minimum size and total cluster limit
            valid_cluster_labels = set()
            for label, size in cluster_sizes.items():
                if (size >= self.min_cluster_points and 
                    total_valid_clusters < self.max_clusters_per_frame):
                    valid_cluster_labels.add(label)
                    total_valid_clusters += 1
            
            # Assign cluster IDs only to valid clusters
            for det, label in zip(group_detections, cluster_labels):
                if label == -1 or label not in valid_cluster_labels:
                    det.cluster_id = -1  # Noise or filtered out
                else:
                    det.cluster_id = cluster_id + label
            
            # Update cluster ID for next group
            if len(valid_cluster_labels) > 0:
                cluster_id += len(valid_cluster_labels)
            
            clustered_detections.extend(group_detections)
        
        return clustered_detections
    
    def _extract_cluster_centroids(self, clustered_detections: List[Detection], timestamp: float) -> List[ClusterCentroid]:
        """Extract centroids from clustered detections - one centroid per cluster"""
        cluster_points = {}
        
        # Group points by cluster ID
        for detection in clustered_detections:
            if detection.cluster_id != -1:  # Valid cluster
                if detection.cluster_id not in cluster_points:
                    cluster_points[detection.cluster_id] = []
                cluster_points[detection.cluster_id].append(detection)
        
        # Calculate centroid for each cluster
        centroids = []
        for cluster_id, points in cluster_points.items():
            if len(points) >= self.min_cluster_points:  # Quality filter
                # Calculate centroid position
                x_coords = [p.x for p in points]
                y_coords = [p.y for p in points]
                
                centroid_x = float(np.mean(x_coords))
                centroid_y = float(np.mean(y_coords))
                
                # Create centroid object
                centroid = ClusterCentroid(
                    x=centroid_x,
                    y=centroid_y,
                    timestamp=timestamp,
                    cluster_id=cluster_id,
                    point_count=len(points)
                )
                
                centroids.append(centroid)
        
        return centroids
    
    def _kalman_predict(self, track: Track, timestamp: float):
        """Kalman filter prediction step"""
        dt = timestamp - track.last_detection_time if track.last_detection_time > 0 else self.dt
        dt = min(dt, 0.2)  # Cap dt to prevent instability
        
        # State transition matrix (constant velocity/acceleration model)
        if track.motion_model == MotionModel.CONSTANT_VELOCITY:
            F = np.array([
                [1, 0, dt, 0,  0,  0],
                [0, 1, 0,  dt, 0,  0],
                [0, 0, 1,  0,  0,  0],
                [0, 0, 0,  1,  0,  0],
                [0, 0, 0,  0,  0,  0],
                [0, 0, 0,  0,  0,  0]
            ])
        else:  # CONSTANT_ACCELERATION
            F = np.array([
                [1, 0, dt, 0,  0.5*dt**2, 0],
                [0, 1, 0,  dt, 0,         0.5*dt**2],
                [0, 0, 1,  0,  dt,        0],
                [0, 0, 0,  1,  0,         dt],
                [0, 0, 0,  0,  1,         0],
                [0, 0, 0,  0,  0,         1]
            ])
        
        # Process noise covariance
        Q = np.eye(6) * self.process_noise * dt
        
        # Prediction
        track.x_state = F @ track.x_state
        track.P_covariance = F @ track.P_covariance @ F.T + Q
        
        # Apply motion constraints
        speed = np.sqrt(track.x_state[2]**2 + track.x_state[3]**2)
        if speed > self.max_vehicle_speed:
            # Scale down velocity if unrealistic
            scale = self.max_vehicle_speed / speed
            track.x_state[2] *= scale
            track.x_state[3] *= scale
    
    def _associate_centroids(self, centroids: List[ClusterCentroid], timestamp: float) -> Dict[int, ClusterCentroid]:
        """Associate cluster centroids to tracks using Hungarian algorithm"""
        active_tracks = list(self.tracks.values())
        
        if not centroids or not active_tracks:
            return {}
        
        # Build cost matrix (distance-based)
        cost_matrix = np.full((len(active_tracks), len(centroids)), np.inf)
        
        for i, track in enumerate(active_tracks):
            track_pos = track.x_state[:2]
            
            for j, centroid in enumerate(centroids):
                centroid_pos = np.array([centroid.x, centroid.y])
                distance = np.linalg.norm(track_pos - centroid_pos)
                
                # Only consider if within association gate
                if distance <= self.max_association_distance:
                    # Add motion consistency bonus
                    motion_bonus = self._calculate_motion_consistency_centroid(track, centroid)
                    cost_matrix[i, j] = distance - motion_bonus
        
        # Hungarian algorithm for optimal assignment
        associations = {}
        if np.any(cost_matrix < np.inf):
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                for row, col in zip(row_indices, col_indices):
                    if cost_matrix[row, col] < self.max_association_distance:
                        track = active_tracks[row]
                        centroid = centroids[col]
                        associations[track.id] = centroid
            except ValueError:
                # Cost matrix infeasible, no associations possible
                pass
        
        return associations
    
    def _calculate_motion_consistency_centroid(self, track: Track, centroid: ClusterCentroid) -> float:
        """Calculate motion consistency bonus for centroid association"""
        # Predict where track should be based on velocity
        predicted_x = track.x_state[0] + track.x_state[2] * self.dt
        predicted_y = track.x_state[1] + track.x_state[3] * self.dt
        
        # Distance between prediction and centroid
        pred_error = np.sqrt((predicted_x - centroid.x)**2 + (predicted_y - centroid.y)**2)
        
        # Return bonus (lower error = higher bonus)
        return max(0, 2.0 - pred_error)
    
    def _kalman_update_centroid(self, track: Track, centroid: ClusterCentroid, timestamp: float):
        """Kalman filter update step using cluster centroid"""
        # Measurement matrix (observe position only)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        R = np.eye(2) * self.measurement_noise
        
        # Measurement (centroid position)
        z = np.array([centroid.x, centroid.y])
        
        # Innovation
        y = z - H @ track.x_state
        S = H @ track.P_covariance @ H.T + R
        
        # Kalman gain
        K = track.P_covariance @ H.T @ np.linalg.inv(S)
        
        # Update
        track.x_state = track.x_state + K @ y
        track.P_covariance = (np.eye(6) - K @ H) @ track.P_covariance
        
        # Update track statistics
        track.detection_count += 1
        track.coast_count = 0
        track.last_detection_time = timestamp
        
        # Update position history
        track.position_history.append((track.x_state[0], track.x_state[1]))
        if len(track.position_history) > 10:
            track.position_history = track.position_history[-10:]
        
        # Calculate confidence based on detection count and covariance
        pos_uncertainty = np.sqrt(track.P_covariance[0,0] + track.P_covariance[1,1])
        track.confidence = min(1.0, track.detection_count / 5.0) * max(0.1, 1.0 - pos_uncertainty/10.0)
    
    def _create_new_tracks_from_centroids(self, centroids: List[ClusterCentroid], timestamp: float):
        """Create new tracks from unassociated cluster centroids - one track per centroid"""
        # Limit new track creation to prevent overwhelming
        max_new_tracks_per_frame = 2  # Very conservative track creation
        new_tracks_created = 0
        
        # Sort centroids by quality (point count and proximity)
        sorted_centroids = sorted(centroids, key=lambda c: (-c.point_count, c.range_val))
        
        for centroid in sorted_centroids:
            if (len(self.tracks) >= self.max_tracks or 
                new_tracks_created >= max_new_tracks_per_frame):
                break
            
            # Additional quality filter - only create tracks for good centroids
            if centroid.range_val > 50.0:  # Skip very far centroids
                continue
                
            # Initialize state [x, y, vx, vy, ax, ay]
            x_state = np.array([centroid.x, centroid.y, 0.0, 0.0, 0.0, 0.0])
            P_covariance = np.eye(6) * 10.0  # High initial uncertainty
            
            track = Track(
                id=self.next_track_id,
                x_state=x_state,
                P_covariance=P_covariance,
                state=TrackState.DETECTION,
                detection_count=1,
                last_detection_time=timestamp
            )
            
            track.position_history.append((centroid.x, centroid.y))
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
            new_tracks_created += 1
    
    def _manage_track_lifecycle(self, timestamp: float):
        """Manage track states: detection -> active -> coasted -> lost"""
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            # State transitions
            if track.state == TrackState.DETECTION:
                if track.detection_count >= self.confirmation_threshold:
                    track.state = TrackState.ACTIVE
                    
            elif track.state == TrackState.ACTIVE:
                if track.coast_count > 0:
                    track.state = TrackState.COASTED
                    
            elif track.state == TrackState.COASTED:
                if track.coast_count >= self.deletion_threshold:
                    track.state = TrackState.LOST
                    tracks_to_delete.append(track_id)
            
            # Increment coast count for tracks without recent detections
            time_since_detection = timestamp - track.last_detection_time
            if time_since_detection > self.dt * 1.5:  # Missed more than 1.5 frames
                track.coast_count += 1
        
        # Delete lost tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
    
    def _analyze_motion_patterns(self):
        """Analyze motion patterns to reject false targets"""
        for track in self.tracks.values():
            if len(track.position_history) >= 3:
                # Calculate recent motion
                recent_positions = track.position_history[-3:]
                distances = []
                for i in range(1, len(recent_positions)):
                    dx = recent_positions[i][0] - recent_positions[i-1][0]
                    dy = recent_positions[i][1] - recent_positions[i-1][1]
                    distances.append(np.sqrt(dx**2 + dy**2))
                
                avg_speed = np.mean(distances) / self.dt if distances else 0
                
                # Update motion model based on speed
                if avg_speed < 0.5:
                    track.motion_model = MotionModel.STATIONARY
                elif avg_speed < 15.0:
                    track.motion_model = MotionModel.CONSTANT_VELOCITY
                else:
                    track.motion_model = MotionModel.CONSTANT_ACCELERATION
                
                # Store velocity history
                track.velocity_history.append(float(avg_speed))
                if len(track.velocity_history) > 5:
                    track.velocity_history = track.velocity_history[-5:]
    
    def _generate_tracking_results(self, timestamp: float, detections: List[Detection], 
                                 clustered_detections: List[Detection], 
                                 cluster_centroids: List[ClusterCentroid]) -> Dict[str, Any]:
        """Generate comprehensive tracking results"""
        active_tracks = [t for t in self.tracks.values() if t.state == TrackState.ACTIVE]
        visible_tracks = [t for t in self.tracks.values() if t.state in [TrackState.ACTIVE, TrackState.DETECTION]]
        
        tracks_info = []
        for track in visible_tracks:  # Show both active and detection state tracks
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
        
        return {
            'timestamp': timestamp,
            'frame_stats': {
                'total_detections': len(detections),
                'valid_detections': len([d for d in clustered_detections if d.cluster_id != -1]),
                'cluster_centroids': len(cluster_centroids),
                'active_tracks': len(active_tracks),
                'total_tracks': len(self.tracks),
                'visible_tracks': len(visible_tracks),
                'total_frames': self.total_frames
            },
            'tracks': tracks_info,
            'detections': [[d.x, d.y] for d in clustered_detections if d.cluster_id != -1],
            'centroids': [[c.x, c.y] for c in cluster_centroids],  # For debugging
            'processing_time': 0.0  # Will be set by caller
        }

# Factory function for easy integration
def create_gtrack_processor_no_snr(max_tracks: int = 25) -> GTRACKNoSNR:
    """Create a GTRACK processor without SNR dependency"""
    return GTRACKNoSNR(max_tracks=max_tracks) 