"""
Enhanced HDBSCAN Tracker with IMM Filtering and Robust Data Association
Designed for robust tracking of objects that stop, start, and move again
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, deque
import time
from typing import List, Optional, Tuple, Dict, Any, Set
import logging

from enhanced_tracking import EnhancedTrack, MotionState

class AdaptiveHDBSCANClusterer:
    """Adaptive HDBSCAN clustering with dynamic parameter adjustment"""
    
    def __init__(self, 
                 base_min_cluster_size: int = 3,
                 base_min_samples: int = 2,
                 base_cluster_selection_epsilon: float = 0.5,
                 density_adaptation_factor: float = 0.3,
                 motion_adaptation_factor: float = 0.2):
        
        self.base_min_cluster_size = base_min_cluster_size
        self.base_min_samples = base_min_samples
        self.base_cluster_selection_epsilon = base_cluster_selection_epsilon
        self.density_adaptation_factor = density_adaptation_factor
        self.motion_adaptation_factor = motion_adaptation_factor
        
        # Adaptive parameters
        self.current_min_cluster_size = base_min_cluster_size
        self.current_min_samples = base_min_samples
        self.current_epsilon = base_cluster_selection_epsilon
        
        # History for adaptation
        self.point_density_history = deque(maxlen=10)
        self.cluster_count_history = deque(maxlen=10)
        self.detection_history = deque(maxlen=20)
        
        # Clustering statistics
        self.last_cluster_count = 0
        self.last_noise_points = 0
        self.last_point_density = 0.0
        
    def adapt_parameters(self, point_cloud: np.ndarray, existing_tracks: List[EnhancedTrack]):
        """Adapt HDBSCAN parameters based on point cloud characteristics and track states"""
        
        if len(point_cloud) < 3:
            return
            
        # Calculate current point density
        if len(point_cloud) > 1:
            distances = pdist(point_cloud)
            avg_distance = np.mean(distances)
            point_density = len(point_cloud) / (avg_distance + 1e-6)
        else:
            point_density = 1.0
            
        self.point_density_history.append(point_density)
        
        # Adapt based on point density
        if len(self.point_density_history) >= 3:
            density_trend = np.mean(list(self.point_density_history)[-3:])
            
            # If density is high, allow smaller clusters
            if density_trend > 10.0:
                self.current_min_cluster_size = max(2, self.base_min_cluster_size - 1)
                self.current_min_samples = max(1, self.base_min_samples - 1)
            # If density is low, require larger clusters
            elif density_trend < 3.0:
                self.current_min_cluster_size = self.base_min_cluster_size + 1
                self.current_min_samples = self.base_min_samples + 1
            else:
                self.current_min_cluster_size = self.base_min_cluster_size
                self.current_min_samples = self.base_min_samples
        
        # Adapt based on existing track motion states
        stationary_tracks = sum(1 for track in existing_tracks 
                              if track.current_motion_state == MotionState.STATIONARY)
        
        if stationary_tracks > 0:
            # For stationary objects, use tighter clustering
            self.current_epsilon = self.base_cluster_selection_epsilon * 0.8
        else:
            # For moving objects, use looser clustering
            self.current_epsilon = self.base_cluster_selection_epsilon * 1.2
        
        # Clamp parameters to reasonable ranges
        self.current_min_cluster_size = np.clip(self.current_min_cluster_size, 2, 10)
        self.current_min_samples = np.clip(self.current_min_samples, 1, 5)
        self.current_epsilon = np.clip(self.current_epsilon, 0.1, 2.0)
    
    def cluster_points(self, point_cloud: np.ndarray, existing_tracks: List[EnhancedTrack]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform adaptive HDBSCAN clustering"""
        
        if len(point_cloud) < 2:
            return np.array([-1] * len(point_cloud)), np.array([])
        
        # Adapt parameters based on current conditions
        self.adapt_parameters(point_cloud, existing_tracks)
        
        # Perform clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.current_min_cluster_size,
            min_samples=self.current_min_samples,
            cluster_selection_epsilon=self.current_epsilon,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        
        cluster_labels = clusterer.fit_predict(point_cloud)
        
        # Update statistics
        unique_labels = set(cluster_labels)
        self.last_cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.last_noise_points = np.sum(cluster_labels == -1)
        
        self.cluster_count_history.append(self.last_cluster_count)
        
        # Get cluster probabilities if available
        probabilities = getattr(clusterer, 'probabilities_', np.ones(len(point_cloud)))
        
        return cluster_labels, probabilities
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get current clustering information"""
        return {
            'min_cluster_size': self.current_min_cluster_size,
            'min_samples': self.current_min_samples,
            'epsilon': self.current_epsilon,
            'last_cluster_count': self.last_cluster_count,
            'last_noise_points': self.last_noise_points,
            'avg_point_density': np.mean(list(self.point_density_history)) if self.point_density_history else 0.0
        }

class RobustDataAssociator:
    """Robust data association using Mahalanobis distance and track confidence"""
    
    def __init__(self, 
                 max_association_distance: float = 5.0,
                 mahalanobis_threshold: float = 9.21,  # Chi-squared 95% confidence for 3D
                 confidence_weight: float = 0.3,
                 motion_state_bonus: float = 0.2):
        
        self.max_association_distance = max_association_distance
        self.mahalanobis_threshold = mahalanobis_threshold
        self.confidence_weight = confidence_weight
        self.motion_state_bonus = motion_state_bonus
        
        # Association statistics
        self.association_distances = deque(maxlen=100)
        self.association_success_rate = deque(maxlen=50)
        
    def calculate_association_cost(self, track: EnhancedTrack, detection: np.ndarray, 
                                 detection_covariance: np.ndarray) -> float:
        """Calculate comprehensive association cost between track and detection"""
        
        # Predict track position
        predicted_pos = track.position
        
        # Calculate Mahalanobis distance
        innovation = detection - predicted_pos
        
        # Combined covariance: track uncertainty + detection uncertainty
        track_pos_cov = track.current_covariance[0:3, 0:3]
        combined_cov = track_pos_cov + detection_covariance
        
        try:
            mahal_dist_squared = innovation.T @ la.inv(combined_cov) @ innovation
            mahal_dist = np.sqrt(mahal_dist_squared)
        except la.LinAlgError:
            mahal_dist = 100.0  # Very high cost for singular covariance
        
        # Base cost from Mahalanobis distance
        if mahal_dist > self.mahalanobis_threshold:
            return float('inf')  # Reject association
        
        base_cost = mahal_dist
        
        # Confidence adjustment
        confidence_factor = 1.0 - (track.confidence * self.confidence_weight)
        base_cost *= confidence_factor
        
        # Motion state bonus
        motion_bonus = 0.0
        euclidean_dist = np.linalg.norm(innovation)
        
        if track.current_motion_state == MotionState.STATIONARY:
            # Stationary objects should have very low movement
            if euclidean_dist < 1.0:
                motion_bonus = -self.motion_state_bonus
            else:
                motion_bonus = euclidean_dist * 0.5  # Penalty for moving too much
        
        elif track.current_motion_state == MotionState.CONSTANT_VELOCITY:
            # Check if detection is consistent with predicted velocity
            if len(track.velocity_history) >= 2:
                expected_pos = track.position + track.velocity * 0.1  # Assume 0.1s dt
                velocity_consistency = np.linalg.norm(detection - expected_pos)
                motion_bonus = velocity_consistency * 0.3
        
        elif track.current_motion_state == MotionState.MANEUVERING:
            # Maneuvering objects can have larger deviations
            motion_bonus = -self.motion_state_bonus * 0.5
        
        final_cost = base_cost + motion_bonus
        
        # Age and hit rate adjustment
        age_factor = 1.0 + (track.age * 0.01)  # Slight penalty for older tracks
        hit_rate_factor = 2.0 - (track.hits / max(track.age, 1))  # Bonus for good hit rate
        
        final_cost *= age_factor * hit_rate_factor
        
        return max(0.0, final_cost)
    
    def associate_tracks_to_detections(self, tracks: List[EnhancedTrack], 
                                     detections: List[np.ndarray],
                                     detection_covariances: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks to detections using Hungarian algorithm with robust cost function"""
        
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Build cost matrix
        cost_matrix = np.full((len(tracks), len(detections)), float('inf'))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                detection_cov = detection_covariances[j] if j < len(detection_covariances) else np.eye(3)
                cost = self.calculate_association_cost(track, detection, detection_cov)
                
                # Only consider associations within maximum distance
                euclidean_dist = np.linalg.norm(detection - track.position)
                if euclidean_dist <= self.max_association_distance:
                    cost_matrix[i, j] = cost
        
        # Handle empty cost matrix
        if np.all(cost_matrix == float('inf')):
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Hungarian algorithm for optimal assignment
        # Replace inf with large number for hungarian algorithm
        cost_matrix_finite = np.where(cost_matrix == float('inf'), 1e6, cost_matrix)
        
        try:
            track_indices, detection_indices = linear_sum_assignment(cost_matrix_finite)
        except ValueError:
            # Fallback to simple nearest neighbor
            return self._fallback_association(tracks, detections)
        
        # Filter out associations with infinite cost
        valid_associations = []
        for i, (track_idx, det_idx) in enumerate(zip(track_indices, detection_indices)):
            if cost_matrix[track_idx, det_idx] < float('inf'):
                valid_associations.append((track_idx, det_idx))
                self.association_distances.append(cost_matrix[track_idx, det_idx])
        
        # Determine unassociated tracks and detections
        associated_tracks = set(assoc[0] for assoc in valid_associations)
        associated_detections = set(assoc[1] for assoc in valid_associations)
        
        unassociated_tracks = [i for i in range(len(tracks)) if i not in associated_tracks]
        unassociated_detections = [i for i in range(len(detections)) if i not in associated_detections]
        
        # Update success rate
        success_rate = len(valid_associations) / len(tracks)
        self.association_success_rate.append(success_rate)
        
        return valid_associations, unassociated_tracks, unassociated_detections
    
    def _fallback_association(self, tracks: List[EnhancedTrack], 
                            detections: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Fallback association method using simple nearest neighbor"""
        associations = []
        used_detections = set()
        
        for i, track in enumerate(tracks):
            best_detection = None
            best_distance = float('inf')
            
            for j, detection in enumerate(detections):
                if j in used_detections:
                    continue
                
                distance = np.linalg.norm(detection - track.position)
                if distance < best_distance and distance <= self.max_association_distance:
                    best_distance = distance
                    best_detection = j
            
            if best_detection is not None:
                associations.append((i, best_detection))
                used_detections.add(best_detection)
        
        associated_tracks = set(assoc[0] for assoc in associations)
        associated_detections = set(assoc[1] for assoc in associations)
        
        unassociated_tracks = [i for i in range(len(tracks)) if i not in associated_tracks]
        unassociated_detections = [i for i in range(len(detections)) if i not in associated_detections]
        
        return associations, unassociated_tracks, unassociated_detections
    
    def get_association_statistics(self) -> Dict[str, Any]:
        """Get data association statistics"""
        return {
            'avg_association_distance': np.mean(list(self.association_distances)) if self.association_distances else 0.0,
            'avg_success_rate': np.mean(list(self.association_success_rate)) if self.association_success_rate else 0.0,
            'max_association_distance': self.max_association_distance,
            'mahalanobis_threshold': self.mahalanobis_threshold
        }

class EnhancedHDBSCANTracker:
    """Enhanced HDBSCAN tracker with IMM filtering and robust data association"""
    
    def __init__(self, 
                 max_tracking_distance: float = 5.0,
                 min_cluster_size: int = 3,
                 min_samples: int = 2,
                 cluster_selection_epsilon: float = 0.5,
                 track_confirmation_threshold: int = 3,
                 track_deletion_threshold: int = 10,
                 min_track_confidence: float = 0.1):
        
        # Core components
        self.clusterer = AdaptiveHDBSCANClusterer(
            base_min_cluster_size=min_cluster_size,
            base_min_samples=min_samples,
            base_cluster_selection_epsilon=cluster_selection_epsilon
        )
        
        self.associator = RobustDataAssociator(
            max_association_distance=max_tracking_distance,
            mahalanobis_threshold=9.21,  # Chi-squared 95% for 3D
            confidence_weight=0.3,
            motion_state_bonus=0.2
        )
        
        # Track management
        self.tracks = []
        self.next_track_id = 1
        self.track_confirmation_threshold = track_confirmation_threshold
        self.track_deletion_threshold = track_deletion_threshold
        self.min_track_confidence = min_track_confidence
        
        # Processing statistics
        self.frame_count = 0
        self.total_detections = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        
        # Range filtering
        self.min_range = 0.5
        self.max_range = 50.0
        self.min_height = -5.0
        self.max_height = 5.0
        
    def preprocess_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Preprocess point cloud with range and height filtering"""
        if len(point_cloud) == 0:
            return point_cloud
        
        # Ensure we work with at least 3D data
        if point_cloud.shape[1] < 3:
            return np.array([]).reshape(0, point_cloud.shape[1])
        
        # Use only first 3 dimensions for position-based filtering
        position_data = point_cloud[:, :3]
        
        # Calculate range (distance from origin)
        ranges = np.linalg.norm(position_data[:, :2], axis=1)  # Use only x,y for range
        
        # Apply filters
        range_mask = (ranges >= self.min_range) & (ranges <= self.max_range)
        height_mask = (position_data[:, 2] >= self.min_height) & (position_data[:, 2] <= self.max_height)
        
        combined_mask = range_mask & height_mask
        
        # Return filtered point cloud with original dimensions
        return point_cloud[combined_mask]
    
    def extract_detections_from_clusters(self, point_cloud: np.ndarray, 
                                       cluster_labels: np.ndarray, 
                                       cluster_probabilities: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract detections from clustered points"""
        detections = []
        detection_covariances = []
        
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_points = point_cloud[cluster_mask]
            cluster_probs = cluster_probabilities[cluster_mask]
            
            if len(cluster_points) < 2:
                continue
            
            # IMPORTANT: Only use first 3 dimensions (x, y, z) for position tracking
            # This prevents dimension mismatch errors in covariance matrix operations
            cluster_points_3d = cluster_points[:, :3]
            
            # Calculate weighted centroid (3D position only)
            weights = cluster_probs / np.sum(cluster_probs)
            centroid = np.average(cluster_points_3d, axis=0, weights=weights)
            
            # Calculate covariance matrix (3D position only)
            diff = cluster_points_3d - centroid
            weighted_cov = np.cov(diff.T, aweights=weights)
            
            # Add minimum uncertainty to avoid singular matrices
            min_variance = 0.1
            if weighted_cov.ndim == 0:
                weighted_cov = np.eye(3) * min_variance
            elif weighted_cov.ndim == 2:
                # Ensure covariance is 3x3
                if weighted_cov.shape[0] == 3 and weighted_cov.shape[1] == 3:
                    weighted_cov += np.eye(3) * min_variance
                else:
                    # Fallback to identity matrix
                    weighted_cov = np.eye(3) * min_variance
            else:
                weighted_cov = np.eye(3) * min_variance
            
            detections.append(centroid)
            detection_covariances.append(weighted_cov)
        
        return detections, detection_covariances
    
    def process_frame(self, point_cloud: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Process a single frame of radar data"""
        start_time = time.time()
        
        # Preprocess point cloud
        filtered_points = self.preprocess_point_cloud(point_cloud)
        
        if len(filtered_points) == 0:
            # No valid points, just predict existing tracks
            for track in self.tracks:
                track.predict(timestamp)
            
            self._manage_tracks()
            
            return {
                'tracks': [track.get_track_info() for track in self.tracks],
                'detections': [],
                'associations': [],
                'processing_time': time.time() - start_time,
                'cluster_info': self.clusterer.get_cluster_info(),
                'association_stats': self.associator.get_association_statistics()
            }
        
        # Perform adaptive clustering
        cluster_labels, cluster_probabilities = self.clusterer.cluster_points(filtered_points, self.tracks)
        
        # Extract detections from clusters
        detections, detection_covariances = self.extract_detections_from_clusters(
            filtered_points, cluster_labels, cluster_probabilities
        )
        
        # Predict all tracks to current timestamp
        for track in self.tracks:
            track.predict(timestamp)
        
        # Data association
        associations, unassociated_tracks, unassociated_detections = \
            self.associator.associate_tracks_to_detections(self.tracks, detections, detection_covariances)
        
        # Update associated tracks
        for track_idx, detection_idx in associations:
            self.tracks[track_idx].update(
                detections[detection_idx], 
                timestamp, 
                detection_covariances[detection_idx]
            )
        
        # Create new tracks for unassociated detections
        for detection_idx in unassociated_detections:
            new_track = EnhancedTrack(
                self.next_track_id, 
                detections[detection_idx], 
                timestamp
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
            self.total_tracks_created += 1
        
        # Manage track lifecycle
        self._manage_tracks()
        
        # Update statistics
        self.frame_count += 1
        self.total_detections += len(detections)
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.detection_counts.append(len(detections))
        
        return {
            'tracks': [track.get_track_info() for track in self.tracks],
            'detections': [det.tolist() for det in detections],
            'associations': associations,
            'processing_time': processing_time,
            'cluster_info': self.clusterer.get_cluster_info(),
            'association_stats': self.associator.get_association_statistics(),
            'frame_stats': {
                'frame_count': self.frame_count,
                'total_detections': self.total_detections,
                'active_tracks': len(self.tracks),
                'confirmed_tracks': len([t for t in self.tracks if t.hits >= self.track_confirmation_threshold])
            }
        }
    
    def _manage_tracks(self):
        """Manage track lifecycle - deletion of poor quality tracks"""
        tracks_to_delete = []
        
        for i, track in enumerate(self.tracks):
            if track.should_delete(
                max_age=self.track_deletion_threshold * 2,
                max_misses=self.track_deletion_threshold,
                min_confidence=self.min_track_confidence
            ):
                tracks_to_delete.append(i)
        
        # Delete tracks in reverse order to maintain indices
        for i in reversed(tracks_to_delete):
            del self.tracks[i]
            self.total_tracks_deleted += 1
    
    def get_confirmed_tracks(self) -> List[EnhancedTrack]:
        """Get only confirmed tracks (tracks with sufficient hits)"""
        return [track for track in self.tracks if track.hits >= self.track_confirmation_threshold]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_deleted': self.total_tracks_deleted,
            'active_tracks': len(self.tracks),
            'confirmed_tracks': len(self.get_confirmed_tracks()),
            'avg_processing_time': np.mean(list(self.processing_times)) if self.processing_times else 0.0,
            'avg_detections_per_frame': np.mean(list(self.detection_counts)) if self.detection_counts else 0.0,
            'cluster_info': self.clusterer.get_cluster_info(),
            'association_stats': self.associator.get_association_statistics(),
            'motion_state_distribution': self._get_motion_state_distribution()
        }
    
    def _get_motion_state_distribution(self) -> Dict[str, int]:
        """Get distribution of motion states among active tracks"""
        distribution = {state.value: 0 for state in MotionState}
        
        for track in self.tracks:
            distribution[track.current_motion_state.value] += 1
        
        return distribution
    
    def update_parameters(self, **kwargs):
        """Update tracker parameters"""
        if 'max_tracking_distance' in kwargs:
            self.associator.max_association_distance = kwargs['max_tracking_distance']
        
        if 'min_cluster_size' in kwargs:
            self.clusterer.base_min_cluster_size = kwargs['min_cluster_size']
        
        if 'min_samples' in kwargs:
            self.clusterer.base_min_samples = kwargs['min_samples']
        
        if 'cluster_selection_epsilon' in kwargs:
            self.clusterer.base_cluster_selection_epsilon = kwargs['cluster_selection_epsilon']

if __name__ == "__main__":
    # Test the enhanced tracker
    tracker = EnhancedHDBSCANTracker()
    
    # Simulate some test data
    test_points = np.random.rand(20, 3) * 10
    result = tracker.process_frame(test_points, time.time())
    
    print("Enhanced HDBSCAN Tracker Test Results:")
    print(f"Detections: {len(result['detections'])}")
    print(f"Tracks: {len(result['tracks'])}")
    print(f"Processing time: {result['processing_time']:.4f}s")
    print("Motion state distribution:", result['frame_stats']) 