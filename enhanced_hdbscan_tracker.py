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
                 base_min_cluster_size: int = 8,  # Increased from 3 for automotive
                 base_min_samples: int = 4,      # Increased from 2 for automotive
                 base_cluster_selection_epsilon: float = 1.0,  # Increased from 0.5 for automotive
                 density_adaptation_factor: float = 0.3,
                 motion_adaptation_factor: float = 0.2,
                 min_detection_points: int = 5,   # NEW: Minimum points to form detection
                 min_cluster_probability: float = 0.6):  # NEW: Minimum cluster probability
        
        self.base_min_cluster_size = base_min_cluster_size
        self.base_min_samples = base_min_samples
        self.base_cluster_selection_epsilon = base_cluster_selection_epsilon
        self.density_adaptation_factor = density_adaptation_factor
        self.motion_adaptation_factor = motion_adaptation_factor
        
        # NEW: Detection quality filters
        self.min_detection_points = min_detection_points
        self.min_cluster_probability = min_cluster_probability
        
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
        
        # NEW: Debug counters
        self.debug_stats = {
            'raw_clusters': 0,
            'filtered_clusters': 0,
            'points_filtered': 0,
            'probability_filtered': 0
        }
        
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
        
        # Adapt based on point density - MORE CONSERVATIVE for automotive
        if len(self.point_density_history) >= 3:
            density_trend = np.mean(list(self.point_density_history)[-3:])
            
            # If density is very high, slightly reduce cluster size
            if density_trend > 20.0:
                self.current_min_cluster_size = max(6, self.base_min_cluster_size - 2)
                self.current_min_samples = max(3, self.base_min_samples - 1)
            # If density is low, require larger clusters
            elif density_trend < 5.0:
                self.current_min_cluster_size = self.base_min_cluster_size + 2
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
        
        # Clamp parameters to reasonable ranges for automotive
        self.current_min_cluster_size = np.clip(self.current_min_cluster_size, 5, 20)
        self.current_min_samples = np.clip(self.current_min_samples, 2, 10)
        self.current_epsilon = np.clip(self.current_epsilon, 0.3, 3.0)
    
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
        
        # Reset debug stats
        self.debug_stats['raw_clusters'] = self.last_cluster_count
        
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
            'avg_point_density': np.mean(list(self.point_density_history)) if self.point_density_history else 0.0,
            'debug_stats': self.debug_stats.copy()
        }

class RobustDataAssociator:
    """Robust data association using Mahalanobis distance and track confidence"""
    
    def __init__(self, 
                 max_association_distance: float = 6.0,  # Increased from 5.0 for automotive
                 mahalanobis_threshold: float = 12.0,    # Increased from 9.21 for automotive
                 confidence_weight: float = 0.3,
                 motion_state_bonus: float = 0.2,
                 velocity_gate_factor: float = 2.0):     # NEW: Velocity-based gating
        
        self.max_association_distance = max_association_distance
        self.mahalanobis_threshold = mahalanobis_threshold
        self.confidence_weight = confidence_weight
        self.motion_state_bonus = motion_state_bonus
        self.velocity_gate_factor = velocity_gate_factor
        
        # Association statistics
        self.association_distances = deque(maxlen=100)
        self.association_success_rate = deque(maxlen=50)
        
        # NEW: Debug counters
        self.debug_stats = {
            'total_associations_attempted': 0,
            'successful_associations': 0,
            'distance_rejected': 0,
            'mahalanobis_rejected': 0
        }
        
    def calculate_association_cost(self, track: EnhancedTrack, detection: np.ndarray, 
                                 detection_covariance: np.ndarray, dt: float = 0.1) -> float:
        """Calculate comprehensive association cost between track and detection"""
        
        # NEW: Predict track position using velocity (velocity-aware gating)
        predicted_pos = track.position + track.velocity * dt
        
        # Calculate Mahalanobis distance
        innovation = detection - predicted_pos  # Use predicted position, not current
        
        # Combined covariance: track uncertainty + detection uncertainty
        track_pos_cov = track.current_covariance[0:3, 0:3]
        combined_cov = track_pos_cov + detection_covariance
        
        try:
            mahal_dist_squared = innovation.T @ la.inv(combined_cov) @ innovation
            mahal_dist = np.sqrt(mahal_dist_squared)
        except la.LinAlgError:
            mahal_dist = 100.0  # Very high cost for singular covariance
            self.debug_stats['mahalanobis_rejected'] += 1
        
        # Base cost from Mahalanobis distance
        if mahal_dist > self.mahalanobis_threshold:
            self.debug_stats['mahalanobis_rejected'] += 1
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
                expected_pos = track.position + track.velocity * dt
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
                                     detection_covariances: List[np.ndarray],
                                     dt: float = 0.1) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks to detections using Hungarian algorithm with robust cost function"""
        
        # Reset debug counters
        self.debug_stats['total_associations_attempted'] = len(tracks) * len(detections)
        self.debug_stats['successful_associations'] = 0
        self.debug_stats['distance_rejected'] = 0
        self.debug_stats['mahalanobis_rejected'] = 0
        
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Build cost matrix
        cost_matrix = np.full((len(tracks), len(detections)), float('inf'))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                detection_cov = detection_covariances[j] if j < len(detection_covariances) else np.eye(3)
                
                # NEW: Use velocity-aware distance check
                predicted_pos = track.position + track.velocity * dt
                euclidean_dist = np.linalg.norm(detection - predicted_pos)
                
                # Only consider associations within maximum distance
                if euclidean_dist <= self.max_association_distance:
                    cost = self.calculate_association_cost(track, detection, detection_cov, dt)
                    cost_matrix[i, j] = cost
                else:
                    self.debug_stats['distance_rejected'] += 1
        
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
            return self._fallback_association(tracks, detections, dt)
        
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
        self.debug_stats['successful_associations'] = len(valid_associations)
        success_rate = len(valid_associations) / len(tracks) if tracks else 0
        self.association_success_rate.append(success_rate)
        
        return valid_associations, unassociated_tracks, unassociated_detections
    
    def _fallback_association(self, tracks: List[EnhancedTrack], 
                            detections: List[np.ndarray],
                            dt: float = 0.1) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Fallback association method using simple nearest neighbor"""
        associations = []
        used_detections = set()
        
        for i, track in enumerate(tracks):
            best_detection = None
            best_distance = float('inf')
            
            for j, detection in enumerate(detections):
                if j in used_detections:
                    continue
                
                # Use velocity-aware distance
                predicted_pos = track.position + track.velocity * dt
                distance = np.linalg.norm(detection - predicted_pos)
                
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
            'mahalanobis_threshold': self.mahalanobis_threshold,
            'debug_stats': self.debug_stats.copy()
        }

class EnhancedHDBSCANTracker:
    """Enhanced HDBSCAN tracker with IMM filtering and robust data association"""
    
    def __init__(self, 
                 max_tracking_distance: float = 6.0,        # Increased from 5.0
                 min_cluster_size: int = 8,                 # Increased from 3
                 min_samples: int = 4,                      # Increased from 2
                 cluster_selection_epsilon: float = 1.0,    # Increased from 0.5
                 track_confirmation_threshold: int = 3,
                 track_deletion_time: float = 2.0,          # NEW: Time-based deletion (seconds)
                 track_coast_time: float = 1.0,             # NEW: Coast time before deletion
                 min_track_confidence: float = 0.1,
                 min_detection_points: int = 5,             # NEW: Minimum points for detection
                 min_cluster_probability: float = 0.6,      # NEW: Minimum cluster probability
                 expected_fps: float = 10.0):               # NEW: Expected radar FPS
        
        # Core components
        self.clusterer = AdaptiveHDBSCANClusterer(
            base_min_cluster_size=min_cluster_size,
            base_min_samples=min_samples,
            base_cluster_selection_epsilon=cluster_selection_epsilon,
            min_detection_points=min_detection_points,
            min_cluster_probability=min_cluster_probability
        )
        
        self.associator = RobustDataAssociator(
            max_association_distance=max_tracking_distance,
            mahalanobis_threshold=12.0,  # Increased for automotive
            confidence_weight=0.3,
            motion_state_bonus=0.2,
            velocity_gate_factor=2.0
        )
        
        # Track management - NEW: Time-based parameters
        self.tracks = []
        self.next_track_id = 1
        self.track_confirmation_threshold = track_confirmation_threshold
        self.track_deletion_time = track_deletion_time
        self.track_coast_time = track_coast_time
        self.min_track_confidence = min_track_confidence
        self.expected_fps = expected_fps
        
        # NEW: Detection quality parameters
        self.min_detection_points = min_detection_points
        self.min_cluster_probability = min_cluster_probability
        
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
        
        # NEW: Debug counters
        self.debug_stats = {
            'points_received': 0,
            'points_after_filter': 0,
            'raw_clusters': 0,
            'quality_filtered_clusters': 0,
            'final_detections': 0,
            'associations_attempted': 0,
            'successful_associations': 0,
            'new_tracks_created': 0,
            'tracks_deleted': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0
        }
        
    def preprocess_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Preprocess point cloud with range and height filtering"""
        self.debug_stats['points_received'] = len(point_cloud)
        
        if len(point_cloud) == 0:
            self.debug_stats['points_after_filter'] = 0
            return point_cloud
        
        # Ensure we work with at least 3D data
        if point_cloud.shape[1] < 3:
            self.debug_stats['points_after_filter'] = 0
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
        filtered_points = point_cloud[combined_mask]
        self.debug_stats['points_after_filter'] = len(filtered_points)
        return filtered_points
    
    def extract_detections_from_clusters(self, point_cloud: np.ndarray, 
                                       cluster_labels: np.ndarray, 
                                       cluster_probabilities: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract detections from clustered points with quality filtering"""
        detections = []
        detection_covariances = []
        
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label
        
        self.debug_stats['raw_clusters'] = len(unique_labels)
        quality_filtered_count = 0
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_points = point_cloud[cluster_mask]
            cluster_probs = cluster_probabilities[cluster_mask]
            
            # NEW: Quality filtering - minimum points check
            if len(cluster_points) < self.min_detection_points:
                self.clusterer.debug_stats['points_filtered'] += 1
                continue
            
            # NEW: Quality filtering - probability check
            avg_probability = np.mean(cluster_probs)
            if avg_probability < self.min_cluster_probability:
                self.clusterer.debug_stats['probability_filtered'] += 1
                continue
            
            quality_filtered_count += 1
            
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
            min_variance = 0.25  # Increased from 0.1 for automotive
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
        
        self.debug_stats['quality_filtered_clusters'] = quality_filtered_count
        self.debug_stats['final_detections'] = len(detections)
        self.clusterer.debug_stats['filtered_clusters'] = quality_filtered_count
        
        return detections, detection_covariances
    
    def process_frame(self, point_cloud: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Process a single frame of radar data"""
        start_time = time.time()
        
        # Calculate dt from last processing
        if hasattr(self, 'last_timestamp'):
            dt = timestamp - self.last_timestamp
        else:
            dt = 1.0 / self.expected_fps
        self.last_timestamp = timestamp
        
        # Preprocess point cloud
        filtered_points = self.preprocess_point_cloud(point_cloud)
        
        if len(filtered_points) == 0:
            # No valid points, just predict existing tracks
            for track in self.tracks:
                track.predict(timestamp)
            
            self._manage_tracks(timestamp)
            
            # Update debug stats
            self.debug_stats['active_tracks'] = len(self.tracks)
            self.debug_stats['confirmed_tracks'] = len([t for t in self.tracks if t.hits >= self.track_confirmation_threshold])
            
            return {
                'tracks': [track.get_track_info() for track in self.tracks],
                'detections': [],
                'associations': [],
                'processing_time': time.time() - start_time,
                'cluster_info': self.clusterer.get_cluster_info(),
                'association_stats': self.associator.get_association_statistics(),
                'debug_stats': self.debug_stats.copy()
            }
        
        # Perform adaptive clustering
        cluster_labels, cluster_probabilities = self.clusterer.cluster_points(filtered_points, self.tracks)
        
        # Extract detections from clusters with quality filtering
        detections, detection_covariances = self.extract_detections_from_clusters(
            filtered_points, cluster_labels, cluster_probabilities
        )
        
        # Predict all tracks to current timestamp
        for track in self.tracks:
            track.predict(timestamp)
        
        # Data association with velocity-aware gating
        associations, unassociated_tracks, unassociated_detections = \
            self.associator.associate_tracks_to_detections(self.tracks, detections, detection_covariances, dt)
        
        # Update debug stats
        self.debug_stats['associations_attempted'] = len(self.tracks) * len(detections)
        self.debug_stats['successful_associations'] = len(associations)
        
        # Update associated tracks
        for track_idx, detection_idx in associations:
            self.tracks[track_idx].update(
                detections[detection_idx], 
                timestamp, 
                detection_covariances[detection_idx]
            )
        
        # Create new tracks for unassociated detections
        new_tracks_created = 0
        for detection_idx in unassociated_detections:
            new_track = EnhancedTrack(
                self.next_track_id, 
                detections[detection_idx], 
                timestamp
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
            self.total_tracks_created += 1
            new_tracks_created += 1
        
        self.debug_stats['new_tracks_created'] = new_tracks_created
        
        # Manage track lifecycle with time-based deletion
        tracks_deleted = self._manage_tracks(timestamp)
        self.debug_stats['tracks_deleted'] = tracks_deleted
        
        # Update statistics
        self.frame_count += 1
        self.total_detections += len(detections)
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.detection_counts.append(len(detections))
        
        # Update debug stats
        self.debug_stats['active_tracks'] = len(self.tracks)
        self.debug_stats['confirmed_tracks'] = len([t for t in self.tracks if t.hits >= self.track_confirmation_threshold])
        
        # Print debug info every 20 frames
        if self.frame_count % 20 == 0:
            print(f"TRACKER DEBUG Frame {self.frame_count}: "
                  f"Points: {self.debug_stats['points_received']}→{self.debug_stats['points_after_filter']}, "
                  f"Clusters: {self.debug_stats['raw_clusters']}→{self.debug_stats['quality_filtered_clusters']}, "
                  f"Detections: {self.debug_stats['final_detections']}, "
                  f"Associations: {self.debug_stats['successful_associations']}/{len(self.tracks)}, "
                  f"Tracks: {self.debug_stats['active_tracks']} ({self.debug_stats['confirmed_tracks']} confirmed), "
                  f"New: {self.debug_stats['new_tracks_created']}, Deleted: {self.debug_stats['tracks_deleted']}")
        
        return {
            'tracks': [track.get_track_info() for track in self.tracks],
            'detections': [det.tolist() for det in detections],
            'associations': associations,
            'processing_time': processing_time,
            'cluster_info': self.clusterer.get_cluster_info(),
            'association_stats': self.associator.get_association_statistics(),
            'debug_stats': self.debug_stats.copy(),
            'frame_stats': {
                'frame_count': self.frame_count,
                'total_detections': self.total_detections,
                'active_tracks': len(self.tracks),
                'confirmed_tracks': len([t for t in self.tracks if t.hits >= self.track_confirmation_threshold])
            }
        }
    
    def _manage_tracks(self, current_time: float) -> int:
        """Manage track lifecycle - TIME-BASED deletion of poor quality tracks"""
        tracks_to_delete = []
        
        for i, track in enumerate(self.tracks):
            # NEW: Time-based deletion logic
            time_since_update = current_time - track.last_update_time
            
            # Delete if track has been coasting too long
            if time_since_update > self.track_coast_time:
                tracks_to_delete.append(i)
            # Or if track has been alive too long with poor confidence
            elif (current_time - track.created_time) > self.track_deletion_time and track.confidence < self.min_track_confidence:
                tracks_to_delete.append(i)
        
        # Delete tracks in reverse order to maintain indices
        for i in reversed(tracks_to_delete):
            del self.tracks[i]
            self.total_tracks_deleted += 1
        
        return len(tracks_to_delete)
    
    def get_confirmed_tracks(self) -> List[EnhancedTrack]:
        """Get only confirmed tracks (tracks with sufficient hits AND time)"""
        current_time = time.time()
        return [track for track in self.tracks 
                if track.hits >= self.track_confirmation_threshold 
                and (current_time - track.created_time) >= 0.3]  # At least 300ms old
    
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
            'motion_state_distribution': self._get_motion_state_distribution(),
            'debug_stats': self.debug_stats.copy()
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
        
        # NEW: Additional parameters
        if 'track_coast_time' in kwargs:
            self.track_coast_time = kwargs['track_coast_time']
        
        if 'track_deletion_time' in kwargs:
            self.track_deletion_time = kwargs['track_deletion_time']
        
        if 'min_detection_points' in kwargs:
            self.min_detection_points = kwargs['min_detection_points']
            self.clusterer.min_detection_points = kwargs['min_detection_points']
        
        if 'min_cluster_probability' in kwargs:
            self.min_cluster_probability = kwargs['min_cluster_probability']
            self.clusterer.min_cluster_probability = kwargs['min_cluster_probability']

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