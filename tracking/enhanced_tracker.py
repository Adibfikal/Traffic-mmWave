"""
Enhanced HDBSCAN tracker with IMM filtering and robust data association.
This module coordinates all tracking components for radar object tracking.
"""

import numpy as np
import time
from collections import deque
from typing import List, Tuple, Dict, Any

# Import the tracking components
from .adaptive_clustering import AdaptiveHDBSCANClusterer
from .data_association import RobustDataAssociator
from enhanced_tracking import EnhancedTrack


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
        """Manage track lifecycle with time-based deletion"""
        tracks_deleted = 0
        tracks_to_remove = []
        
        for i, track in enumerate(self.tracks):
            # Time-based track deletion
            time_since_last_update = current_time - track.last_update_time
            
            # Delete tracks that haven't been updated for too long
            if time_since_last_update > self.track_deletion_time:
                tracks_to_remove.append(i)
                tracks_deleted += 1
                continue
            
            # Delete tracks with very low confidence
            if track.confidence < self.min_track_confidence:
                tracks_to_remove.append(i)
                tracks_deleted += 1
                continue
        
        # Remove tracks in reverse order to maintain indices
        for i in reversed(tracks_to_remove):
            self.tracks.pop(i)
            self.total_tracks_deleted += 1
        
        return tracks_deleted
    
    def get_confirmed_tracks(self) -> List[EnhancedTrack]:
        """Get list of confirmed tracks"""
        return [track for track in self.tracks 
                if track.hits >= self.track_confirmation_threshold]
    
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
            'debug_stats': self.debug_stats.copy(),
            'motion_state_distribution': self._get_motion_state_distribution()
        }
    
    def _get_motion_state_distribution(self) -> Dict[str, int]:
        """Get distribution of motion states among active tracks"""
        distribution = {
            'stationary': 0,
            'constant_velocity': 0,
            'maneuvering': 0
        }
        
        for track in self.tracks:
            state_name = track.current_motion_state.name.lower()
            if state_name in distribution:
                distribution[state_name] += 1
                
        return distribution
    
    def update_parameters(self, **kwargs):
        """Update tracker parameters dynamically"""
        # Update clusterer parameters
        self.clusterer.update_parameters(**kwargs)
        
        # Update associator parameters
        self.associator.update_parameters(**kwargs)
        
        # Update tracker-specific parameters
        if 'track_confirmation_threshold' in kwargs:
            self.track_confirmation_threshold = kwargs['track_confirmation_threshold']
        if 'track_deletion_time' in kwargs:
            self.track_deletion_time = kwargs['track_deletion_time']
        if 'track_coast_time' in kwargs:
            self.track_coast_time = kwargs['track_coast_time']
        if 'min_track_confidence' in kwargs:
            self.min_track_confidence = kwargs['min_track_confidence']
        if 'expected_fps' in kwargs:
            self.expected_fps = kwargs['expected_fps']
        if 'min_detection_points' in kwargs:
            self.min_detection_points = kwargs['min_detection_points']
        if 'min_cluster_probability' in kwargs:
            self.min_cluster_probability = kwargs['min_cluster_probability'] 