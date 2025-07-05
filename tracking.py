"""
Object Tracking Algorithm combining HDBSCAN clustering with Particle Filter
for mmWave radar data.

This module implements a comprehensive tracking system that:
1. Uses HDBSCAN to cluster radar point cloud data into objects
2. Calculates centroids of clusters to detect object positions
3. Uses particle filters to track objects over time
4. Handles data association and track management

Author: AI Assistant
Date: 2025
"""

import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetection:
    """Represents a single object detection with position and metadata"""
    
    def __init__(self, position, confidence=1.0, cluster_size=1, timestamp=None):
        self.position = np.array(position)  # [x, y, z] in meters
        self.confidence = confidence
        self.cluster_size = cluster_size
        self.timestamp = timestamp or time.time()
        
    def __repr__(self):
        return f"Detection(pos={self.position}, conf={self.confidence:.2f})"


class ParticleFilter:
    """Particle Filter implementation for tracking individual objects"""
    
    def __init__(self, initial_position, num_particles=100, process_noise=0.1, 
                 measurement_noise=0.5, track_id=None):
        """
        Initialize particle filter for object tracking
        
        Args:
            initial_position: Initial [x, y, z] position
            num_particles: Number of particles to use
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
            track_id: Unique identifier for this track
        """
        self.track_id = track_id
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State: [x, y, z, vx, vy, vz] - position and velocity
        self.state_dim = 6
        self.position_dim = 3
        
        # Initialize particles around initial position
        self.particles = np.zeros((num_particles, self.state_dim))
        self.particles[:, :3] = initial_position + np.random.normal(0, 0.1, (num_particles, 3))
        self.particles[:, 3:] = np.random.normal(0, 0.1, (num_particles, 3))  # velocity
        
        # Initialize weights
        self.weights = np.ones(num_particles) / num_particles
        
        # Tracking metadata
        self.last_update = time.time()
        self.age = 0
        self.hits = 1  # Number of successful updates
        self.time_since_update = 0
        self.confidence = 1.0
        
        # Prediction history for smoothing
        self.position_history = deque(maxlen=10)
        self.position_history.append(initial_position)
        
    def predict(self, dt=0.05):
        """Predict particle states forward in time"""
        # Simple constant velocity model with random acceleration
        self.particles[:, :3] += self.particles[:, 3:] * dt  # position update
        
        # Add process noise
        noise = np.random.normal(0, self.process_noise, self.particles.shape)
        self.particles += noise
        
        # Add some random acceleration to velocity
        accel_noise = np.random.normal(0, self.process_noise * 0.5, (self.num_particles, 3))
        self.particles[:, 3:6] += accel_noise * dt
        
        # Apply velocity damping to prevent unrealistic speeds
        max_velocity = 10.0  # m/s
        self.particles[:, 3:6] = np.clip(self.particles[:, 3:6], -max_velocity, max_velocity)
        
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection):
        """Update particle filter with new detection"""
        if detection is None:
            return
            
        measurement = detection.position
        
        # Calculate likelihood for each particle
        distances = np.linalg.norm(self.particles[:, :3] - measurement, axis=1)
        likelihoods = np.exp(-0.5 * (distances / self.measurement_noise) ** 2)
        
        # Weight by detection confidence
        likelihoods *= detection.confidence
        
        # Update weights
        self.weights *= likelihoods
        self.weights += 1e-30  # Avoid division by zero
        self.weights /= np.sum(self.weights)
        
        # Resample if needed (effective sample size < threshold)
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2:
            self.resample()
            
        # Update tracking metadata
        self.hits += 1
        self.time_since_update = 0
        self.last_update = time.time()
        self.confidence = min(1.0, self.confidence + 0.1)
        
        # Add to position history
        self.position_history.append(self.get_position())
        
    def resample(self):
        """Resample particles based on weights"""
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Add small amount of noise to avoid particle depletion
        noise = np.random.normal(0, self.process_noise * 0.1, self.particles.shape)
        self.particles += noise
        
    def get_position(self):
        """Get estimated position (weighted average)"""
        return np.average(self.particles[:, :3], weights=self.weights, axis=0)
        
    def get_velocity(self):
        """Get estimated velocity (weighted average)"""
        return np.average(self.particles[:, 3:6], weights=self.weights, axis=0)
        
    def get_state_covariance(self):
        """Get state covariance matrix"""
        mean_state = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - mean_state
        cov = np.average(diff[:, None, :] * diff[:, :, None], weights=self.weights, axis=0)
        return cov
        
    def is_valid(self, max_age=30, min_hits=3):
        """Check if track is still valid"""
        return (self.time_since_update <= max_age and 
                self.hits >= min_hits and 
                self.confidence > 0.1)


class HDBSCANTracker:
    """Main tracking algorithm combining HDBSCAN clustering with particle filters"""
    
    def __init__(self, min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.5,
                 max_tracking_distance=2.0, min_detection_confidence=0.3):
        """
        Initialize the HDBSCAN-based tracker
        
        Args:
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum samples for HDBSCAN core points
            cluster_selection_epsilon: HDBSCAN epsilon parameter
            max_tracking_distance: Maximum distance for data association
            min_detection_confidence: Minimum confidence for valid detections
        """
        # HDBSCAN parameters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        
        # Tracking parameters
        self.max_tracking_distance = max_tracking_distance
        self.min_detection_confidence = min_detection_confidence
        
        # Active tracks
        self.tracks = {}  # track_id -> ParticleFilter
        self.next_track_id = 1
        
        # Performance metrics
        self.frame_count = 0
        self.detection_count = 0
        self.cluster_stats = deque(maxlen=100)
        
        logger.info("HDBSCANTracker initialized")
        
    def preprocess_point_cloud(self, point_cloud_data):
        """
        Preprocess point cloud data for clustering
        
        Args:
            point_cloud_data: Array of points, expected shape (N, 3) for [x, y, z]
            
        Returns:
            Filtered and normalized point cloud
        """
        if point_cloud_data is None or len(point_cloud_data) == 0:
            return np.array([]).reshape(0, 3)
            
        points = np.array(point_cloud_data)
        
        # Ensure we have 3D points
        if points.shape[1] < 3:
            # Pad with zeros if we only have 2D points
            if points.shape[1] == 2:
                points = np.column_stack([points, np.zeros(points.shape[0])])
            else:
                logger.warning(f"Invalid point cloud shape: {points.shape}")
                return np.array([]).reshape(0, 3)
        elif points.shape[1] > 3:
            # Take only first 3 dimensions if we have more
            points = points[:, :3]
            
        # Remove invalid points (NaN, inf)
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        
        # Apply basic range filtering (adjust based on your sensor setup)
        range_filter = (
            (np.abs(points[:, 0]) < 50) &  # x range: -50m to +50m
            (np.abs(points[:, 1]) < 50) &  # y range: -50m to +50m  
            (points[:, 2] > -2) & (points[:, 2] < 10)  # z range: -2m to +10m
        )
        points = points[range_filter]
        
        return points
        
    def cluster_points(self, points):
        """
        Apply HDBSCAN clustering to detect objects
        
        Args:
            points: Preprocessed point cloud array (N, 3)
            
        Returns:
            List of ObjectDetection instances
        """
        if len(points) < self.min_cluster_size:
            return []
            
        try:
            # Apply HDBSCAN clustering
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='euclidean'
            )
            
            cluster_labels = clusterer.fit_predict(points)
            
            # Extract valid clusters (label != -1 means not noise)
            detections = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                # Get points belonging to this cluster
                cluster_mask = cluster_labels == label
                cluster_points = points[cluster_mask]
                
                if len(cluster_points) < self.min_cluster_size:
                    continue
                    
                # Calculate cluster centroid
                centroid = np.mean(cluster_points, axis=0)
                
                # Calculate cluster confidence based on size and density
                cluster_size = len(cluster_points)
                cluster_std = np.std(cluster_points, axis=0)
                density = cluster_size / (np.prod(cluster_std + 1e-6))
                confidence = min(1.0, (cluster_size / 10.0) * (density / 100.0))
                confidence = max(self.min_detection_confidence, confidence)
                
                detection = ObjectDetection(
                    position=centroid,
                    confidence=confidence,
                    cluster_size=cluster_size,
                    timestamp=time.time()
                )
                detections.append(detection)
                
            # Store cluster statistics
            n_clusters = len(detections)
            n_noise = np.sum(cluster_labels == -1)
            self.cluster_stats.append({
                'n_points': len(points),
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'cluster_ratio': n_clusters / len(points) if len(points) > 0 else 0
            })
            
            logger.debug(f"Clustered {len(points)} points into {n_clusters} objects ({n_noise} noise points)")
            return detections
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return []
            
    def associate_detections_to_tracks(self, detections):
        """
        Associate new detections with existing tracks using Hungarian algorithm
        
        Args:
            detections: List of ObjectDetection instances
            
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())
            
        # Get predicted positions for all tracks
        track_ids = list(self.tracks.keys())
        track_positions = np.array([self.tracks[tid].get_position() for tid in track_ids])
        detection_positions = np.array([det.position for det in detections])
        
        # Calculate cost matrix (distances)
        cost_matrix = cdist(detection_positions, track_positions)
        
        # Apply distance threshold - set high cost for distances > threshold
        cost_matrix[cost_matrix > self.max_tracking_distance] = 1e6
        
        # Solve assignment problem
        detection_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments with cost > threshold
        valid_assignments = []
        for det_idx, track_idx in zip(detection_indices, track_indices):
            if cost_matrix[det_idx, track_idx] < self.max_tracking_distance:
                valid_assignments.append((det_idx, track_ids[track_idx]))
                
        # Find unmatched detections and tracks
        matched_det_indices = {pair[0] for pair in valid_assignments}
        matched_track_ids = {pair[1] for pair in valid_assignments}
        
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_ids]
        
        return valid_assignments, unmatched_detections, unmatched_tracks
        
    def update_tracks(self, detections):
        """Update existing tracks and create new ones"""
        # Predict all tracks forward
        dt = 0.05  # Assume ~20Hz update rate
        for track in self.tracks.values():
            track.predict(dt)
            
        # Associate detections to tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self.associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            self.tracks[track_id].update(detections[det_idx])
            
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            if detection.confidence >= self.min_detection_confidence:
                new_track = ParticleFilter(
                    initial_position=detection.position,
                    track_id=self.next_track_id
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
                logger.debug(f"Created new track {new_track.track_id}")
                
        # Remove invalid tracks
        tracks_to_remove = []
        for track_id in unmatched_tracks:
            if not self.tracks[track_id].is_valid():
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            logger.debug(f"Removing invalid track {track_id}")
            del self.tracks[track_id]
            
    def process_frame(self, point_cloud_data):
        """
        Process a single frame of point cloud data
        
        Args:
            point_cloud_data: Raw point cloud data from radar
            
        Returns:
            Dictionary with tracking results
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Preprocess point cloud
        processed_points = self.preprocess_point_cloud(point_cloud_data)
        
        # Detect objects using HDBSCAN
        detections = self.cluster_points(processed_points)
        self.detection_count += len(detections)
        
        # Update tracks
        self.update_tracks(detections)
        
        # Prepare results
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track.is_valid():
                active_tracks.append({
                    'id': track_id,
                    'position': track.get_position(),
                    'velocity': track.get_velocity(),
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.hits
                })
                
        processing_time = time.time() - start_time
        
        results = {
            'frame_id': self.frame_count,
            'processing_time': processing_time,
            'n_input_points': len(point_cloud_data) if point_cloud_data is not None else 0,
            'n_processed_points': len(processed_points),
            'n_detections': len(detections),
            'n_active_tracks': len(active_tracks),
            'tracks': active_tracks,
            'detections': [{'position': det.position.tolist(), 'confidence': det.confidence} 
                          for det in detections]
        }
        
        if self.frame_count % 20 == 0:  # Log every 20 frames
            logger.info(f"Frame {self.frame_count}: {len(active_tracks)} tracks, "
                       f"{len(detections)} detections, {processing_time*1000:.1f}ms")
                       
        return results
        
    def get_statistics(self):
        """Get tracking performance statistics"""
        if not self.cluster_stats:
            return {}
            
        recent_stats = list(self.cluster_stats)[-20:]  # Last 20 frames
        
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.detection_count,
            'active_tracks': len(self.tracks),
            'avg_points_per_frame': np.mean([s['n_points'] for s in recent_stats]),
            'avg_clusters_per_frame': np.mean([s['n_clusters'] for s in recent_stats]),
            'avg_cluster_ratio': np.mean([s['cluster_ratio'] for s in recent_stats]),
            'track_ages': [track.age for track in self.tracks.values()],
            'track_confidences': [track.confidence for track in self.tracks.values()]
        }


# Factory function for easy integration
def create_tracker(**kwargs):
    """Create and return a configured HDBSCANTracker instance"""
    return HDBSCANTracker(**kwargs)


if __name__ == "__main__":
    # Example usage and testing
    tracker = create_tracker()
    
    # Generate some test data
    np.random.seed(42)
    test_points = np.random.normal(0, 5, (50, 3))
    
    result = tracker.process_frame(test_points)
    print("Test result:", result)
    print("Statistics:", tracker.get_statistics()) 