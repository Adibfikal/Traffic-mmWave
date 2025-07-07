"""
Adaptive HDBSCAN clustering for radar tracking.
This module provides dynamic parameter adjustment for HDBSCAN clustering.
"""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import HDBSCAN
from collections import deque
from typing import List, Tuple, Dict, Any

# Import the enhanced tracking module that defines EnhancedTrack and MotionState
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
    
    def update_parameters(self, **kwargs):
        """Update clustering parameters"""
        if 'min_cluster_size' in kwargs:
            self.base_min_cluster_size = kwargs['min_cluster_size']
            self.current_min_cluster_size = self.base_min_cluster_size
            
        if 'min_samples' in kwargs:
            self.base_min_samples = kwargs['min_samples']
            self.current_min_samples = self.base_min_samples
            
        if 'cluster_selection_epsilon' in kwargs:
            self.base_cluster_selection_epsilon = kwargs['cluster_selection_epsilon']
            self.current_epsilon = self.base_cluster_selection_epsilon
            
        if 'min_detection_points' in kwargs:
            self.min_detection_points = kwargs['min_detection_points']
            
        if 'min_cluster_probability' in kwargs:
            self.min_cluster_probability = kwargs['min_cluster_probability'] 