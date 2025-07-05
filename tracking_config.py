"""
Configuration file for HDBSCAN + Particle Filter tracking algorithm.

This file contains all the configurable parameters for the tracking system,
allowing easy tuning and experimentation without modifying the core code.
"""

# HDBSCAN Clustering Parameters
HDBSCAN_CONFIG = {
    # Minimum number of points required to form a cluster
    'min_cluster_size': 3,
    
    # Minimum number of points required for a point to be considered a core point
    'min_samples': 2,
    
    # Distance threshold for cluster selection (smaller = more clusters)
    'cluster_selection_epsilon': 0.5,
    
    # Clustering metric ('euclidean', 'manhattan', 'chebyshev', etc.)
    'metric': 'euclidean'
}

# Particle Filter Parameters
PARTICLE_FILTER_CONFIG = {
    # Number of particles per track
    'num_particles': 100,
    
    # Process noise standard deviation (motion uncertainty)
    'process_noise': 0.1,
    
    # Measurement noise standard deviation (sensor uncertainty)
    'measurement_noise': 0.5,
    
    # Maximum velocity allowed (m/s) - prevents unrealistic motion
    'max_velocity': 10.0,
    
    # Velocity damping factor (0.0 to 1.0)
    'velocity_damping': 0.95
}

# Data Association Parameters
TRACKING_CONFIG = {
    # Maximum distance for associating detections to tracks (meters)
    'max_tracking_distance': 2.0,
    
    # Minimum confidence for valid detections
    'min_detection_confidence': 0.3,
    
    # Maximum age of track without updates before deletion
    'max_track_age': 30,
    
    # Minimum number of hits before track is considered valid
    'min_track_hits': 3,
    
    # Minimum confidence threshold for track validation
    'min_track_confidence': 0.1
}

# Point Cloud Preprocessing Parameters
PREPROCESSING_CONFIG = {
    # Range limits for point cloud filtering (meters)
    'x_range': (-50, 50),      # Left/Right range
    'y_range': (-50, 50),      # Forward/Backward range  
    'z_range': (-2, 10),       # Up/Down range
    
    # Minimum number of points required for processing
    'min_points_threshold': 5,
    
    # Enable/disable range filtering
    'enable_range_filter': True,
    
    # Enable/disable statistical outlier removal
    'enable_outlier_removal': False,
    
    # Outlier removal parameters (if enabled)
    'outlier_std_threshold': 2.0,
    'outlier_neighbors': 20
}

# Visualization Parameters
VISUALIZATION_CONFIG = {
    # Track marker size in 3D plots
    'track_marker_size': 0.3,
    
    # Point cloud marker size
    'point_marker_size': 0.1,
    
    # Color palette for different tracks
    'track_colors': [
        (1, 0, 0, 1),    # Red
        (0, 1, 0, 1),    # Green  
        (0, 0, 1, 1),    # Blue
        (1, 1, 0, 1),    # Yellow
        (1, 0, 1, 1),    # Magenta
        (0, 1, 1, 1),    # Cyan
        (1, 0.5, 0, 1),  # Orange
        (0.5, 0, 1, 1),  # Purple
    ],
    
    # Point cloud color (RGBA)
    'point_cloud_color': (1, 1, 1, 0.5),  # White with transparency
    
    # Show track IDs as text labels
    'show_track_ids': False,
    
    # Show track velocities as arrows
    'show_velocity_vectors': False
}

# Performance Parameters
PERFORMANCE_CONFIG = {
    # Update rate for tracking algorithm (Hz)
    'tracking_update_rate': 20,
    
    # Enable performance profiling
    'enable_profiling': False,
    
    # Log tracking statistics every N frames
    'stats_log_interval': 20,
    
    # Maximum processing time warning threshold (seconds)
    'max_processing_time': 0.05
}

# Radar-Specific Parameters
RADAR_CONFIG = {
    # Expected radar update rate (Hz)
    'radar_frame_rate': 20,
    
    # Radar coordinate system
    # 'x_forward': True means X-axis points forward from radar
    # False means Y-axis points forward
    'x_forward': True,
    
    # Velocity measurement noise characteristics
    'doppler_noise_std': 0.2,
    
    # Range measurement noise characteristics  
    'range_noise_std': 0.1,
    
    # Angle measurement noise characteristics
    'angle_noise_std': 0.1
}

# Advanced Tracking Parameters
ADVANCED_CONFIG = {
    # Track prediction model ('constant_velocity', 'constant_acceleration')
    'motion_model': 'constant_velocity',
    
    # Enable adaptive particle filter (varies particle count based on uncertainty)
    'adaptive_particles': False,
    
    # Resampling threshold (effective sample size ratio)
    'resampling_threshold': 0.5,
    
    # Enable track interpolation for missing detections
    'enable_interpolation': True,
    
    # Maximum frames to interpolate
    'max_interpolation_frames': 5,
    
    # Enable track smoothing (backward pass)
    'enable_smoothing': False,
    
    # Smoothing window size
    'smoothing_window': 5
}

def get_config():
    """Get complete configuration dictionary"""
    return {
        'hdbscan': HDBSCAN_CONFIG,
        'particle_filter': PARTICLE_FILTER_CONFIG,
        'tracking': TRACKING_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'radar': RADAR_CONFIG,
        'advanced': ADVANCED_CONFIG
    }

def create_tracker_from_config(config=None):
    """Create tracker instance from configuration"""
    if config is None:
        config = get_config()
    
    from tracking import HDBSCANTracker
    
    tracker = HDBSCANTracker(
        min_cluster_size=config['hdbscan']['min_cluster_size'],
        min_samples=config['hdbscan']['min_samples'],
        cluster_selection_epsilon=config['hdbscan']['cluster_selection_epsilon'],
        max_tracking_distance=config['tracking']['max_tracking_distance'],
        min_detection_confidence=config['tracking']['min_detection_confidence']
    )
    
    return tracker

def print_config():
    """Print current configuration to console"""
    config = get_config()
    
    print("=== HDBSCAN + Particle Filter Tracking Configuration ===")
    print()
    
    for section_name, section_config in config.items():
        print(f"[{section_name.upper()}]")
        for key, value in section_config.items():
            print(f"  {key}: {value}")
        print()

if __name__ == "__main__":
    print_config() 