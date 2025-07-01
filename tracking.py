import numpy as np
from sklearn.cluster import HDBSCAN
from filterpy.monte_carlo import systematic_resample
from numpy.random import randn, uniform
import time

class ParticleFilter:
    """Particle filter for single object tracking"""
    
    def __init__(self, num_particles=1000, dim_x=6):
        """
        Initialize particle filter
        dim_x: state dimension [x, y, z, vx, vy, vz]
        """
        self.num_particles = num_particles
        self.dim_x = dim_x
        self.particles = np.zeros((num_particles, dim_x))
        self.weights = np.ones(num_particles) / num_particles
        self.initialized = False
        
        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])  # position and velocity noise
        
        # Measurement noise
        self.R = np.diag([0.2, 0.2, 0.2])  # measurement noise for x, y, z
        
    def initialize(self, measurement, init_var=1.0):
        """Initialize particles around first measurement"""
        for i in range(self.num_particles):
            self.particles[i, 0] = measurement[0] + randn() * init_var
            self.particles[i, 1] = measurement[1] + randn() * init_var
            self.particles[i, 2] = measurement[2] + randn() * init_var
            self.particles[i, 3:] = randn(3) * 0.1  # small initial velocity
            
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True
        
    def predict(self, dt=0.1):
        """Predict step - move particles according to motion model"""
        if not self.initialized:
            return
            
        # Simple constant velocity model
        for i in range(self.num_particles):
            # Update position
            self.particles[i, 0] += self.particles[i, 3] * dt + randn() * self.Q[0, 0]
            self.particles[i, 1] += self.particles[i, 4] * dt + randn() * self.Q[1, 1]
            self.particles[i, 2] += self.particles[i, 5] * dt + randn() * self.Q[2, 2]
            
            # Update velocity with noise
            self.particles[i, 3] += randn() * self.Q[3, 3]
            self.particles[i, 4] += randn() * self.Q[4, 4]
            self.particles[i, 5] += randn() * self.Q[5, 5]
            
    def update(self, measurement):
        """Update step - weight particles based on measurement"""
        if not self.initialized:
            self.initialize(measurement)
            return
            
        # Calculate weights based on distance to measurement
        for i in range(self.num_particles):
            diff = measurement - self.particles[i, :3]
            self.weights[i] = np.exp(-0.5 * np.dot(diff, np.linalg.inv(self.R).dot(diff)))
            
        # Normalize weights
        self.weights += 1.e-300  # avoid zeros
        self.weights /= np.sum(self.weights)
        
        # Resample if effective sample size is too low
        neff = 1. / np.sum(np.square(self.weights))
        if neff < self.num_particles / 2:
            self.resample()
            
    def resample(self):
        """Resample particles based on weights"""
        indices = systematic_resample(self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def estimate(self):
        """Get state estimate from particles"""
        if not self.initialized:
            return None
            
        # Weighted mean of particles
        mean = np.average(self.particles, weights=self.weights, axis=0)
        return mean
        
    def get_position(self):
        """Get estimated position"""
        est = self.estimate()
        return est[:3] if est is not None else None


class ObjectTracker:
    """Multi-object tracker using HDBSCAN clustering and Particle Filters"""
    
    def __init__(self, min_cluster_size=3, min_samples=2):
        """Initialize tracker"""
        self.clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        self.tracks = {}  # Dictionary of track_id: ParticleFilter
        self.next_track_id = 0
        self.max_distance = 2.0  # Maximum distance for association
        self.max_frames_missing = 10  # Delete track after this many frames without update
        
        # Track management
        self.track_updates = {}  # track_id: last_update_time
        self.frames_missing = {}  # track_id: frames_missing_count
        
    def update(self, point_cloud):
        """Update tracker with new point cloud data"""
        if point_cloud is None or len(point_cloud) == 0:
            # No detections, predict existing tracks
            self._predict_tracks()
            self._cleanup_tracks()
            return self._get_track_states()
            
        # Cluster points using HDBSCAN
        clusters = self.clusterer.fit_predict(point_cloud)
        
        # Get cluster centers (ignoring noise points with label -1)
        cluster_centers = []
        cluster_labels = []
        
        unique_labels = set(clusters)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            mask = clusters == label
            cluster_points = point_cloud[mask]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
            cluster_labels.append(label)
            
        cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([]).reshape(0, 3)
        
        # Associate clusters with existing tracks
        associations = self._associate_detections(cluster_centers)
        
        # Update existing tracks
        for cluster_idx, track_id in associations.items():
            if track_id in self.tracks:
                self.tracks[track_id].predict()
                self.tracks[track_id].update(cluster_centers[cluster_idx])
                self.frames_missing[track_id] = 0
                
        # Create new tracks for unassociated clusters
        associated_clusters = set(associations.keys())
        for i, center in enumerate(cluster_centers):
            if i not in associated_clusters:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracks[track_id] = ParticleFilter()
                self.tracks[track_id].initialize(center)
                self.frames_missing[track_id] = 0
                
        # Predict tracks that weren't updated
        associated_tracks = set(associations.values())
        for track_id in list(self.tracks.keys()):
            if track_id not in associated_tracks:
                self.tracks[track_id].predict()
                self.frames_missing[track_id] = self.frames_missing.get(track_id, 0) + 1
                
        # Clean up old tracks
        self._cleanup_tracks()
        
        return self._get_track_states()
        
    def _associate_detections(self, detections):
        """Associate detections with existing tracks using nearest neighbor"""
        associations = {}
        
        if len(detections) == 0 or len(self.tracks) == 0:
            return associations
            
        # Get predicted positions from tracks
        track_ids = list(self.tracks.keys())
        track_positions = []
        
        for track_id in track_ids:
            pos = self.tracks[track_id].get_position()
            if pos is not None:
                track_positions.append(pos)
            else:
                track_positions.append(np.array([np.inf, np.inf, np.inf]))
                
        track_positions = np.array(track_positions)
        
        # Simple nearest neighbor association
        used_tracks = set()
        
        for det_idx, detection in enumerate(detections):
            # Find nearest track
            distances = np.linalg.norm(track_positions - detection, axis=1)
            
            while True:
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                if min_dist > self.max_distance:
                    break  # No suitable track found
                    
                track_id = track_ids[min_idx]
                
                if track_id not in used_tracks:
                    associations[det_idx] = track_id
                    used_tracks.add(track_id)
                    break
                else:
                    distances[min_idx] = np.inf  # Mark as used
                    
        return associations
        
    def _predict_tracks(self):
        """Predict all tracks forward"""
        for track in self.tracks.values():
            track.predict()
            
    def _cleanup_tracks(self):
        """Remove tracks that haven't been updated recently"""
        tracks_to_remove = []
        
        for track_id, frames in self.frames_missing.items():
            if frames > self.max_frames_missing:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            del self.frames_missing[track_id]
            
    def _get_track_states(self):
        """Get current state of all tracks"""
        states = {}
        
        for track_id, track in self.tracks.items():
            state = track.estimate()
            if state is not None:
                states[track_id] = {
                    'position': state[:3],
                    'velocity': state[3:6],
                    'id': track_id
                }
                
        return states 