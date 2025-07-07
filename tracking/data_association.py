"""
Robust data association for radar tracking.
This module provides data association using Mahalanobis distance and track confidence.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment
from collections import deque
from typing import List, Tuple, Dict, Any

# Import the enhanced tracking module that defines EnhancedTrack and MotionState
from enhanced_tracking import EnhancedTrack, MotionState


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
    
    def update_parameters(self, **kwargs):
        """Update association parameters"""
        if 'max_tracking_distance' in kwargs:
            self.max_association_distance = kwargs['max_tracking_distance']
        if 'max_association_distance' in kwargs:
            self.max_association_distance = kwargs['max_association_distance']
        if 'mahalanobis_threshold' in kwargs:
            self.mahalanobis_threshold = kwargs['mahalanobis_threshold']
        if 'confidence_weight' in kwargs:
            self.confidence_weight = kwargs['confidence_weight']
        if 'motion_state_bonus' in kwargs:
            self.motion_state_bonus = kwargs['motion_state_bonus']
        if 'velocity_gate_factor' in kwargs:
            self.velocity_gate_factor = kwargs['velocity_gate_factor'] 