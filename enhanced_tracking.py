"""
Enhanced Tracking System with Interactive Multiple Model (IMM)
Supports objects that stop, start, and move again - like TI Traffic Monitoring systems
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.cluster import HDBSCAN
from collections import defaultdict, deque
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

class MotionState(Enum):
    """Motion states for the IMM filter"""
    STATIONARY = "stationary"
    CONSTANT_VELOCITY = "constant_velocity"
    MANEUVERING = "maneuvering"

@dataclass
class TrackState:
    """Enhanced track state with motion information"""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    covariance: np.ndarray  # State covariance
    motion_state: MotionState
    confidence: float
    time_stationary: float
    time_moving: float
    last_update: float

class MotionModel:
    """Base class for motion models"""
    def __init__(self, dt: float, process_noise: float):
        self.dt = dt
        self.process_noise = process_noise
        self.state_dim = 9  # [x, y, z, vx, vy, vz, ax, ay, az]
        
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition matrix F"""
        dt = self.dt
        F = np.eye(9)
        
        # Position update from velocity
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Velocity update from acceleration
        F[3, 6] = dt  # vx += ax * dt
        F[4, 7] = dt  # vy += ay * dt
        F[5, 8] = dt  # vz += az * dt
        
        return F
    
    def get_process_noise_matrix(self) -> np.ndarray:
        """Get process noise matrix Q"""
        dt = self.dt
        q = self.process_noise
        
        # Simplified process noise (can be made more sophisticated)
        Q = np.zeros((9, 9))
        
        # Position noise
        Q[0:3, 0:3] = q * dt**4 / 4 * np.eye(3)
        
        # Velocity noise
        Q[3:6, 3:6] = q * dt**2 * np.eye(3)
        
        # Acceleration noise
        Q[6:9, 6:9] = q * np.eye(3)
        
        # Cross-correlations
        Q[0:3, 3:6] = q * dt**3 / 2 * np.eye(3)
        Q[3:6, 0:3] = q * dt**3 / 2 * np.eye(3)
        Q[0:3, 6:9] = q * dt**2 / 2 * np.eye(3)
        Q[6:9, 0:3] = q * dt**2 / 2 * np.eye(3)
        Q[3:6, 6:9] = q * dt * np.eye(3)
        Q[6:9, 3:6] = q * dt * np.eye(3)
        
        return Q

class StationaryModel(MotionModel):
    """Motion model for stationary objects"""
    def __init__(self, dt: float):
        super().__init__(dt, process_noise=0.01)  # Very low process noise
        
    def get_transition_matrix(self) -> np.ndarray:
        F = super().get_transition_matrix()
        # Zero out velocity and acceleration transitions for stationary model
        F[0:3, 3:6] = 0  # No position change from velocity
        F[3:6, 6:9] = 0  # No velocity change from acceleration
        F[6:9, 6:9] = 0  # Zero acceleration
        return F

class ConstantVelocityModel(MotionModel):
    """Motion model for constant velocity movement"""
    def __init__(self, dt: float):
        super().__init__(dt, process_noise=1.0)
        
    def get_transition_matrix(self) -> np.ndarray:
        F = super().get_transition_matrix()
        # Zero out acceleration effects
        F[3:6, 6:9] = 0  # No velocity change from acceleration
        F[6:9, 6:9] = 0  # Zero acceleration
        return F

class ManeuveringModel(MotionModel):
    """Motion model for maneuvering objects"""
    def __init__(self, dt: float):
        super().__init__(dt, process_noise=5.0)  # Higher process noise for maneuvers

class IMM_Filter:
    """Interactive Multiple Model filter for robust tracking"""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.models = {
            MotionState.STATIONARY: StationaryModel(dt),
            MotionState.CONSTANT_VELOCITY: ConstantVelocityModel(dt),
            MotionState.MANEUVERING: ManeuveringModel(dt)
        }
        
        # Transition probability matrix (how likely to switch between models)
        self.transition_probs = np.array([
            [0.95, 0.04, 0.01],  # From stationary
            [0.05, 0.90, 0.05],  # From constant velocity
            [0.01, 0.05, 0.94]   # From maneuvering
        ])
        
        # Model probabilities
        self.model_probs = np.array([0.33, 0.33, 0.34])
        
        # Individual filters for each model
        self.filters = {}
        self.mixed_states = {}
        self.mixed_covariances = {}
        
    def initialize_track(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        """Initialize tracking with all models"""
        for state in MotionState:
            self.filters[state] = {
                'state': initial_state.copy(),
                'covariance': initial_covariance.copy(),
                'likelihood': 1.0
            }
    
    def mixing_step(self):
        """IMM Step 1: Mixing of estimates"""
        # Calculate mixing probabilities
        c_j = np.dot(self.transition_probs.T, self.model_probs)
        
        for j, state_j in enumerate(MotionState):
            # Mixing probabilities for model j
            mixing_probs = self.transition_probs[:, j] * self.model_probs / c_j[j]
            
            # Mixed initial state for model j
            mixed_state = np.zeros_like(self.filters[state_j]['state'])
            for i, state_i in enumerate(MotionState):
                mixed_state += mixing_probs[i] * self.filters[state_i]['state']
            
            # Mixed initial covariance for model j
            mixed_cov = np.zeros_like(self.filters[state_j]['covariance'])
            for i, state_i in enumerate(MotionState):
                diff = self.filters[state_i]['state'] - mixed_state
                mixed_cov += mixing_probs[i] * (
                    self.filters[state_i]['covariance'] + np.outer(diff, diff)
                )
            
            self.mixed_states[state_j] = mixed_state
            self.mixed_covariances[state_j] = mixed_cov
    
    def model_filtering_step(self, measurement: np.ndarray, measurement_noise: np.ndarray):
        """IMM Step 2: Model-matched filtering"""
        H = np.zeros((3, 9))  # Measurement matrix (observing position only)
        H[0:3, 0:3] = np.eye(3)
        
        for state in MotionState:
            model = self.models[state]
            
            # Prediction step
            F = model.get_transition_matrix()
            Q = model.get_process_noise_matrix()
            
            predicted_state = F @ self.mixed_states[state]
            predicted_cov = F @ self.mixed_covariances[state] @ F.T + Q
            
            # Update step
            innovation = measurement - H @ predicted_state
            innovation_cov = H @ predicted_cov @ H.T + measurement_noise
            
            # Calculate Kalman gain
            try:
                kalman_gain = predicted_cov @ H.T @ la.inv(innovation_cov)
            except la.LinAlgError:
                kalman_gain = np.zeros((9, 3))
                innovation_cov = np.eye(3) * 1000  # Large uncertainty
            
            # Update state and covariance
            updated_state = predicted_state + kalman_gain @ innovation
            updated_cov = (np.eye(9) - kalman_gain @ H) @ predicted_cov
            
            # Calculate likelihood
            try:
                likelihood = self._multivariate_gaussian_pdf(innovation, np.zeros(3), innovation_cov)
            except:
                likelihood = 1e-10
            
            self.filters[state] = {
                'state': updated_state,
                'covariance': updated_cov,
                'likelihood': likelihood
            }
    
    def model_probability_update(self):
        """IMM Step 3: Model probability update"""
        # Calculate normalization factor
        c_j = np.dot(self.transition_probs.T, self.model_probs)
        
        # Update model probabilities
        likelihoods = np.array([self.filters[state]['likelihood'] for state in MotionState])
        
        # Avoid numerical issues
        likelihoods = np.maximum(likelihoods, 1e-10)
        
        self.model_probs = likelihoods * c_j
        total_prob = np.sum(self.model_probs)
        
        if total_prob > 0:
            self.model_probs /= total_prob
        else:
            self.model_probs = np.array([0.33, 0.33, 0.34])
    
    def estimate_and_covariance(self) -> Tuple[np.ndarray, np.ndarray, MotionState]:
        """IMM Step 4: Estimate and covariance computation"""
        # Combined estimate
        combined_state = np.zeros(9)
        for i, state in enumerate(MotionState):
            combined_state += self.model_probs[i] * self.filters[state]['state']
        
        # Combined covariance
        combined_cov = np.zeros((9, 9))
        for i, state in enumerate(MotionState):
            diff = self.filters[state]['state'] - combined_state
            combined_cov += self.model_probs[i] * (
                self.filters[state]['covariance'] + np.outer(diff, diff)
            )
        
        # Determine most likely motion state
        best_state_idx = np.argmax(self.model_probs)
        best_state = list(MotionState)[best_state_idx]
        
        return combined_state, combined_cov, best_state
    
    def predict_only(self) -> Tuple[np.ndarray, np.ndarray, MotionState]:
        """Prediction step without measurement update (for coasted tracks)"""
        self.mixing_step()
        
        for state in MotionState:
            model = self.models[state]
            F = model.get_transition_matrix()
            Q = model.get_process_noise_matrix()
            
            predicted_state = F @ self.mixed_states[state]
            predicted_cov = F @ self.mixed_covariances[state] @ F.T + Q
            
            self.filters[state] = {
                'state': predicted_state,
                'covariance': predicted_cov,
                'likelihood': self.filters[state]['likelihood'] * 0.9  # Decay likelihood
            }
        
        return self.estimate_and_covariance()
    
    def update(self, measurement: np.ndarray, measurement_noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray, MotionState]:
        """Complete IMM update cycle"""
        self.mixing_step()
        self.model_filtering_step(measurement, measurement_noise)
        self.model_probability_update()
        return self.estimate_and_covariance()
    
    def _multivariate_gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Calculate multivariate Gaussian PDF"""
        k = len(x)
        try:
            diff = x - mean
            inv_cov = la.inv(cov)
            exp_term = -0.5 * diff.T @ inv_cov @ diff
            norm_term = 1.0 / np.sqrt((2 * np.pi) ** k * la.det(cov))
            return norm_term * np.exp(exp_term)
        except:
            return 1e-10

class EnhancedTrack:
    """Enhanced track with IMM filter and robust state management"""
    
    def __init__(self, track_id: int, initial_detection: np.ndarray, timestamp: float):
        self.track_id = track_id
        self.created_time = timestamp
        self.last_update_time = timestamp
        self.age = 0
        self.hits = 1
        self.consecutive_misses = 0
        
        # Initialize IMM filter
        self.imm_filter = IMM_Filter()
        
        # Initial state [x, y, z, vx, vy, vz, ax, ay, az]
        initial_state = np.zeros(9)
        initial_state[0:3] = initial_detection
        
        # Initial covariance (high uncertainty)
        initial_covariance = np.eye(9)
        initial_covariance[0:3, 0:3] *= 100  # Position uncertainty
        initial_covariance[3:6, 3:6] *= 25   # Velocity uncertainty
        initial_covariance[6:9, 6:9] *= 10   # Acceleration uncertainty
        
        self.imm_filter.initialize_track(initial_state, initial_covariance)
        
        # Track history for motion analysis
        self.position_history = deque(maxlen=50)
        self.velocity_history = deque(maxlen=30)
        self.motion_state_history = deque(maxlen=20)
        
        # Motion state timing
        self.time_stationary = 0.0
        self.time_moving = 0.0
        self.stationary_threshold = 0.5  # m/s
        
        # Track confidence and quality metrics
        self.confidence = 1.0
        self.mahalanobis_distances = deque(maxlen=10)
        
        # Current state
        self.current_state, self.current_covariance, self.current_motion_state = \
            self.imm_filter.estimate_and_covariance()
        
        self.position_history.append(self.position)
        self.motion_state_history.append(self.current_motion_state)
    
    @property
    def position(self) -> np.ndarray:
        """Current position [x, y, z]"""
        return self.current_state[0:3]
    
    @property
    def velocity(self) -> np.ndarray:
        """Current velocity [vx, vy, vz]"""
        return self.current_state[3:6]
    
    @property
    def speed(self) -> float:
        """Current speed magnitude"""
        return np.linalg.norm(self.velocity)
    
    @property
    def acceleration(self) -> np.ndarray:
        """Current acceleration [ax, ay, az]"""
        return self.current_state[6:9]
    
    def predict(self, timestamp: float):
        """Predict track state to given timestamp"""
        dt = timestamp - self.last_update_time
        
        # Update IMM filter dt if needed
        if abs(dt - self.imm_filter.dt) > 0.01:
            self.imm_filter.dt = dt
            # Update all model dt values
            for model in self.imm_filter.models.values():
                model.dt = dt
        
        self.current_state, self.current_covariance, self.current_motion_state = \
            self.imm_filter.predict_only()
        
        self.age += 1
        self.consecutive_misses += 1
        
        # Update motion timing
        self._update_motion_timing(dt)
    
    def update(self, measurement: np.ndarray, timestamp: float, measurement_noise: Optional[np.ndarray] = None):
        """Update track with new measurement"""
        if measurement_noise is None:
            measurement_noise = np.eye(3) * 1.0  # Default measurement noise
        
        dt = timestamp - self.last_update_time
        
        # Update IMM filter dt if needed
        if abs(dt - self.imm_filter.dt) > 0.01:
            self.imm_filter.dt = dt
            for model in self.imm_filter.models.values():
                model.dt = dt
        
        # Update with measurement
        self.current_state, self.current_covariance, self.current_motion_state = \
            self.imm_filter.update(measurement, measurement_noise)
        
        # Update track statistics
        self.last_update_time = timestamp
        self.hits += 1
        self.consecutive_misses = 0
        self.age += 1
        
        # Update histories
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.motion_state_history.append(self.current_motion_state)
        
        # Update motion timing
        self._update_motion_timing(dt)
        
        # Calculate Mahalanobis distance for quality assessment
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        predicted_measurement = H @ self.current_state
        innovation = measurement - predicted_measurement
        innovation_cov = H @ self.current_covariance @ H.T + measurement_noise
        
        try:
            mahal_dist = np.sqrt(innovation.T @ la.inv(innovation_cov) @ innovation)
            self.mahalanobis_distances.append(mahal_dist)
        except:
            mahal_dist = 100.0
            self.mahalanobis_distances.append(mahal_dist)
        
        # Update confidence based on track quality
        self._update_confidence()
    
    def _update_motion_timing(self, dt: float):
        """Update time spent in different motion states"""
        if self.speed < self.stationary_threshold:
            self.time_stationary += dt
            self.time_moving = max(0, self.time_moving - dt * 0.1)  # Gradual decay
        else:
            self.time_moving += dt
            self.time_stationary = max(0, self.time_stationary - dt * 0.1)  # Gradual decay
    
    def _update_confidence(self):
        """Update track confidence based on various factors"""
        # Base confidence on hit rate
        hit_rate = self.hits / max(self.age, 1)
        confidence = hit_rate
        
        # Penalize based on recent Mahalanobis distances
        if len(self.mahalanobis_distances) > 0:
            avg_mahal = np.mean(list(self.mahalanobis_distances))
            # Confidence decreases with large innovations
            mahal_confidence = np.exp(-avg_mahal / 10.0)
            confidence *= mahal_confidence
        
        # Bonus for consistent motion state
        if len(self.motion_state_history) >= 5:
            recent_states = list(self.motion_state_history)[-5:]
            consistency = recent_states.count(self.current_motion_state) / len(recent_states)
            confidence *= (0.5 + 0.5 * consistency)
        
        # Penalize for too many consecutive misses
        miss_penalty = np.exp(-self.consecutive_misses / 5.0)
        confidence *= miss_penalty
        
        self.confidence = np.clip(confidence, 0.0, 1.0)
    
    def should_delete(self, max_age: int = 30, max_misses: int = 10, min_confidence: float = 0.1) -> bool:
        """Determine if track should be deleted"""
        if self.age > max_age:
            return True
        if self.consecutive_misses > max_misses:
            return True
        if self.confidence < min_confidence and self.age > 5:
            return True
        return False
    
    def get_predicted_position(self, prediction_time: float) -> np.ndarray:
        """Get predicted position at future time"""
        dt = prediction_time - self.last_update_time
        
        # Simple prediction using current velocity
        predicted_pos = self.position + self.velocity * dt
        
        # For stationary objects, don't predict movement
        if self.current_motion_state == MotionState.STATIONARY:
            return self.position
        
        return predicted_pos
    
    def get_track_info(self) -> Dict[str, Any]:
        """Get comprehensive track information"""
        return {
            'id': self.track_id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'speed': self.speed,
            'acceleration': self.acceleration.tolist(),
            'motion_state': self.current_motion_state.value,
            'confidence': self.confidence,
            'age': self.age,
            'hits': self.hits,
            'consecutive_misses': self.consecutive_misses,
            'time_stationary': self.time_stationary,
            'time_moving': self.time_moving,
            'model_probabilities': self.imm_filter.model_probs.tolist(),
            'covariance_trace': np.trace(self.current_covariance[0:3, 0:3])
        }

if __name__ == "__main__":
    # Simple test of the enhanced tracking system
    print("Enhanced IMM-based tracking system initialized")
    print("Key features:")
    print("- Interactive Multiple Model (IMM) filtering")
    print("- Multiple motion states: stationary, constant velocity, maneuvering")
    print("- Robust data association with Mahalanobis distance")
    print("- Track confidence and quality assessment")
    print("- Handles objects that stop and start moving") 