"""
Enhanced Visualization System for Radar Tracking
Provides distinct visualization for tracked objects with centroids, motion states, and confidence indicators
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque
import pyqtgraph.opengl as gl
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from enhanced_tracking import EnhancedTrack, MotionState

class TrackVisualizer:
    """Enhanced visualization system for tracked objects"""
    
    def __init__(self, gl_widget):
        self.gl_widget = gl_widget
        
        # Track visualization objects
        self.track_scatter_plots = {}  # track_id -> GLScatterPlotItem
        self.track_centroids = {}      # track_id -> GLScatterPlotItem (larger markers)
        self.track_trails = {}         # track_id -> GLLinePlotItem
        self.track_velocity_arrows = {} # track_id -> GLLinePlotItem
        self.track_confidence_rings = {} # track_id -> GLMeshItem
        
        # Detection visualization
        self.detection_scatter = None
        self.cluster_scatter = None
        
        # Color schemes
        self.motion_state_colors = {
            MotionState.STATIONARY: (1.0, 0.2, 0.2, 1.0),      # Red
            MotionState.CONSTANT_VELOCITY: (0.2, 1.0, 0.2, 1.0), # Green
            MotionState.MANEUVERING: (0.2, 0.2, 1.0, 1.0)       # Blue
        }
        
        self.confidence_colors = {
            'high': (0.0, 1.0, 0.0, 1.0),     # Green
            'medium': (1.0, 1.0, 0.0, 1.0),   # Yellow
            'low': (1.0, 0.0, 0.0, 1.0)       # Red
        }
        
        # Visualization parameters
        self.centroid_size = 15.0          # Large markers for centroids
        self.track_point_size = 8.0        # Smaller markers for track points
        self.detection_size = 4.0          # Small markers for detections
        self.trail_length = 20             # Number of points in trail
        self.arrow_scale = 2.0             # Scale factor for velocity arrows
        self.confidence_ring_segments = 20 # Number of segments in confidence rings
        
        # Trail history
        self.track_position_history = {}   # track_id -> deque of positions
        
        # Animation parameters
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animations)
        self.animation_timer.start(100)  # Update every 100ms
        
    def clear_all_visualizations(self):
        """Clear all existing visualizations"""
        # Clear track visualizations
        for track_id in list(self.track_scatter_plots.keys()):
            self.remove_track_visualization(track_id)
        
        # Clear detection visualizations
        if self.detection_scatter:
            self.gl_widget.removeItem(self.detection_scatter)
            self.detection_scatter = None
        
        if self.cluster_scatter:
            self.gl_widget.removeItem(self.cluster_scatter)
            self.cluster_scatter = None
    
    def remove_track_visualization(self, track_id: int):
        """Remove all visualization elements for a specific track"""
        if track_id in self.track_scatter_plots:
            self.gl_widget.removeItem(self.track_scatter_plots[track_id])
            del self.track_scatter_plots[track_id]
        
        if track_id in self.track_centroids:
            self.gl_widget.removeItem(self.track_centroids[track_id])
            del self.track_centroids[track_id]
        
        if track_id in self.track_trails:
            self.gl_widget.removeItem(self.track_trails[track_id])
            del self.track_trails[track_id]
        
        if track_id in self.track_velocity_arrows:
            self.gl_widget.removeItem(self.track_velocity_arrows[track_id])
            del self.track_velocity_arrows[track_id]
        
        if track_id in self.track_confidence_rings:
            self.gl_widget.removeItem(self.track_confidence_rings[track_id])
            del self.track_confidence_rings[track_id]
        
        if track_id in self.track_position_history:
            del self.track_position_history[track_id]
    
    def get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= 0.7:
            return 'high'
        elif confidence >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def create_confidence_ring(self, position: np.ndarray, confidence: float, motion_state: MotionState) -> gl.GLMeshItem:
        """Create a confidence ring around the object"""
        # Ring radius based on confidence (larger ring = less confident)
        radius = 0.5 + (1.0 - confidence) * 2.0
        
        # Create ring vertices
        angles = np.linspace(0, 2*np.pi, self.confidence_ring_segments + 1)
        vertices = np.zeros((self.confidence_ring_segments + 1, 3))
        
        for i, angle in enumerate(angles):
            vertices[i] = [
                position[0] + radius * np.cos(angle),
                position[1] + radius * np.sin(angle),
                position[2]
            ]
        
        # Create triangular faces for the ring
        faces = []
        for i in range(self.confidence_ring_segments):
            # Create thin triangular segments
            center_idx = len(vertices)
            vertices = np.vstack([vertices, position])  # Add center point
            
            faces.append([center_idx, i, i + 1])
        
        # Create mesh
        mesh = gl.GLMeshItem(
            vertexes=vertices,
            faces=np.array(faces),
            color=self.motion_state_colors[motion_state],
            smooth=True,
            computeNormals=True,
            shader='balloon'
        )
        
        return mesh
    
    def create_velocity_arrow(self, position: np.ndarray, velocity: np.ndarray) -> Optional[gl.GLLinePlotItem]:
        """Create a velocity arrow visualization"""
        if np.linalg.norm(velocity) < 0.1:
            return None  # Don't show arrow for very slow movement
        
        # Arrow start and end points
        arrow_start = position
        arrow_end = position + velocity * self.arrow_scale
        
        # Create arrow shaft
        points = np.array([arrow_start, arrow_end])
        
        # Create arrow line
        arrow_line = gl.GLLinePlotItem(
            pos=points,
            color=(1.0, 1.0, 0.0, 1.0),  # Yellow
            width=3.0,
            antialias=True
        )
        
        return arrow_line
    
    def update_track_visualization(self, track: Union[EnhancedTrack, 'TrackVisualizationProxy']):
        """Update visualization for a single track"""
        track_id = track.track_id
        position = track.position
        velocity = track.velocity
        motion_state = track.current_motion_state
        confidence = track.confidence
        
        # Update position history
        if track_id not in self.track_position_history:
            from collections import deque
            self.track_position_history[track_id] = deque(maxlen=self.trail_length)
        
        self.track_position_history[track_id].append(position.copy())
        
        # Get colors based on motion state and confidence
        motion_color = self.motion_state_colors[motion_state]
        confidence_level = self.get_confidence_level(confidence)
        
        # Update or create centroid marker (large, distinct marker)
        if track_id not in self.track_centroids:
            self.track_centroids[track_id] = gl.GLScatterPlotItem()
            self.gl_widget.addItem(self.track_centroids[track_id])
        
        # Create centroid with size based on confidence
        centroid_size = self.centroid_size * (0.5 + 0.5 * confidence)
        self.track_centroids[track_id].setData(
            pos=position.reshape(1, 3),
            color=np.array([motion_color]),
            size=centroid_size,
            pxMode=False
        )
        
        # Update or create track trail
        if len(self.track_position_history[track_id]) > 1:
            if track_id not in self.track_trails:
                self.track_trails[track_id] = gl.GLLinePlotItem()
                self.gl_widget.addItem(self.track_trails[track_id])
            
            # Create trail with fading effect
            trail_positions = np.array(list(self.track_position_history[track_id]))
            
            # Create colors that fade from current to transparent
            trail_colors = []
            for i in range(len(trail_positions)):
                alpha = (i + 1) / len(trail_positions) * 0.7  # Fade from 0 to 0.7
                color = (*motion_color[:3], alpha)
                trail_colors.append(color)
            
            self.track_trails[track_id].setData(
                pos=trail_positions,
                color=np.array(trail_colors),
                width=2.0,
                antialias=True
            )
        
        # Update or create velocity arrow
        if track_id in self.track_velocity_arrows:
            self.gl_widget.removeItem(self.track_velocity_arrows[track_id])
        
        arrow = self.create_velocity_arrow(position, velocity)
        if arrow:
            self.track_velocity_arrows[track_id] = arrow
            self.gl_widget.addItem(arrow)
        
        # Update or create confidence ring
        if track_id in self.track_confidence_rings:
            self.gl_widget.removeItem(self.track_confidence_rings[track_id])
        
        # Only show confidence ring for stationary objects or low confidence
        if motion_state == MotionState.STATIONARY or confidence < 0.5:
            confidence_ring = self.create_confidence_ring(position, confidence, motion_state)
            self.track_confidence_rings[track_id] = confidence_ring
            self.gl_widget.addItem(confidence_ring)
    
    def update_detection_visualization(self, detections: List[np.ndarray], 
                                     cluster_labels: Optional[np.ndarray] = None):
        """Update visualization for raw detections"""
        if not detections:
            if self.detection_scatter:
                self.gl_widget.removeItem(self.detection_scatter)
                self.detection_scatter = None
            return
        
        detection_array = np.array(detections)
        
        # Create or update detection scatter plot
        if self.detection_scatter:
            self.gl_widget.removeItem(self.detection_scatter)
        
        # Color detections based on cluster assignment
        if cluster_labels is not None:
            colors = []
            unique_labels = set(cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            color_map = [
                (0.8, 0.8, 0.8, 0.6),  # Gray for noise
                (1.0, 0.5, 0.5, 0.8),  # Light red
                (0.5, 1.0, 0.5, 0.8),  # Light green
                (0.5, 0.5, 1.0, 0.8),  # Light blue
                (1.0, 1.0, 0.5, 0.8),  # Light yellow
                (1.0, 0.5, 1.0, 0.8),  # Light magenta
                (0.5, 1.0, 1.0, 0.8),  # Light cyan
            ]
            
            for label in cluster_labels:
                if label == -1:
                    colors.append(color_map[0])  # Gray for noise
                else:
                    colors.append(color_map[(label % (len(color_map) - 1)) + 1])
            
            colors = np.array(colors)
        else:
            # Default white color for all detections
            colors = np.array([(1.0, 1.0, 1.0, 0.6)] * len(detections))
        
        self.detection_scatter = gl.GLScatterPlotItem(
            pos=detection_array,
            color=colors,
            size=self.detection_size,
            pxMode=False
        )
        self.gl_widget.addItem(self.detection_scatter)
    
    def update_complete_visualization(self, tracking_result: Dict[str, Any]):
        """Update all visualizations from tracking result"""
        # Clear old track visualizations for deleted tracks
        active_track_ids = set(track['id'] for track in tracking_result['tracks'])
        for track_id in list(self.track_scatter_plots.keys()):
            if track_id not in active_track_ids:
                self.remove_track_visualization(track_id)
        
        # Update track visualizations
        for track_info in tracking_result['tracks']:
            # Create a minimal track object for visualization
            track = self.create_track_from_info(track_info)
            self.update_track_visualization(track)
        
        # Update detection visualization
        detections = [np.array(det) for det in tracking_result['detections']]
        self.update_detection_visualization(detections)
    
    def create_track_from_info(self, track_info: Dict[str, Any]) -> 'TrackVisualizationProxy':
        """Create a proxy track object for visualization purposes"""
        return TrackVisualizationProxy(track_info)
    
    def update_animations(self):
        """Update animated elements (called by timer)"""
        # Update any animated elements here
        # For example, pulsing confidence rings or rotating velocity arrows
        pass
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization statistics"""
        return {
            'active_tracks': len(self.track_centroids),
            'tracks_with_trails': len(self.track_trails),
            'tracks_with_arrows': len(self.track_velocity_arrows),
            'tracks_with_confidence_rings': len(self.track_confidence_rings),
            'motion_state_colors': {state.value: color for state, color in self.motion_state_colors.items()},
            'visualization_parameters': {
                'centroid_size': self.centroid_size,
                'track_point_size': self.track_point_size,
                'detection_size': self.detection_size,
                'trail_length': self.trail_length,
                'arrow_scale': self.arrow_scale
            }
        }
    
    def set_visualization_parameters(self, **kwargs):
        """Update visualization parameters"""
        if 'centroid_size' in kwargs:
            self.centroid_size = kwargs['centroid_size']
        if 'track_point_size' in kwargs:
            self.track_point_size = kwargs['track_point_size']
        if 'detection_size' in kwargs:
            self.detection_size = kwargs['detection_size']
        if 'trail_length' in kwargs:
            self.trail_length = kwargs['trail_length']
            # Update existing trail histories
            for track_id in self.track_position_history:
                self.track_position_history[track_id] = deque(
                    list(self.track_position_history[track_id])[-self.trail_length:],
                    maxlen=self.trail_length
                )
        if 'arrow_scale' in kwargs:
            self.arrow_scale = kwargs['arrow_scale']

class TrackVisualizationProxy:
    """Proxy object for track visualization when we only have track info dict"""
    
    def __init__(self, track_info: Dict[str, Any]):
        self.track_id = track_info['id']
        self.position = np.array(track_info['position'])
        self.velocity = np.array(track_info['velocity'])
        self.confidence = track_info['confidence']
        
        # Convert motion state string back to enum
        motion_state_str = track_info['motion_state']
        self.current_motion_state = MotionState(motion_state_str)

class EnhancedTrackingVisualization:
    """Complete enhanced tracking visualization system"""
    
    def __init__(self, gl_widget):
        self.track_visualizer = TrackVisualizer(gl_widget)
        self.gl_widget = gl_widget
        
        # Visualization modes
        self.show_detections = True
        self.show_trails = True
        self.show_velocity_arrows = True
        self.show_confidence_rings = True
        self.show_centroids = True
        
        # Color coding options
        self.color_by_motion_state = True
        self.color_by_confidence = False
        self.color_by_track_id = False
        
    def update_from_tracking_result(self, tracking_result: Dict[str, Any]):
        """Update all visualizations from tracking result"""
        if not tracking_result:
            return
        
        # Update track visualizations
        self.track_visualizer.update_complete_visualization(tracking_result)
    
    def set_visualization_mode(self, **kwargs):
        """Set visualization mode flags"""
        if 'show_detections' in kwargs:
            self.show_detections = kwargs['show_detections']
        if 'show_trails' in kwargs:
            self.show_trails = kwargs['show_trails']
        if 'show_velocity_arrows' in kwargs:
            self.show_velocity_arrows = kwargs['show_velocity_arrows']
        if 'show_confidence_rings' in kwargs:
            self.show_confidence_rings = kwargs['show_confidence_rings']
        if 'show_centroids' in kwargs:
            self.show_centroids = kwargs['show_centroids']
    
    def set_color_mode(self, **kwargs):
        """Set color coding mode"""
        if 'color_by_motion_state' in kwargs:
            self.color_by_motion_state = kwargs['color_by_motion_state']
        if 'color_by_confidence' in kwargs:
            self.color_by_confidence = kwargs['color_by_confidence']
        if 'color_by_track_id' in kwargs:
            self.color_by_track_id = kwargs['color_by_track_id']
    
    def clear_all(self):
        """Clear all visualizations"""
        self.track_visualizer.clear_all_visualizations()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive visualization statistics"""
        return {
            'visualizer_stats': self.track_visualizer.get_visualization_statistics(),
            'modes': {
                'show_detections': self.show_detections,
                'show_trails': self.show_trails,
                'show_velocity_arrows': self.show_velocity_arrows,
                'show_confidence_rings': self.show_confidence_rings,
                'show_centroids': self.show_centroids
            },
            'color_modes': {
                'color_by_motion_state': self.color_by_motion_state,
                'color_by_confidence': self.color_by_confidence,
                'color_by_track_id': self.color_by_track_id
            }
        }

if __name__ == "__main__":
    print("Enhanced Tracking Visualization System")
    print("Key features:")
    print("- Large, distinct centroid markers for tracked objects")
    print("- Motion state color coding (Red=Stationary, Green=CV, Blue=Maneuvering)")
    print("- Confidence-based sizing and rings")
    print("- Velocity arrows for moving objects")
    print("- Fading trails showing object history")
    print("- Clustered detection visualization") 