"""
Visualization managers for 3D tracking displays.
This module provides managers for bounding boxes and tracking trails.
"""

import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import GLLinePlotItem
    _has_pyqtgraph = True
except ImportError:
    # Fallback for when pyqtgraph is not available
    gl = None
    GLLinePlotItem = None
    _has_pyqtgraph = False


class BoundingBoxManager:
    """Manages 3D bounding boxes for tracked objects"""
    
    def __init__(self, gl_view_widget):
        if gl is None:
            raise ImportError("pyqtgraph is required for BoundingBoxManager")
            
        self.gl_view = gl_view_widget
        self.bounding_boxes = {}  # track_id -> GLLinePlotItem
        self.box_colors = [
            [1, 0, 0, 0.8],  # Red
            [0, 1, 0, 0.8],  # Green  
            [0, 0, 1, 0.8],  # Blue
            [1, 1, 0, 0.8],  # Yellow
            [1, 0, 1, 0.8],  # Magenta
            [0, 1, 1, 0.8],  # Cyan
            [1, 0.5, 0, 0.8],  # Orange
            [0.5, 0, 1, 0.8],  # Purple
        ]
        
    def create_box_wireframe(self, center: List[float], size: List[float]) -> np.ndarray:
        """Create wireframe points for a 3D bounding box
        
        Args:
            center: [x, y, z] center position
            size: [sx, sy, sz] dimensions
            
        Returns:
            Array of wireframe line segments
        """
        x, y, z = center
        sx, sy, sz = size
        
        # Define the 8 corners of the box
        corners = np.array([
            [x-sx/2, y-sy/2, z-sz/2],  # 0: bottom-front-left
            [x+sx/2, y-sy/2, z-sz/2],  # 1: bottom-front-right  
            [x+sx/2, y+sy/2, z-sz/2],  # 2: bottom-back-right
            [x-sx/2, y+sy/2, z-sz/2],  # 3: bottom-back-left
            [x-sx/2, y-sy/2, z+sz/2],  # 4: top-front-left
            [x+sx/2, y-sy/2, z+sz/2],  # 5: top-front-right
            [x+sx/2, y+sy/2, z+sz/2],  # 6: top-back-right
            [x-sx/2, y+sy/2, z+sz/2],  # 7: top-back-left
        ])
        
        # Define the 12 edges of the box
        edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Create line segments for wireframe
        lines = []
        for edge in edges:
            lines.extend([corners[edge[0]], corners[edge[1]], [np.nan, np.nan, np.nan]])
        
        return np.array(lines[:-1])  # Remove last NaN
        
    def update_bounding_box(self, track_id: int, center: List[float], size: List[float]):
        """Update or create bounding box for a tracked object
        
        Args:
            track_id: Unique identifier for the track
            center: [x, y, z] center position
            size: [sx, sy, sz] box dimensions
        """
        wireframe = self.create_box_wireframe(center, size)
        color = self.box_colors[track_id % len(self.box_colors)]
        
        if track_id in self.bounding_boxes:
            # Update existing box
            self.bounding_boxes[track_id].setData(pos=wireframe, color=color, width=2)
        else:
            # Create new box
            if _has_pyqtgraph:
                box_item = gl.GLLinePlotItem(pos=wireframe, color=color, width=2, antialias=True)
                self.bounding_boxes[track_id] = box_item
                self.gl_view.addItem(box_item)
            
    def remove_bounding_box(self, track_id: int):
        """Remove bounding box for a track
        
        Args:
            track_id: Unique identifier for the track
        """
        if track_id in self.bounding_boxes:
            self.gl_view.removeItem(self.bounding_boxes[track_id])
            del self.bounding_boxes[track_id]
            
    def clear_all_boxes(self):
        """Clear all bounding boxes"""
        for track_id in list(self.bounding_boxes.keys()):
            self.remove_bounding_box(track_id)
            
    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs
        
        Returns:
            List of track IDs that have active bounding boxes
        """
        return list(self.bounding_boxes.keys())


class TrailManager:
    """Manages tracking trails/history for objects"""
    
    def __init__(self, gl_view_widget, max_trail_length: int = 50):
        if gl is None:
            raise ImportError("pyqtgraph is required for TrailManager")
            
        self.gl_view = gl_view_widget
        self.max_trail_length = max_trail_length
        self.trails = {}  # track_id -> deque of positions
        self.trail_items = {}  # track_id -> GLLinePlotItem
        self.trail_colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue  
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [1, 0.5, 0],  # Orange
            [0.5, 0, 1],  # Purple
        ]
        
    def add_position(self, track_id: int, position: np.ndarray):
        """Add a new position to the trail
        
        Args:
            track_id: Unique identifier for the track
            position: [x, y, z] position to add to trail
        """
        if track_id not in self.trails:
            self.trails[track_id] = deque(maxlen=self.max_trail_length)
            
        self.trails[track_id].append(position.copy())
        self._update_trail_visualization(track_id)
        
    def _update_trail_visualization(self, track_id: int):
        """Update the visual trail for a track
        
        Args:
            track_id: Unique identifier for the track
        """
        if track_id not in self.trails or len(self.trails[track_id]) < 2:
            return
            
        positions = np.array(list(self.trails[track_id]))
        base_color = self.trail_colors[track_id % len(self.trail_colors)]
        
        # Create fading effect by varying alpha along the trail
        trail_length = len(positions)
        colors = []
        for i in range(trail_length):
            alpha = (i + 1) / trail_length  # Fade from 0 to 1
            color = base_color + [alpha]
            colors.append(color)
        colors = np.array(colors)
        
        if track_id in self.trail_items:
            # Update existing trail
            self.trail_items[track_id].setData(pos=positions, color=colors, width=3)
        else:
            # Create new trail
            if _has_pyqtgraph:
                trail_item = gl.GLLinePlotItem(pos=positions, color=colors, width=3, antialias=True)
                self.trail_items[track_id] = trail_item
                self.gl_view.addItem(trail_item)
            
    def remove_trail(self, track_id: int):
        """Remove trail for a track
        
        Args:
            track_id: Unique identifier for the track
        """
        if track_id in self.trails:
            del self.trails[track_id]
        if track_id in self.trail_items:
            self.gl_view.removeItem(self.trail_items[track_id])
            del self.trail_items[track_id]
            
    def clear_all_trails(self):
        """Clear all trails"""
        for track_id in list(self.trails.keys()):
            self.remove_trail(track_id)
            
    def set_max_trail_length(self, max_length: int):
        """Set maximum trail length for all tracks
        
        Args:
            max_length: Maximum number of positions to keep in trail
        """
        self.max_trail_length = max_length
        
        # Update existing trails
        for track_id in self.trails:
            # Create new deque with updated max length
            old_trail = list(self.trails[track_id])
            self.trails[track_id] = deque(old_trail[-max_length:], maxlen=max_length)
            self._update_trail_visualization(track_id)
            
    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs
        
        Returns:
            List of track IDs that have active trails
        """
        return list(self.trails.keys())
        
    def get_trail_length(self, track_id: int) -> int:
        """Get the current length of a specific trail
        
        Args:
            track_id: Unique identifier for the track
            
        Returns:
            Number of positions in the trail, or 0 if track doesn't exist
        """
        if track_id in self.trails:
            return len(self.trails[track_id])
        return 0


class VisualizationCoordinator:
    """Coordinates multiple visualization managers"""
    
    def __init__(self, gl_view_widget, max_trail_length: int = 50):
        self.bounding_box_manager = BoundingBoxManager(gl_view_widget)
        self.trail_manager = TrailManager(gl_view_widget, max_trail_length)
        self.show_bounding_boxes = True
        self.show_trails = True
        
    def update_track_visualization(self, track_id: int, position: List[float], 
                                 bbox_size: Optional[List[float]] = None):
        """Update visualization for a single track
        
        Args:
            track_id: Unique identifier for the track
            position: [x, y, z] current position
            bbox_size: [sx, sy, sz] bounding box size (optional)
        """
        # Update trail if enabled
        if self.show_trails:
            self.trail_manager.add_position(track_id, np.array(position))
            
        # Update bounding box if enabled and size provided
        if self.show_bounding_boxes and bbox_size is not None:
            self.bounding_box_manager.update_bounding_box(track_id, position, bbox_size)
            
    def remove_track_visualization(self, track_id: int):
        """Remove all visualization for a track
        
        Args:
            track_id: Unique identifier for the track
        """
        self.bounding_box_manager.remove_bounding_box(track_id)
        self.trail_manager.remove_trail(track_id)
        
    def clear_all_visualizations(self):
        """Clear all visualizations"""
        self.bounding_box_manager.clear_all_boxes()
        self.trail_manager.clear_all_trails()
        
    def set_visualization_options(self, show_bounding_boxes: bool = None, 
                                show_trails: bool = None, max_trail_length: int = None):
        """Configure visualization options
        
        Args:
            show_bounding_boxes: Whether to show bounding boxes
            show_trails: Whether to show trails
            max_trail_length: Maximum trail length
        """
        if show_bounding_boxes is not None:
            self.show_bounding_boxes = show_bounding_boxes
            if not show_bounding_boxes:
                self.bounding_box_manager.clear_all_boxes()
                
        if show_trails is not None:
            self.show_trails = show_trails
            if not show_trails:
                self.trail_manager.clear_all_trails()
                
        if max_trail_length is not None:
            self.trail_manager.set_max_trail_length(max_trail_length)
            
    def get_active_tracks(self) -> List[int]:
        """Get list of all active track IDs
        
        Returns:
            List of track IDs that have any active visualization
        """
        bbox_tracks = set(self.bounding_box_manager.get_active_tracks())
        trail_tracks = set(self.trail_manager.get_active_tracks())
        return list(bbox_tracks.union(trail_tracks)) 