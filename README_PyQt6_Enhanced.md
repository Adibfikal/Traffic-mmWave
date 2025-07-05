# Enhanced Radar Tracking System with PyQt6

## Overview

This is an enhanced version of the IWR6843ISK radar visualization system, reimplemented with **PyQt6** instead of PySide6, and featuring advanced object tracking visualizations including **3D bounding boxes** and **tracking trails**.

## New Features

### 🔲 3D Bounding Boxes
- **Wireframe bounding boxes** around each tracked object
- **Color-coded** per track for easy identification
- **Adjustable size** based on object spread and user preference
- **Real-time updates** as objects move
- **Automatic cleanup** when tracks are lost

### 🛤️ Tracking Trails
- **Historical path visualization** showing object movement over time
- **Fading effect** - older positions fade out gradually
- **Configurable trail length** (10-200 points)
- **Unique colors** per track matching the bounding boxes
- **Memory efficient** with automatic cleanup

### 🎮 Enhanced UI Controls
- **New "Enhanced Viz" tab** with dedicated controls
- **Real-time toggling** of bounding boxes and trails
- **Size/length sliders** for customization
- **Color-coded status indicators**
- **Performance monitoring** and optimization

## Key Improvements Over Original

### PyQt6 Migration
- ✅ **Complete migration** from PySide6 to PyQt6
- ✅ **Signal/slot compatibility** maintained
- ✅ **Performance optimizations** included
- ✅ **Modern Qt6 features** utilized

### Enhanced Tracking Visualization
- 🔲 **3D wireframe bounding boxes** for spatial awareness
- 🛤️ **Trailing paths** for motion analysis
- 🎨 **Color coding** for easy track identification
- ⚡ **Real-time updates** with minimal performance impact

### Advanced UI Features
- 📱 **Tabbed interface** for organized controls
- 🎛️ **Granular control** over visualization features
- 📊 **Enhanced statistics** and performance monitoring
- 🎯 **User-friendly** configuration options

## Technical Implementation

### Core Classes

#### `BoundingBoxManager`
```python
class BoundingBoxManager:
    """Manages 3D bounding boxes for tracked objects"""
    
    def create_box_wireframe(self, center, size):
        """Create wireframe points for a 3D bounding box"""
        
    def update_bounding_box(self, track_id, center, size):
        """Update or create bounding box for a tracked object"""
```

#### `TrailManager`
```python
class TrailManager:
    """Manages tracking trails/history for objects"""
    
    def add_position(self, track_id, position):
        """Add a new position to the trail"""
        
    def _update_trail_visualization(self, track_id):
        """Update the visual trail with fading effect"""
```

### Enhanced Data Processing
- **Integrated tracking** with bounding box calculation
- **Trail history management** with configurable length
- **Performance optimization** with adaptive update rates
- **Memory management** to prevent leaks

## Installation & Usage

### Requirements
```bash
pip install PyQt6
pip install pyqtgraph
pip install opencv-python
pip install numpy
pip install scikit-learn  # for HDBSCAN tracking
```

### Running the Application
```bash
python radar_gui_pyqt6.py
```

### Using the Enhanced Features

1. **Enable Tracking**: Go to "Object Tracking" tab and click "Enable Tracking"

2. **Configure Bounding Boxes**: 
   - Switch to "Enhanced Viz" tab
   - Toggle "Show Bounding Boxes"
   - Adjust "Box Size Factor" as needed

3. **Configure Trails**:
   - Toggle "Show Tracking Trails" 
   - Adjust "Trail Length" (10-200 points)
   - Use "Clear All Trails" to reset

4. **Customize Visualization**:
   - Adjust point and track marker sizes
   - Monitor performance metrics
   - Use tabbed interface for organized control

## Performance Optimizations

### Adaptive Rendering
- **Dynamic FPS adjustment** based on point cloud density
- **Efficient memory usage** with circular buffers for trails
- **Selective updates** only when tracking data changes
- **Background processing** for non-critical operations

### Memory Management
- **Automatic cleanup** of expired tracks
- **Bounded trail lengths** prevent memory growth
- **Efficient data structures** for real-time performance
- **Garbage collection** optimization

## Visualization Features

### Color Coding System
- **Red**: Stationary objects
- **Green**: Constant velocity objects  
- **Blue**: Maneuvering objects
- **Unique colors**: Each track gets distinct coloring
- **Transparency**: Bounding boxes use alpha blending

### 3D Wireframe Bounding Boxes
- **12-edge wireframe** representation
- **Adaptive sizing** based on object detection spread
- **Real-time updates** following object movement
- **Efficient rendering** using OpenGL line plots

### Tracking Trails
- **Smooth path visualization** connecting historical positions
- **Gradient fading** from transparent to opaque
- **Configurable length** for different analysis needs
- **High-performance** updates using deque data structures

## Architecture Benefits

### Modular Design
- **Separate managers** for different visualization types
- **Clean separation** of concerns
- **Easy extension** for new features
- **Maintainable codebase**

### PyQt6 Advantages
- **Modern Qt6 features** and performance improvements
- **Better cross-platform** support
- **Enhanced OpenGL** integration
- **Future-proof** technology stack

### Real-time Performance
- **Threaded data processing** prevents UI blocking
- **Efficient rendering** with OpenGL acceleration
- **Smart update scheduling** based on system load
- **Responsive user interface** even under high data rates

## Usage Examples

### Basic Tracking with Bounding Boxes
```python
# Enable tracking
self.tracking_enabled = True

# Configure bounding box visualization
self.show_bounding_boxes = True
bbox_size_factor = 1.5  # 1.5x object spread

# Update visualization
self.update_enhanced_visualizations(tracking_results)
```

### Trail Configuration
```python
# Configure trails
self.show_trails = True
trail_length = 50  # 50 historical positions

# Add new position
self.trail_manager.add_position(track_id, position)
```

## Future Enhancements

### Potential Additions
- 📈 **Motion prediction** visualization
- 🎯 **Velocity vectors** display
- 📊 **Track statistics** overlay
- 🔍 **Object classification** indicators
- 📱 **Touch/gesture** controls for mobile platforms

### Performance Improvements
- 🚀 **GPU acceleration** for large point clouds
- 💾 **Disk caching** for long tracking sessions  
- 🌐 **Network streaming** capabilities
- 🔄 **Multi-threading** for heavy computations

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure PyQt6 is properly installed
2. **Performance Issues**: Reduce trail length or bounding box detail
3. **Memory Usage**: Enable automatic cleanup features
4. **OpenGL Issues**: Update graphics drivers

### Debug Features
- **Debug mode**: Enable for detailed point cloud logging
- **Performance metrics**: Monitor FPS and processing times
- **Status logging**: Comprehensive system state reporting

## Conclusion

This enhanced PyQt6 implementation provides a modern, feature-rich radar tracking visualization system with advanced 3D bounding boxes and tracking trails. The modular architecture ensures maintainability while delivering real-time performance for professional radar applications.

---

**Note**: This implementation requires a compatible radar system (IWR6843ISK) and properly configured hardware connections for full functionality. The enhanced visualization features work with both live data and recorded sessions.

## 🔧 **Recent Fixes - Tracking for Recorded Data**

### **Issue Fixed**: Tracking didn't work during playback of recorded data

**Root Cause**: The original implementation had different data paths for live vs recorded data, and the tracking system wasn't properly integrated with the playback functionality.

**Solution**: 
1. **Fixed Data Format Handling**: Updated `update_data()` to handle both live data format (`data['pointCloud']`) and recorded data format (`data['frameData']['pointCloud']`)
2. **Enhanced Playback Processing**: Modified `update_visualization()` to process tracking during playback by extracting point cloud data and calling the tracker
3. **Unified Tracking Path**: Ensured both live and recorded data follow the same tracking pipeline

### **CRITICAL FIX**: Data Dimension Compatibility

**Issue**: `TRACKING ERROR (playback): operands could not be broadcast together with shapes (7,7) (3,3) (7,7)`

**Root Cause**: Recorded data contains **7 dimensions** `[x, y, z, doppler, snr, range, noise]` but the Enhanced HDBSCAN tracker expects **4 dimensions** `[x, y, z, doppler]`. This caused matrix operation failures in the tracking algorithm.

**Solution**: 
- **Live Data**: Preprocessed to use only first 4 dimensions before tracking
- **Recorded Data**: Preprocessed to use only first 4 dimensions before tracking  
- **Visualization**: Uses full dimensional data for display
- **Debug Messages**: Added to confirm dimension reduction is working

**Key Changes**:
```python
# Before (caused shape mismatch)
self.point_cloud_data = np.array(frame_data["pointCloud"])  # 7 dimensions

# After (tracking compatible)
raw_points = np.array(frame_data["pointCloud"])  # 7 dimensions
self.point_cloud_data = raw_points[:, :4]  # Only x, y, z, doppler for tracking
self.point_cloud_data_full = raw_points  # Full data for visualization
```

**Key Changes**:
- `update_visualization()` now processes tracking for playback data
- `update_data()` handles both data formats seamlessly  
- Tracking statistics show "(Playback)" status during recorded data playback
- All enhanced visualizations (bounding boxes, trails) work with recorded data
- **Data preprocessing ensures 4D compatibility for tracking while preserving full data for visualization**

## 🎯 **Usage Instructions**

### **For Live Data Tracking**:
1. Connect radar and start sensor
2. Go to "Object Tracking" tab
3. Click "Enable Tracking"
4. Adjust parameters as needed
5. Objects will appear with colored markers and optional bounding boxes/trails

### **For Recorded Data Tracking**:
1. Load a recording using "Load Recording" button
2. Enable tracking in the "Object Tracking" tab
3. Click "Play" to start playback
4. **NEW**: Tracking will now work during playback with the same visualizations as live data

### **Enhanced Visualizations**:
- **Red markers**: Stationary objects
- **Green markers**: Constant velocity objects  
- **Blue markers**: Maneuvering objects
- **Bounding boxes**: 3D wireframe boxes around tracked objects
- **Trails**: Historical path visualization with fading effect

## 🐛 **Known Issues Fixed**

- ✅ **Tracking for recorded data now works properly**
- ✅ **Data format compatibility between live and recorded data**
- ✅ **PyQt6 signal/slot compatibility**
- ✅ **Proper playback functionality with full feature support**

## 📊 **Performance Notes**

- Tracking works at full framerate during playback
- Bounding boxes and trails are updated in real-time
- Performance optimization maintains smooth visualization
- Memory usage is managed efficiently for long recordings 