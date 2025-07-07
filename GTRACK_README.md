# GTRACK Vehicle Tracking Implementation

A Texas Instruments GTRACK-style multi-object tracking algorithm implementation for mmWave radar data, specifically optimized for vehicle detection and tracking.

## Features

### Core Algorithm Components
- **Dynamic DBSCAN Clustering** - SNR-based adaptive clustering for different signal strengths
- **Unscented Kalman Filter** - Advanced state estimation with motion prediction
- **Probability Matrix Data Association** - Hungarian algorithm for track-to-detection assignment
- **Track Lifecycle Management** - Automatic track creation, confirmation, and deletion
- **Motion Pattern Analysis** - False target rejection based on realistic vehicle behavior
- **Real-time Performance** - Optimized for up to 25 simultaneous vehicle tracks

### Vehicle-Specific Optimizations
- **Motion Models**: Constant velocity, constant acceleration, and coordinated turn
- **Speed Limits**: Realistic vehicle speed constraints (max 60 m/s)
- **Acceleration Limits**: Reasonable vehicle acceleration bounds (max 10 m/s²)
- **Clustering Parameters**: Tuned for vehicle-sized targets at various SNR levels

## Installation & Dependencies

### Required Python Packages
```bash
pip install numpy scipy scikit-learn matplotlib PySide6 pyqtgraph opencv-python
```

### Core Files
- `gtrack_algorithm.py` - Main GTRACK implementation
- `recording_analyzer.py` - Recording analysis and visualization tools  
- `radar_gui.py` - GUI integration (modified for GTRACK)
- `test_gtrack.py` - Testing and validation script

## Quick Start

### 1. Test the Algorithm
```bash
# Quick test with synthetic data
python test_gtrack.py quick

# Test with a specific recording
python test_gtrack.py single recordings/your_recording_name

# Test all recordings
python test_gtrack.py all
```

### 2. Analyze Recordings
```python
from recording_analyzer import RecordingAnalyzer

# Create analyzer
analyzer = RecordingAnalyzer("recordings")

# List available recordings
recordings = analyzer.list_recordings()
print(f"Found {len(recordings)} recordings")

# Analyze a specific recording
analysis = analyzer.analyze_recording("session_20250704_171209")

# Analyze all recordings
all_analyses = analyzer.analyze_all_recordings()
```

### 3. GUI Integration
```bash
# Run the radar GUI with GTRACK integration
python radar_gui.py
```

In the GUI:
1. Enable "GTRACK Vehicle Tracking" checkbox
2. Adjust "Max Vehicles" (1-50, default: 25)
3. Tune "Clustering" sensitivity (1-10, default: 5)
4. Use "Reset All Tracks" to clear tracking state

## Algorithm Details

### Processing Pipeline

1. **Data Preprocessing**
   - Extract x,y coordinates from point cloud
   - Filter noise based on SNR thresholds
   - Handle both live and recorded data

2. **Dynamic DBSCAN Clustering**
   ```python
   # SNR-based clustering parameters
   dbscan_params = {
       'high_snr': {'eps': 1.5, 'min_samples': 3},    # Strong returns
       'medium_snr': {'eps': 2.0, 'min_samples': 2},  # Medium returns  
       'low_snr': {'eps': 3.0, 'min_samples': 4}      # Weak returns
   }
   ```

3. **Kalman Filter Prediction**
   - State vector: [x, y, vx, vy, ax, ay]
   - Constant acceleration motion model
   - Process noise adapted for vehicle dynamics

4. **Data Association**
   - Mahalanobis distance with uncertainty consideration
   - Hungarian algorithm for optimal assignment
   - Statistical gating (95% confidence)

5. **Track Management**
   - 3 detections needed for track confirmation
   - 5 coast cycles before track deletion
   - State machine: DETECTION → ACTIVE → COASTED → FREE

6. **Motion Pattern Analysis**
   - Reject tracks with excessive position jitter
   - Filter impossible vehicle speeds
   - Background noise reduction

### Performance Characteristics

- **Processing Speed**: ~5-15ms per frame (real-time capable)
- **Memory Usage**: Efficient with deque-based history management
- **Accuracy**: Balanced false positive/missed detection minimization
- **Scalability**: Optimized for up to 25 simultaneous tracks

## Configuration Parameters

### Key Parameters (in GTRACKProcessor)
```python
# Track management
max_tracks = 25                    # Maximum simultaneous tracks
confirmation_threshold = 3         # Detections needed to confirm
deletion_threshold = 5            # Coast cycles before deletion

# Vehicle motion limits
max_vehicle_speed = 60.0          # m/s (216 km/h)
max_vehicle_accel = 10.0          # m/s²

# Association gating
gate_probability = 0.95           # Statistical gate probability
gate_threshold = 9.21             # Chi-squared threshold

# SNR thresholds (0-255 range)
snr_thresholds = {
    'high': 200,    # Strong vehicle reflections
    'medium': 100,  # Medium strength
    'low': 50       # Minimum detectable
}
```

## Usage Examples

### Basic Processing
```python
from gtrack_algorithm import GTRACKProcessor

# Initialize processor
processor = GTRACKProcessor(max_tracks=25)

# Process a frame
frame_data = {
    'timestamp': 1000.0,  # milliseconds
    'frameData': {
        'error': 0,
        'frameNum': 1,
        'pointCloud': [[x, y, z, doppler, snr_low, snr_high, intensity], ...]
    }
}

result = processor.process_frame(frame_data)
print(f"Active tracks: {result['frame_stats']['active_tracks']}")
```

### Recording Analysis with Visualization
```python
from recording_analyzer import RecordingAnalyzer

analyzer = RecordingAnalyzer()

# Analyze and visualize
analysis = analyzer.analyze_recording("my_recording")
analyzer.visualize_tracking_results("my_recording", analysis, save_animation=True)
```

### Batch Processing
```python
from gtrack_algorithm import analyze_recording_folder

# Process all recordings
results = analyze_recording_folder("recordings")

for recording_name, analysis in results.items():
    stats = analysis['statistics']
    print(f"{recording_name}: {stats['track_statistics']['total_unique_tracks']} tracks")
```

## Output Format

### Frame Results
```python
{
    'timestamp': 1000.0,
    'frame_stats': {
        'total_detections': 15,
        'valid_detections': 12,
        'active_tracks': 3,
        'total_tracks': 5
    },
    'tracks': [
        {
            'id': 1,
            'position': [5.2, 12.8],
            'velocity': [2.1, -0.5],
            'acceleration': [0.1, 0.0],
            'confidence': 0.95,
            'state': 'active',
            'motion_model': 'cv'
        }
    ],
    'detections': [[x, y], ...],
    'processing_time': 0.008
}
```

### Analysis Statistics
```python
{
    'frame_statistics': {
        'total_frames': 1000,
        'avg_detections_per_frame': 8.5,
        'avg_active_tracks_per_frame': 2.3,
        'max_simultaneous_tracks': 7
    },
    'track_statistics': {
        'total_unique_tracks': 25,
        'avg_track_lifetime_frames': 45.2,
        'max_track_lifetime_frames': 120
    },
    'performance_statistics': {
        'avg_frame_processing_time_ms': 7.8,
        'effective_fps': 128.2,
        'real_time_factor': 6.41
    }
}
```

## Visualization Features

### Real-time GUI Display
- Live track visualization in 3D radar plots
- Track IDs and velocity vectors
- Statistics display (tracks, detections, FPS)
- Interactive controls for tuning parameters

### Recording Analysis Plots
- Track count over time
- Processing time performance
- Detection statistics
- Animated tracking visualization with MP4 export

## Performance Optimization

### Real-time Considerations
- Adaptive visualization update rates
- Efficient data structures (numpy arrays)
- Minimal memory allocation in processing loop
- Background processing for non-critical operations

### Memory Management
- Fixed-size history buffers (deque with maxlen)
- Automatic track cleanup
- Efficient matrix operations with scipy

### Tuning for Different Scenarios
```python
# Urban environment (many vehicles)
processor.max_tracks = 50
processor.confirmation_threshold = 2  # Faster confirmation

# Highway environment (high speeds)
processor.max_vehicle_speed = 80.0    # Higher speed limit
processor.gate_threshold = 12.0       # Larger association gate

# Parking lot (low speeds)
processor.max_vehicle_speed = 20.0    # Lower speed limit
processor.deletion_threshold = 10     # Longer coast time
```

## Troubleshooting

### Common Issues

1. **No tracks created**
   - Check SNR thresholds (reduce if signal is weak)
   - Verify clustering parameters (increase eps for sparser data)
   - Ensure point cloud format is correct

2. **Too many false tracks**
   - Increase confirmation_threshold
   - Tighten clustering parameters (reduce eps)
   - Increase SNR thresholds

3. **Tracks disappearing quickly**
   - Increase deletion_threshold
   - Check motion model appropriateness
   - Verify association gate settings

4. **Performance issues**
   - Reduce max_tracks limit
   - Optimize clustering parameters
   - Check visualization update rate

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use test script with verbose output
python test_gtrack.py single recordings/test_recording
```

## Advanced Configuration

### Custom Motion Models
Extend the MotionModel enum and modify Kalman filter parameters:
```python
class MotionModel(Enum):
    CONSTANT_VELOCITY = "cv"
    CONSTANT_ACCELERATION = "ca"
    COORDINATED_TURN = "ct"
    CUSTOM_VEHICLE = "custom"  # Add custom model
```

### SNR-based Parameter Adaptation
```python
# Customize clustering for specific radar configurations
processor.snr_thresholds = {
    'high': 180,    # Adjust for your radar sensitivity
    'medium': 90,
    'low': 40
}
```

## Integration Notes

### With Existing Radar Systems
- Compatible with mmWave radar point cloud data
- Supports both live and recorded data processing
- Integrates with PySide6 GUI frameworks
- JSON-based configuration and data formats

### Data Format Requirements
- Point cloud: `[x, y, z, doppler, snr_low, snr_high, intensity]`
- Only x,y coordinates are used for tracking
- Timestamp in milliseconds
- Frame number for sequencing

## Contributing

To extend or modify the GTRACK implementation:

1. **Algorithm improvements**: Modify `gtrack_algorithm.py`
2. **GUI enhancements**: Update `radar_gui.py`
3. **Analysis tools**: Extend `recording_analyzer.py`
4. **Testing**: Add cases to `test_gtrack.py`

## References

- Texas Instruments GTRACK Algorithm Documentation
- "Multiple Target Tracking with mmWave Radar" - TI Application Note
- Kalman Filtering and Object Tracking literature
- DBSCAN clustering algorithm papers 