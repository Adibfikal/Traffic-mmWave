# IWR6843ISK mmWave Radar GUI

A PyQt-based GUI application for reading and visualizing data from the IWR6843ISK mmWave radar sensor. The application features real-time point cloud visualization, object tracking using Particle Filters and HDBSCAN clustering, and trajectory visualization.

## Features

- **Real-time Point Cloud Visualization**: Display 3D point cloud data from the radar
- **2D Bird's Eye View**: Top-down view of tracked objects starting from y=0 (radar position)
- **Object Tracking**: Track multiple objects using Particle Filter and HDBSCAN clustering
- **Trajectory Visualization**: Show movement trails for tracked objects in both 3D and 2D views
- **Interactive 3D Display**: Rotate, zoom, and pan the 3D visualization
- **Serial Port Configuration**: Easy configuration of CLI and data ports
- **Status Monitoring**: Real-time status updates and statistics

## Requirements

- Python 3.7+
- IWR6843ISK mmWave radar sensor
- USB connection to the radar (two COM ports required)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Hardware Setup

1. Connect the IWR6843ISK radar to your computer via USB
2. Install the necessary drivers (if on Windows)
3. Note the COM port numbers assigned to the radar:
   - CLI Port (Command Line Interface) - typically the lower numbered port
   - Data Port - typically the higher numbered port

## Usage

1. Run the application:
```bash
python radar_gui.py
```

2. Configure the COM ports:
   - Enter the CLI port (e.g., COM4)
   - Enter the Data port (e.g., COM3)

3. Click "Connect" to establish connection with the radar

4. Click "Send Configuration" to configure the radar parameters

5. Click "Start Sensor" to begin data acquisition

6. The visualization will show two views:
   
   **3D View (Top Panel):**
   - White points: Raw point cloud data
   - Colored points: Tracked objects (each color represents a different object)
   - Colored lines: Movement trails showing object trajectories
   
   **2D Bird's Eye View (Bottom Panel):**
   - White dots: Raw point cloud data projected on X-Y plane
   - Colored circles: Tracked objects from top-down perspective
   - Colored lines: Movement trails on the ground plane
   - White circle at origin: Radar position (0, 0)

## Configuration

The radar configuration is stored in `xwr68xx_config.cfg`. You can modify this file to change:
- Range resolution
- Velocity resolution
- Frame rate
- Detection parameters

## Tracking Algorithm

The application uses a sophisticated tracking system that combines:

1. **HDBSCAN Clustering**: Groups radar points into potential objects
2. **Particle Filter**: Tracks object states (position and velocity) over time
3. **Data Association**: Matches new detections with existing tracks

### Tracking Parameters

You can adjust tracking parameters in the `ObjectTracker` class:
- `min_cluster_size`: Minimum points to form a cluster
- `max_distance`: Maximum distance for track association
- `max_frames_missing`: Frames before a track is deleted

## Troubleshooting

### Connection Issues
- Ensure the radar is properly connected and powered
- Check that the COM port numbers are correct
- Try different USB ports or cables
- On Windows, check Device Manager for the correct COM ports

### No Data Displayed
- Verify the radar configuration was sent successfully
- Check that the sensor is started
- Ensure there are objects in the radar's field of view
- Check the status window for error messages

### Performance Issues
- Reduce the number of particles in the Particle Filter
- Increase the update interval in the GUI
- Close other applications that might be using the COM ports

## References

This project was inspired by:
- [OpenRadar](https://github.com/PreSenseRadar/OpenRadar)
- [IWR6843-Read-Data-Python-MMWAVE-SDK](https://github.com/kirkster96/IWR6843-Read-Data-Python-MMWAVE-SDK)

## License

This project is provided as-is for educational and research purposes. 