#!/usr/bin/env python3
"""
GTRACK Test Script
Quick test of the GTRACK algorithm implementation with recorded data
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our GTRACK implementation
from gtrack_algorithm import GTRACKProcessor, process_recording_file
from recording_analyzer import RecordingAnalyzer

def test_single_recording(recording_path="recordings/session_20250704_171209"):
    """Test GTRACK on a single recording"""
    radar_data_file = os.path.join(recording_path, "radar_data.json")
    
    if not os.path.exists(radar_data_file):
        print(f"Error: Recording not found at {radar_data_file}")
        return False
    
    print(f"Testing GTRACK on recording: {recording_path}")
    print("=" * 50)
    
    # Initialize GTRACK processor
    processor = GTRACKProcessor(max_tracks=25)
    
    # Load and process recording
    try:
        with open(radar_data_file, 'r') as f:
            frames = json.load(f)
        
        print(f"Loaded {len(frames)} frames from recording")
        
        # Process first 100 frames for quick test
        test_frames = frames[:min(100, len(frames))]
        results = []
        
        print("Processing frames...")
        start_time = time.time()
        
        for i, frame in enumerate(test_frames):
            result = processor.process_frame(frame)
            results.append(result)
            
            # Print progress every 20 frames
            if (i + 1) % 20 == 0:
                tracks = result['frame_stats']['active_tracks']
                detections = result['frame_stats']['valid_detections']
                proc_time = result['processing_time'] * 1000
                print(f"  Frame {i+1:3d}: {tracks} tracks, {detections} detections, {proc_time:.1f}ms")
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        total_tracks_created = len(set(track['id'] for result in results for track in result['tracks']))
        max_simultaneous_tracks = max(r['frame_stats']['active_tracks'] for r in results)
        avg_processing_time = np.mean([r['processing_time'] for r in results])
        avg_detections = np.mean([r['frame_stats']['valid_detections'] for r in results])
        
        print(f"\nResults Summary:")
        print(f"  Total processing time: {processing_time:.2f}s")
        print(f"  Average frame processing: {avg_processing_time*1000:.2f}ms")
        print(f"  Effective FPS: {len(test_frames)/processing_time:.1f}")
        print(f"  Total unique tracks created: {total_tracks_created}")
        print(f"  Max simultaneous tracks: {max_simultaneous_tracks}")
        print(f"  Average detections per frame: {avg_detections:.1f}")
        
        # Simple visualization of track count over time
        track_counts = [r['frame_stats']['active_tracks'] for r in results]
        detection_counts = [r['frame_stats']['valid_detections'] for r in results]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(track_counts, 'b-', label='Active Tracks', linewidth=2)
        plt.plot(detection_counts, 'r-', alpha=0.7, label='Valid Detections')
        plt.title('GTRACK Performance Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        processing_times = [r['processing_time'] * 1000 for r in results]
        plt.plot(processing_times, 'g-', label='Processing Time', linewidth=2)
        plt.title('Processing Time per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gtrack_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nTest completed successfully! ✓")
        print(f"Results visualization saved as 'gtrack_test_results.png'")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def test_all_recordings():
    """Test GTRACK on all available recordings"""
    analyzer = RecordingAnalyzer()
    recordings = analyzer.list_recordings()
    
    if not recordings:
        print("No recordings found!")
        return
    
    print(f"Found {len(recordings)} recordings")
    print("Testing GTRACK on first few recordings...")
    
    for i, recording in enumerate(recordings[:3]):  # Test first 3 recordings
        print(f"\n{'-'*50}")
        print(f"Testing recording {i+1}/{min(3, len(recordings))}: {recording}")
        
        try:
            analysis = analyzer.analyze_recording(recording, save_results=True)
            print(f"✓ Recording {recording} processed successfully")
        except Exception as e:
            print(f"✗ Error processing {recording}: {e}")

def run_quick_test():
    """Run a quick test to verify GTRACK works"""
    print("GTRACK Algorithm Quick Test")
    print("=" * 30)
    
    # Test with synthetic data first
    print("1. Testing with synthetic data...")
    
    processor = GTRACKProcessor(max_tracks=5)
    
    # Create a simple synthetic frame
    synthetic_frame = {
        'timestamp': 1000.0,
        'frameData': {
            'error': 0,
            'frameNum': 1,
            'pointCloud': [
                [5.0, 10.0, 0.0, 0.0, 150.0, 200.0, 255.0],  # Vehicle 1
                [5.2, 10.1, 0.0, 0.0, 145.0, 195.0, 255.0],  # Vehicle 1 (cluster)
                [-3.0, 8.0, 0.0, 0.0, 120.0, 180.0, 255.0],  # Vehicle 2
                [15.0, 5.0, 0.0, 0.0, 100.0, 160.0, 255.0],  # Vehicle 3
            ]
        }
    }
    
    try:
        result = processor.process_frame(synthetic_frame)
        print(f"  ✓ Synthetic frame processed successfully")
        print(f"    Detections: {result['frame_stats']['valid_detections']}")
        print(f"    Tracks: {result['frame_stats']['active_tracks']}")
        print(f"    Processing time: {result['processing_time']*1000:.2f}ms")
    except Exception as e:
        print(f"  ✗ Error processing synthetic frame: {e}")
        return False
    
    # Test with real recording if available
    print("\n2. Testing with real recording data...")
    
    recordings_dir = "recordings"
    if os.path.exists(recordings_dir):
        # Find the first available recording
        for item in os.listdir(recordings_dir):
            recording_path = os.path.join(recordings_dir, item)
            radar_file = os.path.join(recording_path, "radar_data.json")
            
            if os.path.exists(radar_file):
                print(f"  Found recording: {item}")
                success = test_single_recording(recording_path)
                if success:
                    print("  ✓ Real data test completed successfully")
                else:
                    print("  ✗ Real data test failed")
                break
        else:
            print("  No recordings found, skipping real data test")
    else:
        print("  No recordings directory found, skipping real data test")
    
    print("\nQuick test completed!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            run_quick_test()
        elif command == "single":
            recording_path = sys.argv[2] if len(sys.argv) > 2 else "recordings/session_20250704_171209"
            test_single_recording(recording_path)
        elif command == "all":
            test_all_recordings()
        else:
            print("Usage:")
            print("  python test_gtrack.py quick          # Quick test with synthetic data")
            print("  python test_gtrack.py single [path]  # Test single recording")
            print("  python test_gtrack.py all            # Test all recordings")
    else:
        # Default: run quick test
        run_quick_test() 