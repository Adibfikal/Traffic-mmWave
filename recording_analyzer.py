"""
Recording Analyzer Tool
Processes recordings from the recordings folder using GTRACK algorithm
and provides analysis, visualization and export capabilities.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

from gtrack_algorithm import GTRACKProcessor, process_recording_file

class RecordingAnalyzer:
    """Tool for analyzing recordings with GTRACK tracking"""
    
    def __init__(self, recordings_path: str = "recordings"):
        """
        Initialize the recording analyzer
        
        Args:
            recordings_path: Path to recordings folder
        """
        self.recordings_path = recordings_path
        self.processor = GTRACKProcessor(max_tracks=25)
        
    def list_recordings(self) -> List[str]:
        """List all available recordings"""
        recordings = []
        
        if not os.path.exists(self.recordings_path):
            print(f"Recordings path not found: {self.recordings_path}")
            return recordings
        
        for item in os.listdir(self.recordings_path):
            recording_path = os.path.join(self.recordings_path, item)
            radar_data_path = os.path.join(recording_path, "radar_data.json")
            
            if os.path.isdir(recording_path) and os.path.exists(radar_data_path):
                recordings.append(item)
        
        return sorted(recordings)
    
    def analyze_recording(self, recording_name: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze a specific recording with GTRACK
        
        Args:
            recording_name: Name of the recording folder
            save_results: Whether to save analysis results to file
            
        Returns:
            Analysis results dictionary
        """
        radar_data_path = os.path.join(self.recordings_path, recording_name, "radar_data.json")
        
        if not os.path.exists(radar_data_path):
            raise FileNotFoundError(f"Radar data not found: {radar_data_path}")
        
        print(f"Analyzing recording: {recording_name}")
        print(f"Processing file: {radar_data_path}")
        
        # Reset processor for fresh analysis
        self.processor = GTRACKProcessor(max_tracks=25)
        
        # Process the recording
        start_time = datetime.now()
        results = process_recording_file(radar_data_path, self.processor)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate comprehensive statistics
        stats = self._calculate_statistics(results, processing_time)
        
        # Prepare final analysis
        analysis = {
            'recording_name': recording_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'frame_results': results,
            'total_processing_time_seconds': processing_time
        }
        
        # Save results if requested
        if save_results:
            self._save_analysis_results(recording_name, analysis)
        
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _calculate_statistics(self, results: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the analysis"""
        if not results:
            return {}
        
        # Frame statistics
        total_frames = len(results)
        processing_times = [r['processing_time'] for r in results]
        active_tracks_per_frame = [r['frame_stats']['active_tracks'] for r in results]
        detections_per_frame = [r['frame_stats']['total_detections'] for r in results]
        valid_detections_per_frame = [r['frame_stats']['valid_detections'] for r in results]
        
        # Track statistics
        all_track_ids = set()
        for result in results:
            for track in result['tracks']:
                all_track_ids.add(track['id'])
        
        # Calculate track lifetimes
        track_lifetimes = {}
        for track_id in all_track_ids:
            first_frame = None
            last_frame = None
            
            for frame_idx, result in enumerate(results):
                track_found = any(t['id'] == track_id for t in result['tracks'])
                if track_found:
                    if first_frame is None:
                        first_frame = frame_idx
                    last_frame = frame_idx
            
            if first_frame is not None and last_frame is not None:
                track_lifetimes[track_id] = last_frame - first_frame + 1
        
        # Performance statistics
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        return {
            'frame_statistics': {
                'total_frames': total_frames,
                'avg_detections_per_frame': np.mean(detections_per_frame),
                'avg_valid_detections_per_frame': np.mean(valid_detections_per_frame),
                'detection_rate': np.mean(valid_detections_per_frame) / np.mean(detections_per_frame) if np.mean(detections_per_frame) > 0 else 0,
                'avg_active_tracks_per_frame': np.mean(active_tracks_per_frame),
                'max_simultaneous_tracks': np.max(active_tracks_per_frame),
                'min_active_tracks': np.min(active_tracks_per_frame)
            },
            'track_statistics': {
                'total_unique_tracks': len(all_track_ids),
                'avg_track_lifetime_frames': np.mean(list(track_lifetimes.values())) if track_lifetimes else 0,
                'max_track_lifetime_frames': np.max(list(track_lifetimes.values())) if track_lifetimes else 0,
                'min_track_lifetime_frames': np.min(list(track_lifetimes.values())) if track_lifetimes else 0,
                'track_lifetimes': track_lifetimes
            },
            'performance_statistics': {
                'total_processing_time_seconds': processing_time,
                'avg_frame_processing_time_ms': avg_processing_time * 1000,
                'max_frame_processing_time_ms': max_processing_time * 1000,
                'effective_fps': fps,
                'real_time_factor': fps / 20.0 if fps > 0 else 0  # Assuming 20 FPS target
            }
        }
    
    def _save_analysis_results(self, recording_name: str, analysis: Dict[str, Any]):
        """Save analysis results to JSON file"""
        output_dir = os.path.join(self.recordings_path, recording_name)
        output_file = os.path.join(output_dir, "gtrack_analysis.json")
        
        # Create simplified version for saving (without frame_results to reduce file size)
        save_analysis = {
            'recording_name': analysis['recording_name'],
            'analysis_timestamp': analysis['analysis_timestamp'],
            'statistics': analysis['statistics'],
            'total_processing_time_seconds': analysis['total_processing_time_seconds']
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_analysis, f, indent=2)
        
        print(f"Analysis results saved to: {output_file}")
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print analysis summary to console"""
        stats = analysis['statistics']
        
        print(f"\n=== GTRACK Analysis Summary for {analysis['recording_name']} ===")
        print(f"Analysis completed at: {analysis['analysis_timestamp']}")
        
        print(f"\nFrame Statistics:")
        print(f"  Total frames: {stats['frame_statistics']['total_frames']}")
        print(f"  Avg detections per frame: {stats['frame_statistics']['avg_detections_per_frame']:.1f}")
        print(f"  Avg valid detections per frame: {stats['frame_statistics']['avg_valid_detections_per_frame']:.1f}")
        print(f"  Detection rate: {stats['frame_statistics']['detection_rate']:.2%}")
        print(f"  Avg active tracks per frame: {stats['frame_statistics']['avg_active_tracks_per_frame']:.1f}")
        print(f"  Max simultaneous tracks: {stats['frame_statistics']['max_simultaneous_tracks']}")
        
        print(f"\nTrack Statistics:")
        print(f"  Total unique tracks created: {stats['track_statistics']['total_unique_tracks']}")
        print(f"  Avg track lifetime: {stats['track_statistics']['avg_track_lifetime_frames']:.1f} frames")
        print(f"  Max track lifetime: {stats['track_statistics']['max_track_lifetime_frames']} frames")
        
        print(f"\nPerformance Statistics:")
        print(f"  Total processing time: {stats['performance_statistics']['total_processing_time_seconds']:.2f} seconds")
        print(f"  Avg frame processing time: {stats['performance_statistics']['avg_frame_processing_time_ms']:.2f} ms")
        print(f"  Effective FPS: {stats['performance_statistics']['effective_fps']:.1f}")
        print(f"  Real-time factor: {stats['performance_statistics']['real_time_factor']:.2f}x")
        
        if stats['performance_statistics']['real_time_factor'] >= 1.0:
            print("  ✓ Real-time capable!")
        else:
            print("  ⚠ Not real-time capable")
    
    def visualize_tracking_results(self, recording_name: str, analysis: Optional[Dict[str, Any]] = None, 
                                 save_animation: bool = False) -> None:
        """
        Create visualization of tracking results
        
        Args:
            recording_name: Name of the recording
            analysis: Pre-computed analysis results (optional)
            save_animation: Whether to save animation as video file
        """
        if analysis is None:
            analysis = self.analyze_recording(recording_name, save_results=False)
        
        results = analysis['frame_results']
        
        if not results:
            print("No results to visualize")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Setup tracking visualization (left plot)
        ax1.set_title(f'GTRACK Vehicle Tracking - {recording_name}')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Setup statistics plot (right plot)
        ax2.set_title('Tracking Statistics')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Count')
        
        # Prepare data for statistics plot
        frame_numbers = list(range(len(results)))
        active_tracks = [r['frame_stats']['active_tracks'] for r in results]
        valid_detections = [r['frame_stats']['valid_detections'] for r in results]
        
        # Plot statistics
        ax2.plot(frame_numbers, active_tracks, 'b-', label='Active Tracks', linewidth=2)
        ax2.plot(frame_numbers, valid_detections, 'r-', alpha=0.7, label='Valid Detections')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Animation function
        def animate(frame_idx):
            ax1.clear()
            
            result = results[frame_idx]
            
            ax1.set_title(f'GTRACK Vehicle Tracking - Frame {frame_idx + 1}/{len(results)}')
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.grid(True, alpha=0.3)
            
            # Plot detections
            if result['detections']:
                detections = np.array(result['detections'])
                ax1.scatter(detections[:, 0], detections[:, 1], 
                           c='lightgray', s=20, alpha=0.6, label='Detections')
            
            # Plot tracks
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            for i, track in enumerate(result['tracks']):
                color = colors[track['id'] % 10]
                x, y = track['position']
                vx, vy = track['velocity']
                
                # Plot track position
                ax1.scatter(x, y, c=[color], s=100, marker='o', 
                           edgecolor='black', linewidth=1, label=f"Track {track['id']}" if i < 5 else "")
                
                # Plot velocity vector
                if np.linalg.norm([vx, vy]) > 0.5:  # Only show significant velocities
                    ax1.arrow(x, y, vx*2, vy*2, head_width=1, head_length=0.5, 
                             fc=color, ec='black', alpha=0.7)
                
                # Add track ID label
                ax1.text(x + 1, y + 1, f'T{track["id"]}', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
            
            # Set axis limits based on data
            if result['detections'] or result['tracks']:
                all_x, all_y = [], []
                
                for det in result['detections']:
                    all_x.append(det[0])
                    all_y.append(det[1])
                
                for track in result['tracks']:
                    all_x.append(track['position'][0])
                    all_y.append(track['position'][1])
                
                if all_x and all_y:
                    margin = 10
                    ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
                    ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)
            
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Update progress line on statistics plot
            ax2.axvline(x=frame_idx, color='green', linestyle='--', alpha=0.7)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(results), 
                                     interval=100, blit=False, repeat=True)
        
        # Save animation if requested
        if save_animation:
            output_path = os.path.join(self.recordings_path, recording_name, "gtrack_animation.mp4")
            print(f"Saving animation to: {output_path}")
            anim.save(output_path, writer='ffmpeg', fps=10)
            print("Animation saved successfully!")
        
        plt.tight_layout()
        plt.show()
    
    def analyze_all_recordings(self) -> Dict[str, Any]:
        """Analyze all recordings in the recordings folder"""
        recordings = self.list_recordings()
        
        if not recordings:
            print("No recordings found in the recordings folder")
            return {}
        
        print(f"Found {len(recordings)} recordings to analyze")
        
        all_analyses = {}
        
        for recording in recordings:
            try:
                print(f"\n{'='*50}")
                analysis = self.analyze_recording(recording, save_results=True)
                all_analyses[recording] = analysis
            except Exception as e:
                print(f"Error analyzing {recording}: {str(e)}")
                continue
        
        # Generate comparative report
        self._generate_comparative_report(all_analyses)
        
        return all_analyses
    
    def _generate_comparative_report(self, all_analyses: Dict[str, Any]):
        """Generate a comparative report across all recordings"""
        if not all_analyses:
            return
        
        print(f"\n{'='*70}")
        print("COMPARATIVE ANALYSIS ACROSS ALL RECORDINGS")
        print(f"{'='*70}")
        
        # Collect comparative statistics
        recording_stats = []
        
        for recording_name, analysis in all_analyses.items():
            stats = analysis['statistics']
            recording_stats.append({
                'name': recording_name,
                'frames': stats['frame_statistics']['total_frames'],
                'avg_tracks': stats['frame_statistics']['avg_active_tracks_per_frame'],
                'max_tracks': stats['frame_statistics']['max_simultaneous_tracks'],
                'total_tracks': stats['track_statistics']['total_unique_tracks'],
                'avg_lifetime': stats['track_statistics']['avg_track_lifetime_frames'],
                'fps': stats['performance_statistics']['effective_fps'],
                'real_time_factor': stats['performance_statistics']['real_time_factor']
            })
        
        # Print comparative table
        print(f"{'Recording':<20} {'Frames':<8} {'Avg Tracks':<10} {'Max Tracks':<10} {'Total Tracks':<12} {'FPS':<8} {'RT Factor':<10}")
        print("-" * 85)
        
        for stat in recording_stats:
            print(f"{stat['name']:<20} {stat['frames']:<8} {stat['avg_tracks']:<10.1f} "
                  f"{stat['max_tracks']:<10} {stat['total_tracks']:<12} "
                  f"{stat['fps']:<8.1f} {stat['real_time_factor']:<10.2f}")
        
        # Overall statistics
        total_frames = sum(s['frames'] for s in recording_stats)
        avg_fps = np.mean([s['fps'] for s in recording_stats])
        
        print(f"\nOverall Statistics:")
        print(f"  Total frames processed: {total_frames}")
        print(f"  Average FPS across all recordings: {avg_fps:.1f}")
        print(f"  Recordings capable of real-time: {sum(1 for s in recording_stats if s['real_time_factor'] >= 1.0)}/{len(recording_stats)}")

def main():
    """Command line interface for the recording analyzer"""
    parser = argparse.ArgumentParser(description='GTRACK Recording Analyzer')
    parser.add_argument('--recordings-path', default='recordings', 
                       help='Path to recordings folder (default: recordings)')
    parser.add_argument('--recording', type=str, 
                       help='Specific recording to analyze')
    parser.add_argument('--list', action='store_true', 
                       help='List all available recordings')
    parser.add_argument('--analyze-all', action='store_true', 
                       help='Analyze all recordings')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization of tracking results')
    parser.add_argument('--save-animation', action='store_true', 
                       help='Save animation as video file')
    
    args = parser.parse_args()
    
    analyzer = RecordingAnalyzer(args.recordings_path)
    
    if args.list:
        recordings = analyzer.list_recordings()
        print(f"Available recordings ({len(recordings)}):")
        for recording in recordings:
            print(f"  - {recording}")
        return
    
    if args.analyze_all:
        analyzer.analyze_all_recordings()
        return
    
    if args.recording:
        analysis = analyzer.analyze_recording(args.recording, save_results=True)
        
        if args.visualize:
            analyzer.visualize_tracking_results(args.recording, analysis, args.save_animation)
    else:
        print("Please specify --recording, --analyze-all, or --list")
        print("Use --help for more information")

if __name__ == "__main__":
    main() 