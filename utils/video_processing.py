"""
Advanced Video Processing Module
Includes: Object tracking, frame logging, video export, analytics
"""

import cv2
import numpy as np
from datetime import datetime
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import time


class ObjectTracker:
    """
    Multi-object tracking using centroid tracking or DeepSORT
    """
    
    def __init__(self, max_disappeared: int = 50, max_distance: int = 50):
        """
        Initialize object tracker
        
        Args:
            max_disappeared: Max frames an object can disappear before deregistration
            max_distance: Maximum distance for matching objects between frames
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary of object_id -> centroid
        self.disappeared = {}  # Dictionary of object_id -> disappeared_count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Track history for motion trails
        self.track_history = defaultdict(lambda: deque(maxlen=30))
    
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.track_history:
            del self.track_history[object_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
        
        Returns:
            Dictionary of object_id -> centroid
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Calculate centroids from bounding boxes
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distance between each pair
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = np.linalg.norm(
                        np.array(object_centroids[i]) - input_centroids[j]
                    )
            
            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update track history
                self.track_history[object_id].append(tuple(input_centroids[col]))
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched rows (disappeared objects)
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Handle unmatched cols (new objects)
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects
    
    def get_track_history(self, object_id: int) -> List[Tuple[int, int]]:
        """Get motion trail for an object"""
        return list(self.track_history.get(object_id, []))


class VideoLogger:
    """
    Log video detection results to file
    """
    
    def __init__(self, output_dir: str = "output/logs"):
        """
        Initialize video logger
        
        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"detection_log_{timestamp}.json"
        self.csv_file = self.output_dir / f"detection_log_{timestamp}.csv"
        
        self.frame_logs = []
        self.csv_writer = None
        self.csv_file_handle = None
        
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file"""
        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow([
            'Frame', 'Timestamp', 'Object_ID', 'Class', 'Confidence',
            'X1', 'Y1', 'X2', 'Y2', 'Centroid_X', 'Centroid_Y'
        ])
    
    def log_frame(self, frame_number: int, detections: Dict):
        """
        Log detections for a frame
        
        Args:
            frame_number: Frame index
            detections: Dictionary with detection results
        """
        timestamp = datetime.now().isoformat()
        
        frame_data = {
            'frame': frame_number,
            'timestamp': timestamp,
            'detections': []
        }
        
        # Log each detection
        if 'boxes' in detections and len(detections['boxes']) > 0:
            for i, (box, label, score) in enumerate(zip(
                detections['boxes'],
                detections.get('labels', []),
                detections.get('scores', [])
            )):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                detection_data = {
                    'object_id': detections.get('object_ids', [i])[i] if 'object_ids' in detections else i,
                    'class': label,
                    'confidence': float(score),
                    'bbox': {
                        'x1': float(x1), 'y1': float(y1),
                        'x2': float(x2), 'y2': float(y2)
                    },
                    'centroid': {'x': float(cx), 'y': float(cy)}
                }
                
                frame_data['detections'].append(detection_data)
                
                # Write to CSV
                self.csv_writer.writerow([
                    frame_number, timestamp,
                    detection_data['object_id'],
                    label, score,
                    x1, y1, x2, y2, cx, cy
                ])
        
        self.frame_logs.append(frame_data)
    
    def save(self):
        """Save logs to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump({
                'total_frames': len(self.frame_logs),
                'frames': self.frame_logs
            }, f, indent=2)
        
        if self.csv_file_handle:
            self.csv_file_handle.close()
        
        print(f"Logs saved to:")
        print(f"  JSON: {self.log_file}")
        print(f"  CSV: {self.csv_file}")


class VideoExporter:
    """
    Export processed video with annotations
    """
    
    def __init__(self, output_path: str, fps: int = 30, codec: str = 'mp4v'):
        """
        Initialize video exporter
        
        Args:
            output_path: Path to save video
            fps: Frames per second
            codec: Video codec (mp4v, XVID, etc.)
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.writer = None
        self.frame_size = None
    
    def write_frame(self, frame: np.ndarray):
        """
        Write frame to video
        
        Args:
            frame: Frame to write
        """
        if self.writer is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.writer = cv2.VideoWriter(
                self.output_path,
                self.codec,
                self.fps,
                self.frame_size
            )
        
        self.writer.write(frame)
    
    def close(self):
        """Close video writer"""
        if self.writer:
            self.writer.release()
            print(f"Video saved to: {self.output_path}")


class AdvancedVideoProcessor:
    """
    Advanced video processor with tracking, logging, and export
    """
    
    def __init__(self, detector, enable_tracking: bool = True,
                 enable_logging: bool = True, enable_export: bool = False):
        """
        Initialize advanced video processor
        
        Args:
            detector: Object detection model
            enable_tracking: Enable multi-object tracking
            enable_logging: Enable frame logging
            enable_export: Enable video export
        """
        self.detector = detector
        self.enable_tracking = enable_tracking
        self.enable_logging = enable_logging
        self.enable_export = enable_export
        
        self.tracker = ObjectTracker() if enable_tracking else None
        self.logger = VideoLogger() if enable_logging else None
        self.exporter = None
        
        # Analytics
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = defaultdict(int)
        self.fps_history = deque(maxlen=30)
    
    def process_video(self, video_source, output_path: Optional[str] = None,
                     display: bool = True, max_frames: Optional[int] = None):
        """
        Process video with tracking and logging
        
        Args:
            video_source: Video file path or camera index
            output_path: Path to save output video
            display: Whether to display results
            max_frames: Maximum frames to process
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize exporter if needed
        if self.enable_export and output_path:
            self.exporter = VideoExporter(output_path, fps)
        
        try:
            while cap.isOpened():
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Stop if max frames reached
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Run detection
                # Save frame temporarily for detection
                temp_path = "/tmp/temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                detection_results = self.detector.detect(temp_path)
                
                # Update tracking if enabled
                if self.enable_tracking and len(detection_results['boxes']) > 0:
                    tracked_objects = self.tracker.update(detection_results['boxes'])
                    detection_results['object_ids'] = list(tracked_objects.keys())
                    
                    # Draw tracking IDs and trails
                    frame = self._draw_tracking(frame, tracked_objects, detection_results)
                else:
                    frame = detection_results.get('image', frame)
                
                # Log frame if enabled
                if self.enable_logging:
                    self.logger.log_frame(self.frame_count, detection_results)
                
                # Update analytics
                self._update_analytics(detection_results)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = 1 / elapsed if elapsed > 0 else 0
                self.fps_history.append(current_fps)
                
                # Draw FPS and frame info
                frame = self._draw_info(frame, current_fps)
                
                # Export frame if enabled
                if self.exporter:
                    self.exporter.write_frame(frame)
                
                # Display if enabled
                if display:
                    cv2.imshow('Advanced Video Processing', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_count += 1
                
                yield {
                    'frame': frame,
                    'frame_number': self.frame_count,
                    'detections': detection_results,
                    'fps': current_fps
                }
        
        finally:
            cap.release()
            
            if self.exporter:
                self.exporter.close()
            
            if self.enable_logging:
                self.logger.save()
            
            if display:
                cv2.destroyAllWindows()
            
            self._print_summary()
    
    def _draw_tracking(self, frame, tracked_objects, detections):
        """Draw tracking information on frame"""
        annotated_frame = frame.copy()
        
        for obj_id, centroid in tracked_objects.items():
            # Draw centroid
            cv2.circle(annotated_frame, tuple(centroid), 5, (0, 255, 0), -1)
            
            # Draw ID
            cv2.putText(annotated_frame, f"ID:{obj_id}", 
                       (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw motion trail
            trail = self.tracker.get_track_history(obj_id)
            if len(trail) > 1:
                points = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], False, (0, 255, 255), 2)
        
        return annotated_frame
    
    def _draw_info(self, frame, fps):
        """Draw FPS and frame information"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw frame count
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw detection count
        cv2.putText(annotated_frame, f"Objects: {self.total_detections}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _update_analytics(self, detections):
        """Update analytics"""
        if 'labels' in detections:
            for label in detections['labels']:
                self.class_counts[label] += 1
            self.total_detections = sum(self.class_counts.values())
    
    def _print_summary(self):
        """Print processing summary"""
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        print("\n" + "="*60)
        print("Video Processing Summary")
        print("="*60)
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total Detections: {self.total_detections}")
        print("\nClass Distribution:")
        for cls, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count}")
        print("="*60)
