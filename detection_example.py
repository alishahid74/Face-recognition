"""
Example: Object Detection

This script demonstrates how to use YOLO models for object detection.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detection import DetectionModelFactory, count_objects
import cv2


def detect_objects(image_path, model_name='YOLOv8n', confidence=0.5, iou=0.45, save_output=True):
    """
    Detect objects in an image
    
    Args:
        image_path: Path to the image file
        model_name: YOLO model variant
        confidence: Confidence threshold
        iou: IOU threshold for NMS
        save_output: Whether to save annotated image
    """
    print(f"\n{'='*60}")
    print(f"Object Detection Example")
    print(f"{'='*60}\n")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Create detector
    print(f"Loading {model_name} model...")
    detector = DetectionModelFactory.create_model(model_name)
    print(f"✓ Model loaded successfully\n")
    
    # Get model info
    model_info = detector.get_model_info()
    print("Model Information:")
    print(f"  Name: {model_info['name']}")
    print(f"  Framework: {model_info['framework']}")
    print(f"  Input Size: {model_info['input_size']}")
    print(f"  Classes: {model_info['num_classes']}")
    print(f"  Confidence Threshold: {confidence}")
    print(f"  IOU Threshold: {iou}\n")
    
    # Run detection
    print("Running detection...")
    results = detector.detect(image_path, confidence=confidence, iou=iou)
    print(f"✓ Detection complete\n")
    
    # Display results
    print(f"Detection Results:")
    print(f"{'-'*60}")
    print(f"Total objects detected: {results['count']}\n")
    
    if results['count'] > 0:
        # Count objects by class
        object_counts = count_objects(results['labels'])
        
        print("Object counts:")
        for obj_class, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {obj_class}: {count}")
        
        print(f"\nDetailed detections:")
        for i, (label, score, box) in enumerate(zip(results['labels'], results['scores'], results['boxes']), 1):
            x1, y1, x2, y2 = box
            print(f"  {i}. {label:15s} ({score*100:5.2f}%) - Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("No objects detected with current confidence threshold.")
        print(f"Try lowering the confidence threshold (current: {confidence})")
    
    print(f"{'-'*60}\n")
    
    # Save annotated image
    if save_output and results['count'] > 0:
        output_path = image_path.rsplit('.', 1)[0] + '_detected.' + image_path.rsplit('.', 1)[1]
        cv2.imwrite(output_path, results['image'])
        print(f"✓ Annotated image saved to: {output_path}\n")


def detect_video(video_path, model_name='YOLOv8n', confidence=0.5, save_output=False):
    """
    Detect objects in video
    
    Args:
        video_path: Path to video file or camera index (0 for webcam)
        model_name: YOLO model variant
        confidence: Confidence threshold
        save_output: Whether to save output video
    """
    print(f"\n{'='*60}")
    print(f"Video Object Detection Example")
    print(f"{'='*60}\n")
    
    # Create detector
    print(f"Loading {model_name} model...")
    detector = DetectionModelFactory.create_model(model_name)
    print(f"✓ Model loaded successfully\n")
    
    # Determine video source
    if video_path.isdigit():
        source = int(video_path)
        print(f"Using webcam (device {source})")
    else:
        source = video_path
        if not os.path.exists(source):
            print(f"Error: Video not found at {source}")
            return
        print(f"Processing video: {source}")
    
    print("\nPress 'q' to quit\n")
    print("Processing...")
    
    frame_count = 0
    total_detections = 0
    
    try:
        for frame_results in detector.detect_video(source, confidence=confidence, display=True):
            frame_count += 1
            total_detections += frame_results['count']
            
            if frame_count % 30 == 0:  # Print every 30 frames
                avg_detections = total_detections / frame_count
                print(f"Frame {frame_count}: {frame_results['count']} objects | Avg: {avg_detections:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    print(f"\n{'='*60}")
    print(f"Processed {frame_count} frames")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")
    print(f"{'='*60}\n")


def detect_specific_classes(image_path, target_classes, model_name='YOLOv8n', confidence=0.5):
    """
    Detect only specific object classes
    
    Args:
        image_path: Path to image
        target_classes: List of classes to detect (e.g., ['person', 'car'])
        model_name: YOLO model variant
        confidence: Confidence threshold
    """
    from models.detection import filter_detections
    
    print(f"\n{'='*60}")
    print(f"Detecting specific classes: {', '.join(target_classes)}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Create detector
    detector = DetectionModelFactory.create_model(model_name)
    
    # Run detection
    results = detector.detect(image_path, confidence=confidence)
    
    # Filter results
    filtered_boxes, filtered_labels, filtered_scores = filter_detections(
        results['boxes'],
        results['labels'],
        results['scores'],
        target_classes=target_classes
    )
    
    print(f"Total detections: {results['count']}")
    print(f"Filtered detections: {len(filtered_labels)}\n")
    
    if len(filtered_labels) > 0:
        for i, (label, score) in enumerate(zip(filtered_labels, filtered_scores), 1):
            print(f"  {i}. {label}: {score*100:.2f}%")
    else:
        print(f"No objects from classes {target_classes} detected")
    
    print(f"\n{'='*60}\n")


def benchmark_models(image_path, models=['YOLOv8n', 'YOLOv8s', 'YOLOv8m']):
    """
    Benchmark different YOLO models
    
    Args:
        image_path: Path to test image
        models: List of YOLO models to benchmark
    """
    import time
    
    print(f"\n{'='*60}")
    print(f"Model Benchmark")
    print(f"{'='*60}\n")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    results_data = []
    
    for model_name in models:
        print(f"Benchmarking {model_name}...")
        
        try:
            # Load model
            detector = DetectionModelFactory.create_model(model_name)
            
            # Warm-up run
            _ = detector.detect(image_path)
            
            # Timed runs
            times = []
            for _ in range(5):
                start_time = time.time()
                results = detector.detect(image_path)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            fps = 1 / avg_time
            
            results_data.append({
                'model': model_name,
                'avg_time': avg_time,
                'fps': fps,
                'detections': results['count']
            })
            
            print(f"  ✓ Avg time: {avg_time*1000:.2f}ms | FPS: {fps:.1f} | Detections: {results['count']}\n")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
    
    # Summary
    print(f"{'='*60}")
    print(f"Benchmark Summary:")
    print(f"{'-'*60}")
    print(f"{'Model':<15} {'Time (ms)':<15} {'FPS':<10} {'Detections':<12}")
    print(f"{'-'*60}")
    for data in results_data:
        print(f"{data['model']:<15} {data['avg_time']*1000:<15.2f} {data['fps']:<10.1f} {data['detections']:<12}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Detection Example')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file or camera index (0 for webcam)')
    parser.add_argument('--model', type=str, default='YOLOv8n',
                       choices=['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l'],
                       help='YOLO model variant (default: YOLOv8n)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold (default: 0.45)')
    parser.add_argument('--classes', type=str, nargs='+',
                       help='Detect only specific classes (e.g., person car)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark multiple models')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output')
    
    args = parser.parse_args()
    
    if args.benchmark and args.image:
        benchmark_models(args.image)
    elif args.classes and args.image:
        detect_specific_classes(args.image, args.classes, args.model, args.confidence)
    elif args.video:
        detect_video(args.video, args.model, args.confidence)
    elif args.image:
        detect_objects(args.image, args.model, args.confidence, args.iou, not args.no_save)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python detection_example.py --image street.jpg")
        print("  python detection_example.py --image people.jpg --model YOLOv8m --confidence 0.6")
        print("  python detection_example.py --video 0  # Use webcam")
        print("  python detection_example.py --video traffic.mp4")
        print("  python detection_example.py --image street.jpg --classes person car")
        print("  python detection_example.py --image test.jpg --benchmark")
