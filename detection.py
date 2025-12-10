"""
Object Detection Model Handlers
"""

from ultralytics import YOLO
import cv2
import numpy as np
from config import MODELS_CONFIG, COCO_CLASSES, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD



class YOLODetector:
    """YOLO-based object detection"""
    
    def __init__(self, model_name='YOLOv8n'):
        self.model_name = model_name
        self.model = None
        self.class_names = COCO_CLASSES
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained YOLO model"""
        print(f"Loading {self.model_name} model...")
        
        # Map model names to YOLO model files
        model_map = {
            'YOLOv8n': 'yolov8n.pt',
            'YOLOv8s': 'yolov8s.pt',
            'YOLOv8m': 'yolov8m.pt',
            'YOLOv8l': 'yolov8l.pt',
        }
        
        if self.model_name not in model_map:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        try:
            self.model = YOLO(model_map[self.model_name])
            print(f"{self.model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect(self, image_path, confidence=None, iou=None):
        """
        Detect objects in image
        
        Args:
            image_path: Path to input image
            confidence: Confidence threshold (optional)
            iou: IOU threshold for NMS (optional)
        
        Returns:
            Dictionary containing boxes, labels, scores, and annotated image
        """
        conf_threshold = confidence if confidence is not None else self.confidence_threshold
        iou_threshold = iou if iou is not None else self.iou_threshold
        
        # Run inference
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Extract results
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Get class names
        labels = [self.model.names[class_id] for class_id in class_ids]
        
        # Get annotated image
        annotated_image = result.plot()
        
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'class_ids': class_ids,
            'image': annotated_image,
            'count': len(boxes)
        }
    
    def detect_video(self, video_source, confidence=None, iou=None, display=True):
        """
        Detect objects in video stream
        
        Args:
            video_source: Video source (0 for webcam, or video file path)
            confidence: Confidence threshold
            iou: IOU threshold
            display: Whether to display results
        
        Yields:
            Detection results for each frame
        """
        conf_threshold = confidence if confidence is not None else self.confidence_threshold
        iou_threshold = iou if iou is not None else self.iou_threshold
        
        cap = cv2.VideoCapture(video_source)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                
                # Extract results
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                labels = [self.model.names[class_id] for class_id in class_ids]
                
                # Get annotated frame
                annotated_frame = result.plot()
                
                if display:
                    cv2.imshow('YOLO Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                yield {
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores,
                    'class_ids': class_ids,
                    'frame': annotated_frame,
                    'count': len(boxes)
                }
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold"""
        self.confidence_threshold = threshold
    
    def set_iou_threshold(self, threshold):
        """Set IOU threshold"""
        self.iou_threshold = threshold
    
    def get_model_info(self):
        """Get model information"""
        config = MODELS_CONFIG['object_detection'][self.model_name]
        return {
            'name': self.model_name,
            'framework': config['framework'],
            'input_size': config['input_size'],
            'description': config['description'],
            'num_classes': len(self.class_names),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold
        }


class DetectionModelFactory:
    """Factory class to create object detection models"""
    
    @staticmethod
    def create_model(model_name):
        """
        Create and return an object detection model
        
        Args:
            model_name: Name of the model to create
        
        Returns:
            Model instance
        """
        if model_name not in MODELS_CONFIG['object_detection']:
            raise ValueError(f"Unsupported model: {model_name}")
        
        config = MODELS_CONFIG['object_detection'][model_name]
        framework = config['framework']
        
        if framework == 'ultralytics':
            return YOLODetector(model_name)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    @staticmethod
    def get_available_models():
        """Get list of available detection models"""
        return list(MODELS_CONFIG['object_detection'].keys())
    
    @staticmethod
    def get_model_info(model_name):
        """Get information about a specific model"""
        if model_name in MODELS_CONFIG['object_detection']:
            return MODELS_CONFIG['object_detection'][model_name]
        return None


def count_objects(labels):
    """
    Count detected objects by class
    
    Args:
        labels: List of object labels
    
    Returns:
        Dictionary with object counts
    """
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def filter_detections(boxes, labels, scores, target_classes=None, min_confidence=0.5):
    """
    Filter detections by class and confidence
    
    Args:
        boxes: Bounding boxes
        labels: Object labels
        scores: Confidence scores
        target_classes: List of classes to keep (None = all)
        min_confidence: Minimum confidence threshold
    
    Returns:
        Filtered boxes, labels, scores
    """
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= min_confidence:
            if target_classes is None or label in target_classes:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_scores.append(score)
    
    return np.array(filtered_boxes), filtered_labels, np.array(filtered_scores)
