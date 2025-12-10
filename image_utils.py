"""
Image Preprocessing Utilities
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class ImagePreprocessor:
    """Handles image preprocessing for different models"""
    
    def __init__(self):
        # Standard ImageNet normalization
        self.imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_for_classification(self, image_path, input_size=(224, 224), framework='pytorch'):
        """
        Preprocess image for classification models
        
        Args:
            image_path: Path to input image
            input_size: Target size (width, height)
            framework: 'pytorch', 'tensorflow', or 'onnx'
        
        Returns:
            Preprocessed image tensor/array
        """
        img = Image.open(image_path).convert('RGB')
        
        if framework == 'pytorch':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size[0]),
                transforms.ToTensor(),
                self.imagenet_normalize
            ])
            img_tensor = transform(img).unsqueeze(0)
            return img_tensor
        
        elif framework == 'tensorflow':
            img = img.resize(input_size)
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array.astype(np.float32)
        
        elif framework == 'onnx':
            img = img.resize(input_size)
            img_array = np.array(img).astype(np.float32)
            img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
            img_array = img_array / 255.0
            # Normalize
            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            img_array = (img_array - mean) / std
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
    
    def preprocess_for_detection(self, image_path, input_size=(640, 640)):
        """
        Preprocess image for object detection models (YOLO)
        
        Args:
            image_path: Path to input image
            input_size: Target size for detection
        
        Returns:
            Original image and preprocessed image
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize while maintaining aspect ratio
        h, w = img_rgb.shape[:2]
        scale = min(input_size[0] / w, input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # Create padded image
        padded_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        padded_img[:new_h, :new_w] = img_resized
        
        return img_rgb, padded_img
    
    def preprocess_for_face(self, image_path):
        """
        Preprocess image for face recognition
        
        Args:
            image_path: Path to input image
        
        Returns:
            Image array in RGB format
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    @staticmethod
    def load_image_cv2(image_path):
        """Load image using OpenCV"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    @staticmethod
    def load_image_pil(image_path):
        """Load image using PIL"""
        img = Image.open(image_path).convert('RGB')
        return img
    
    @staticmethod
    def resize_image(image, target_size):
        """Resize image to target size"""
        if isinstance(image, Image.Image):
            return image.resize(target_size)
        elif isinstance(image, np.ndarray):
            return cv2.resize(image, target_size)
        else:
            raise ValueError("Unsupported image type")


class VideoProcessor:
    """Handles video processing for real-time inference"""
    
    def __init__(self, source=0):
        """
        Initialize video processor
        
        Args:
            source: Video source (0 for webcam, or video file path)
        """
        self.source = source
        self.cap = None
    
    def open(self):
        """Open video stream"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")
        return self.cap
    
    def read_frame(self):
        """Read a frame from video stream"""
        if self.cap is None:
            raise ValueError("Video stream not opened")
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Release video stream"""
        if self.cap is not None:
            self.cap.release()
    
    def get_fps(self):
        """Get video FPS"""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0
    
    def get_frame_count(self):
        """Get total frame count"""
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0


def draw_bbox(image, boxes, labels=None, scores=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        labels: List of labels
        scores: List of confidence scores
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with bounding boxes
    """
    img_copy = image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Add label and score
        if labels is not None and i < len(labels):
            label_text = labels[i]
            if scores is not None and i < len(scores):
                label_text += f" {scores[i]:.2f}"
            
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(img_copy, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img_copy


def draw_landmarks(image, landmarks, connections=None, color=(0, 255, 0), radius=3):
    """
    Draw landmarks and connections on image
    
    Args:
        image: Input image
        landmarks: List of (x, y) coordinates
        connections: List of (start_idx, end_idx) tuples
        color: Drawing color (BGR)
        radius: Point radius
    
    Returns:
        Image with landmarks
    """
    img_copy = image.copy()
    
    # Draw connections
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = tuple(map(int, landmarks[start_idx]))
                end_point = tuple(map(int, landmarks[end_idx]))
                cv2.line(img_copy, start_point, end_point, color, 2)
    
    # Draw landmarks
    for x, y in landmarks:
        cv2.circle(img_copy, (int(x), int(y)), radius, color, -1)
    
    return img_copy
