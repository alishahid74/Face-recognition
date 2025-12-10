"""
Advanced Model Support for AI Vision Framework
Includes: CLIP, SAM, DETR, OpenPose, Advanced YOLO variants
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional
import requests
from io import BytesIO

# Try importing advanced models
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("DETR not available. Install with: pip install transformers")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class CLIPModel:
    """CLIP model for zero-shot image classification and image-text similarity"""
    
    def __init__(self, model_name='ViT-B/32'):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not installed")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"CLIP loaded on {self.device}")
    
    def classify_image(self, image_path: str, classes: List[str]) -> List[Tuple[str, float]]:
        """
        Zero-shot classification with custom classes
        
        Args:
            image_path: Path to image
            classes: List of class names to classify against
        
        Returns:
            List of (class_name, probability) tuples
        """
        # Load and preprocess image
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        # Prepare text
        text = clip.tokenize([f"a photo of a {c}" for c in classes]).to(self.device)
        
        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get results
        values, indices = similarity[0].topk(len(classes))
        results = [(classes[idx], val.item()) for idx, val in zip(indices, values)]
        
        return results
    
    def get_image_features(self, image_path: str) -> np.ndarray:
        """Extract CLIP image features for similarity search"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    
    def find_similar_text(self, image_path: str, text_queries: List[str]) -> List[Tuple[str, float]]:
        """Find most similar text descriptions for an image"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(text_queries).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T)[0]
        
        results = [(text_queries[i], similarity[i].item()) for i in range(len(text_queries))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


class DETRDetector:
    """DETR (Detection Transformer) for object detection"""
    
    def __init__(self, model_name='facebook/detr-resnet-50'):
        if not DETR_AVAILABLE:
            raise ImportError("transformers not installed")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading DETR model: {model_name}")
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"DETR loaded on {self.device}")
    
    def detect(self, image_path: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Detect objects using DETR
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence score
        
        Returns:
            Dictionary with detection results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]
        
        # Extract results
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        # Get label names
        label_names = [self.model.config.id2label[label.item()] for label in results["labels"]]
        
        # Draw results
        img_array = np.array(image)
        annotated_image = self._draw_boxes(img_array, boxes, label_names, scores)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': label_names,
            'label_ids': labels,
            'count': len(boxes),
            'image': annotated_image
        }
    
    def _draw_boxes(self, image, boxes, labels, scores):
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            text = f"{label}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(img_copy, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_copy


class HolisticPoseEstimator:
    """MediaPipe Holistic for full-body pose, face, and hand tracking"""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed")
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True
        )
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process image for full-body pose estimation
        
        Returns:
            Dictionary with pose, face, and hand landmarks
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.holistic.process(image_rgb)
        
        # Draw landmarks
        annotated_image = image.copy()
        
        # Draw face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return {
            'pose_landmarks': results.pose_landmarks,
            'face_landmarks': results.face_landmarks,
            'left_hand_landmarks': results.left_hand_landmarks,
            'right_hand_landmarks': results.right_hand_landmarks,
            'segmentation_mask': results.segmentation_mask,
            'image': annotated_image
        }
    
    def process_video(self, video_source: int = 0):
        """Process video stream for real-time pose estimation"""
        cap = cv2.VideoCapture(video_source)
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(frame_rgb)
                
                # Draw landmarks (similar to process_image)
                annotated_frame = frame.copy()
                
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_holistic.POSE_CONNECTIONS
                    )
                
                cv2.imshow('Holistic Pose Estimation', annotated_frame)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
                yield {
                    'pose_landmarks': results.pose_landmarks,
                    'face_landmarks': results.face_landmarks,
                    'left_hand_landmarks': results.left_hand_landmarks,
                    'right_hand_landmarks': results.right_hand_landmarks,
                    'frame': annotated_frame
                }
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def close(self):
        """Close the holistic model"""
        self.holistic.close()


class AdvancedModelFactory:
    """Factory for advanced models"""
    
    @staticmethod
    def create_clip_model(model_name='ViT-B/32'):
        """Create CLIP model"""
        return CLIPModel(model_name)
    
    @staticmethod
    def create_detr_detector(model_name='facebook/detr-resnet-50'):
        """Create DETR detector"""
        return DETRDetector(model_name)
    
    @staticmethod
    def create_holistic_estimator():
        """Create holistic pose estimator"""
        return HolisticPoseEstimator()
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available advanced models"""
        models = {
            'zero_shot': [],
            'detection': [],
            'pose': []
        }
        
        if CLIP_AVAILABLE:
            models['zero_shot'].extend(['CLIP-ViT-B/32', 'CLIP-ViT-L/14'])
        
        if DETR_AVAILABLE:
            models['detection'].extend(['DETR-ResNet-50', 'DETR-ResNet-101'])
        
        if MEDIAPIPE_AVAILABLE:
            models['pose'].extend(['MediaPipe-Holistic'])
        
        return models


# Utility functions for advanced features

def zero_shot_detection(image_path: str, object_queries: List[str]) -> Dict:
    """
    Zero-shot object detection using CLIP
    
    Args:
        image_path: Path to image
        object_queries: List of objects to detect
    
    Returns:
        Dictionary with detection results
    """
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP not available")
    
    clip_model = CLIPModel()
    results = clip_model.classify_image(image_path, object_queries)
    
    return {
        'predictions': results,
        'top_prediction': results[0] if results else None
    }


def compare_detection_models(image_path: str, models: List[str]) -> Dict:
    """
    Compare multiple detection models on the same image
    
    Args:
        image_path: Path to image
        models: List of model names
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for model_name in models:
        if 'detr' in model_name.lower():
            detector = DETRDetector()
            results[model_name] = detector.detect(image_path)
        # Add more model comparisons here
    
    return results
