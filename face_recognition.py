"""
Face Recognition Model Handlers
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import face_recognition
from config import MODELS_CONFIG



class MTCNNFaceDetector:
    """MTCNN-based face detection"""
    
    def __init__(self):
        self.model_name = 'MTCNN'
        self.detector = None
        self._load_model()
    
    def _load_model(self):
        """Load MTCNN face detector"""
        print("Loading MTCNN face detector...")
        try:
            self.detector = MTCNN()
            print("MTCNN loaded successfully")
        except Exception as e:
            print(f"Error loading MTCNN: {e}")
            raise
    
    def detect_faces(self, image_path):
        """
        Detect faces in image
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary containing face information and annotated image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(image_rgb)
        
        # Draw bounding boxes and landmarks
        annotated_image = image.copy()
        
        face_data = []
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(annotated_image, f"{confidence:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw facial landmarks
            keypoints = face['keypoints']
            for key, point in keypoints.items():
                cv2.circle(annotated_image, point, 3, (0, 0, 255), -1)
            
            face_data.append({
                'box': [x, y, x+w, y+h],
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        return {
            'faces': face_data,
            'count': len(faces),
            'image': annotated_image
        }
    
    def get_model_info(self):
        """Get model information"""
        config = MODELS_CONFIG['face_recognition'][self.model_name]
        return {
            'name': self.model_name,
            'framework': config['framework'],
            'description': config['description']
        }


class FaceRecognizer:
    """Face recognition using face_recognition library"""
    
    def __init__(self):
        self.model_name = 'FaceRecognition'
        self.known_faces = {}  # Store known face encodings
    
    def add_known_face(self, image_path, name):
        """
        Add a known face to the database
        
        Args:
            image_path: Path to face image
            name: Name/ID of the person
        
        Returns:
            Success status
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                return False, "No face detected in image"
            
            self.known_faces[name] = encodings[0]
            return True, f"Face added for {name}"
        
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def recognize_faces(self, image_path, tolerance=0.6):
        """
        Recognize faces in image
        
        Args:
            image_path: Path to input image
            tolerance: Face matching tolerance (lower = more strict)
        
        Returns:
            Dictionary containing recognition results and annotated image
        """
        # Load image
        image = face_recognition.load_image_file(image_path)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Recognize faces
        annotated_image = image_bgr.copy()
        recognized_faces = []
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                list(self.known_faces.values()), encoding, tolerance=tolerance
            )
            
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                face_distances = face_recognition.face_distance(
                    list(self.known_faces.values()), encoding
                )
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    name = list(self.known_faces.keys())[best_match_idx]
                    confidence = 1 - face_distances[best_match_idx]
            
            # Draw box and label
            cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(annotated_image, (left, bottom - 35), (right, bottom), (0, 255, 0), -1)
            cv2.putText(annotated_image, f"{name} ({confidence:.2f})", 
                       (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            recognized_faces.append({
                'name': name,
                'confidence': float(confidence),
                'box': [left, top, right, bottom]
            })
        
        return {
            'faces': recognized_faces,
            'count': len(face_locations),
            'image': annotated_image,
            'known_faces_count': len(self.known_faces)
        }
    
    def remove_known_face(self, name):
        """Remove a known face from database"""
        if name in self.known_faces:
            del self.known_faces[name]
            return True
        return False
    
    def get_known_faces(self):
        """Get list of known face names"""
        return list(self.known_faces.keys())
    
    def clear_known_faces(self):
        """Clear all known faces"""
        self.known_faces.clear()
    
    def get_model_info(self):
        """Get model information"""
        config = MODELS_CONFIG['face_recognition'][self.model_name]
        return {
            'name': self.model_name,
            'framework': config['framework'],
            'description': config['description'],
            'known_faces': len(self.known_faces)
        }


class FaceModelFactory:
    """Factory class to create face recognition models"""
    
    @staticmethod
    def create_model(model_name='MTCNN'):
        """
        Create and return a face recognition model
        
        Args:
            model_name: Name of the model to create
        
        Returns:
            Model instance
        """
        if model_name not in MODELS_CONFIG['face_recognition']:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if model_name == 'MTCNN':
            return MTCNNFaceDetector()
        elif model_name == 'FaceRecognition':
            return FaceRecognizer()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    @staticmethod
    def get_available_models():
        """Get list of available face recognition models"""
        return list(MODELS_CONFIG['face_recognition'].keys())
