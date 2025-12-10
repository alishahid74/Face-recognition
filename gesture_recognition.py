"""
Gesture Recognition Model Handlers
"""

import cv2
import numpy as np
from config import MODELS_CONFIG

# MediaPipe is optional - will be loaded only if available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe")


class MediaPipeGestureRecognizer:
    """MediaPipe-based hand tracking and gesture recognition"""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not installed")
        
        self.model_name = 'MediaPipe'
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = None
        self._load_model()
    
    def _load_model(self):
        """Initialize MediaPipe Hands"""
        print("Loading MediaPipe Hands...")
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Hands loaded successfully")
        except Exception as e:
            print(f"Error loading MediaPipe: {e}")
            raise
    
    def detect_hands(self, image_path):
        """
        Detect hands and landmarks in image
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary containing hand information and annotated image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        # Draw landmarks
        annotated_image = image.copy()
        hands_data = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
                
                # Detect handedness
                handedness = results.multi_handedness[idx].classification[0].label
                confidence = results.multi_handedness[idx].classification[0].score
                
                # Recognize gesture
                gesture = self._recognize_gesture(hand_landmarks)
                
                # Add label
                cv2.putText(annotated_image, f"{handedness} - {gesture}", 
                           landmarks[0], cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                hands_data.append({
                    'handedness': handedness,
                    'confidence': float(confidence),
                    'landmarks': landmarks,
                    'gesture': gesture
                })
        
        return {
            'hands': hands_data,
            'count': len(hands_data),
            'image': annotated_image
        }
    
    def detect_hands_video(self, video_source=0):
        """
        Detect hands in video stream
        
        Args:
            video_source: Video source (0 for webcam, or video file path)
        
        Yields:
            Hand detection results for each frame
        """
        cap = cv2.VideoCapture(video_source)
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                # Draw landmarks
                annotated_frame = frame.copy()
                hands_data = []
                
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Extract landmarks
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            h, w, _ = frame.shape
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            landmarks.append((x, y))
                        
                        # Handedness
                        handedness = results.multi_handedness[idx].classification[0].label
                        confidence = results.multi_handedness[idx].classification[0].score
                        
                        # Gesture
                        gesture = self._recognize_gesture(hand_landmarks)
                        
                        # Label
                        cv2.putText(annotated_frame, f"{handedness} - {gesture}", 
                                   landmarks[0], cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 255, 0), 2)
                        
                        hands_data.append({
                            'handedness': handedness,
                            'confidence': float(confidence),
                            'landmarks': landmarks,
                            'gesture': gesture
                        })
                
                cv2.imshow('Hand Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                yield {
                    'hands': hands_data,
                    'count': len(hands_data),
                    'frame': annotated_frame
                }
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _recognize_gesture(self, hand_landmarks):
        """
        Recognize basic hand gestures
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
        
        Returns:
            Gesture name
        """
        # Get landmark positions
        landmarks = hand_landmarks.landmark
        
        # Finger tip and base indices
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        # Count extended fingers
        extended_fingers = 0
        
        # Thumb (check x coordinate)
        if abs(thumb_tip.x - thumb_mcp.x) > 0.05:
            extended_fingers += 1
        
        # Other fingers (check y coordinate)
        if index_tip.y < index_mcp.y:
            extended_fingers += 1
        if middle_tip.y < middle_mcp.y:
            extended_fingers += 1
        if ring_tip.y < ring_mcp.y:
            extended_fingers += 1
        if pinky_tip.y < pinky_mcp.y:
            extended_fingers += 1
        
        # Recognize gestures based on extended fingers
        if extended_fingers == 0:
            return "Fist"
        elif extended_fingers == 1:
            return "Point"
        elif extended_fingers == 2:
            return "Peace"
        elif extended_fingers == 3:
            return "Three"
        elif extended_fingers == 4:
            return "Four"
        elif extended_fingers == 5:
            return "Open Hand"
        else:
            return "Unknown"
    
    def close(self):
        """Close MediaPipe hands"""
        if self.hands:
            self.hands.close()
    
    def get_model_info(self):
        """Get model information"""
        config = MODELS_CONFIG['gesture_recognition'][self.model_name]
        return {
            'name': self.model_name,
            'framework': config['framework'],
            'description': config['description']
        }


class GestureModelFactory:
    """Factory class to create gesture recognition models"""
    
    @staticmethod
    def create_model(model_name='MediaPipe'):
        """
        Create and return a gesture recognition model
        
        Args:
            model_name: Name of the model to create
        
        Returns:
            Model instance
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for gesture recognition")
        
        if model_name not in MODELS_CONFIG['gesture_recognition']:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if model_name == 'MediaPipe':
            return MediaPipeGestureRecognizer()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    @staticmethod
    def get_available_models():
        """Get list of available gesture recognition models"""
        if MEDIAPIPE_AVAILABLE:
            return list(MODELS_CONFIG['gesture_recognition'].keys())
        return []
