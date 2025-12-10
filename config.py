"""
Configuration file for AI Vision Framework
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'weights')
CACHE_DIR = os.path.join(BASE_DIR, 'models', 'cache')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Supported Models Configuration
MODELS_CONFIG = {
    'image_classification': {
        'ResNet50': {
            'framework': 'pytorch',
            'source': 'torchvision',
            'input_size': (224, 224),
            'description': 'ResNet-50 pre-trained on ImageNet'
        },
        'ResNet101': {
            'framework': 'pytorch',
            'source': 'torchvision',
            'input_size': (224, 224),
            'description': 'ResNet-101 pre-trained on ImageNet'
        },
        'MobileNetV2': {
            'framework': 'pytorch',
            'source': 'torchvision',
            'input_size': (224, 224),
            'description': 'MobileNet V2 - Lightweight model'
        },
        'MobileNetV3': {
            'framework': 'pytorch',
            'source': 'torchvision',
            'input_size': (224, 224),
            'description': 'MobileNet V3 - Latest lightweight model'
        },
        'VGG16': {
            'framework': 'pytorch',
            'source': 'torchvision',
            'input_size': (224, 224),
            'description': 'VGG-16 pre-trained on ImageNet'
        },
        'EfficientNetB0': {
            'framework': 'pytorch',
            'source': 'timm',
            'input_size': (224, 224),
            'description': 'EfficientNet-B0 - Efficient architecture'
        },
        'InceptionV3': {
            'framework': 'tensorflow',
            'source': 'keras',
            'input_size': (299, 299),
            'description': 'Inception V3 from TensorFlow'
        }
    },
    'object_detection': {
        'YOLOv8n': {
            'framework': 'ultralytics',
            'source': 'ultralytics',
            'input_size': (640, 640),
            'description': 'YOLOv8 Nano - Fastest'
        },
        'YOLOv8s': {
            'framework': 'ultralytics',
            'source': 'ultralytics',
            'input_size': (640, 640),
            'description': 'YOLOv8 Small - Balanced'
        },
        'YOLOv8m': {
            'framework': 'ultralytics',
            'source': 'ultralytics',
            'input_size': (640, 640),
            'description': 'YOLOv8 Medium - High accuracy'
        },
        'YOLOv8l': {
            'framework': 'ultralytics',
            'source': 'ultralytics',
            'input_size': (640, 640),
            'description': 'YOLOv8 Large - Best accuracy'
        }
    },
    'face_recognition': {
        'MTCNN': {
            'framework': 'mtcnn',
            'source': 'mtcnn',
            'input_size': (160, 160),
            'description': 'MTCNN Face Detection'
        },
        'FaceRecognition': {
            'framework': 'face_recognition',
            'source': 'dlib',
            'input_size': None,
            'description': 'Face Recognition Library'
        }
    },
    'gesture_recognition': {
        'MediaPipe': {
            'framework': 'mediapipe',
            'source': 'mediapipe',
            'input_size': (224, 224),
            'description': 'MediaPipe Hand Tracking'
        }
    }
}

# ImageNet Classes (top 1000)
IMAGENET_CLASSES_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'

# COCO Classes for Object Detection
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Default settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45

# Supported file formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
