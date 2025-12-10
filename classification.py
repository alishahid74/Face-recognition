"""
Image Classification Model Handlers
"""

import torch
import torchvision.models as models
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG16, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np
import json
import requests
from config import IMAGENET_CLASSES_URL, MODELS_CONFIG
import timm


class PyTorchClassifier:
    """PyTorch-based image classification models"""
    
    def __init__(self, model_name='ResNet50'):
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = None
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load pre-trained PyTorch model"""
        print(f"Loading {self.model_name} model...")
        
        if self.model_name == 'ResNet50':
            self.model = models.resnet50(pretrained=True)
        elif self.model_name == 'ResNet101':
            self.model = models.resnet101(pretrained=True)
        elif self.model_name == 'MobileNetV2':
            self.model = models.mobilenet_v2(pretrained=True)
        elif self.model_name == 'MobileNetV3':
            self.model = models.mobilenet_v3_large(pretrained=True)
        elif self.model_name == 'VGG16':
            self.model = models.vgg16(pretrained=True)
        elif self.model_name == 'EfficientNetB0':
            self.model = timm.create_model('efficientnet_b0', pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        print(f"{self.model_name} loaded successfully on {self.device}")
    
    def _load_labels(self):
        """Load ImageNet class labels"""
        try:
            response = requests.get(IMAGENET_CLASSES_URL)
            self.class_labels = response.json()
        except:
            # Fallback to basic labels
            self.class_labels = [f"Class_{i}" for i in range(1000)]
    
    def predict(self, image_tensor, top_k=5):
        """
        Make predictions on input image
        
        Args:
            image_tensor: Preprocessed image tensor
            top_k: Number of top predictions to return
        
        Returns:
            List of (label, confidence) tuples
        """
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            label = self.class_labels[idx.item()] if idx.item() < len(self.class_labels) else f"Class_{idx.item()}"
            results.append((label, prob.item()))
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        config = MODELS_CONFIG['image_classification'][self.model_name]
        return {
            'name': self.model_name,
            'framework': config['framework'],
            'input_size': config['input_size'],
            'description': config['description'],
            'device': str(self.device)
        }


class TensorFlowClassifier:
    """TensorFlow/Keras-based image classification models"""
    
    def __init__(self, model_name='InceptionV3'):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained TensorFlow model"""
        print(f"Loading {self.model_name} model...")
        
        if self.model_name == 'ResNet50':
            self.model = ResNet50(weights='imagenet')
            self.preprocess_fn = tf.keras.applications.resnet50.preprocess_input
        elif self.model_name == 'MobileNetV2':
            self.model = MobileNetV2(weights='imagenet')
            self.preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        elif self.model_name == 'VGG16':
            self.model = VGG16(weights='imagenet')
            self.preprocess_fn = tf.keras.applications.vgg16.preprocess_input
        elif self.model_name == 'InceptionV3':
            self.model = InceptionV3(weights='imagenet')
            self.preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        print(f"{self.model_name} loaded successfully")
    
    def predict(self, image_array, top_k=5):
        """
        Make predictions on input image
        
        Args:
            image_array: Preprocessed image array
            top_k: Number of top predictions to return
        
        Returns:
            List of (label, confidence) tuples
        """
        # Preprocess
        preprocessed = self.preprocess_fn(image_array * 255.0)
        
        # Predict
        predictions = self.model.predict(preprocessed, verbose=0)
        
        # Decode predictions
        decoded = decode_predictions(predictions, top=top_k)[0]
        
        results = [(label, float(score)) for (_, label, score) in decoded]
        return results
    
    def get_model_info(self):
        """Get model information"""
        config = MODELS_CONFIG['image_classification'][self.model_name]
        return {
            'name': self.model_name,
            'framework': config['framework'],
            'input_size': config['input_size'],
            'description': config['description']
        }


class ClassificationModelFactory:
    """Factory class to create classification models"""
    
    @staticmethod
    def create_model(model_name):
        """
        Create and return a classification model
        
        Args:
            model_name: Name of the model to create
        
        Returns:
            Model instance
        """
        if model_name not in MODELS_CONFIG['image_classification']:
            raise ValueError(f"Unsupported model: {model_name}")
        
        config = MODELS_CONFIG['image_classification'][model_name]
        framework = config['framework']
        
        if framework == 'pytorch' or framework == 'timm':
            return PyTorchClassifier(model_name)
        elif framework == 'tensorflow':
            return TensorFlowClassifier(model_name)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    @staticmethod
    def get_available_models():
        """Get list of available classification models"""
        return list(MODELS_CONFIG['image_classification'].keys())
    
    @staticmethod
    def get_model_info(model_name):
        """Get information about a specific model"""
        if model_name in MODELS_CONFIG['image_classification']:
            return MODELS_CONFIG['image_classification'][model_name]
        return None
