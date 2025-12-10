"""
Example: Image Classification

This script demonstrates how to use the image classification models
to classify images into ImageNet categories.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classification import ClassificationModelFactory
from utils.image_utils import ImagePreprocessor
from config.config import MODELS_CONFIG


def classify_image(image_path, model_name='ResNet50', top_k=5):
    """
    Classify an image using a pre-trained model
    
    Args:
        image_path: Path to the image file
        model_name: Name of the model to use
        top_k: Number of top predictions to return
    """
    print(f"\n{'='*60}")
    print(f"Image Classification Example")
    print(f"{'='*60}\n")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Create model
    print(f"Loading {model_name} model...")
    model = ClassificationModelFactory.create_model(model_name)
    print(f"✓ Model loaded successfully\n")
    
    # Get model info
    model_info = model.get_model_info()
    print("Model Information:")
    print(f"  Name: {model_info['name']}")
    print(f"  Framework: {model_info['framework']}")
    print(f"  Input Size: {model_info['input_size']}")
    print(f"  Device: {model_info['device']}")
    print(f"  Description: {model_info['description']}\n")
    
    # Preprocess image
    print("Preprocessing image...")
    preprocessor = ImagePreprocessor()
    config = MODELS_CONFIG['image_classification'][model_name]
    
    image_tensor = preprocessor.preprocess_for_classification(
        image_path,
        input_size=config['input_size'],
        framework=config['framework']
    )
    print(f"✓ Image preprocessed\n")
    
    # Run prediction
    print("Running inference...")
    results = model.predict(image_tensor, top_k=top_k)
    print(f"✓ Inference complete\n")
    
    # Display results
    print(f"Top {top_k} Predictions:")
    print(f"{'-'*60}")
    for i, (label, confidence) in enumerate(results, 1):
        bar_length = int(confidence * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"{i}. {label:30s} {bar} {confidence*100:5.2f}%")
    print(f"{'-'*60}\n")


def compare_models(image_path, models=['ResNet50', 'MobileNetV2', 'VGG16'], top_k=3):
    """
    Compare predictions from multiple models
    
    Args:
        image_path: Path to the image file
        models: List of model names to compare
        top_k: Number of top predictions to return
    """
    print(f"\n{'='*60}")
    print(f"Model Comparison")
    print(f"{'='*60}\n")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    preprocessor = ImagePreprocessor()
    
    for model_name in models:
        print(f"\n{model_name}:")
        print(f"{'-'*60}")
        
        try:
            # Load model
            model = ClassificationModelFactory.create_model(model_name)
            
            # Preprocess
            config = MODELS_CONFIG['image_classification'][model_name]
            image_tensor = preprocessor.preprocess_for_classification(
                image_path,
                input_size=config['input_size'],
                framework=config['framework']
            )
            
            # Predict
            results = model.predict(image_tensor, top_k=top_k)
            
            # Display
            for i, (label, confidence) in enumerate(results, 1):
                print(f"  {i}. {label:25s} {confidence*100:5.2f}%")
        
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    print(f"\n{'='*60}\n")


def list_available_models():
    """List all available classification models"""
    print(f"\n{'='*60}")
    print(f"Available Classification Models")
    print(f"{'='*60}\n")
    
    models = ClassificationModelFactory.get_available_models()
    
    for model_name in models:
        info = ClassificationModelFactory.get_model_info(model_name)
        print(f"• {model_name}")
        print(f"  Framework: {info['framework']}")
        print(f"  Input Size: {info['input_size']}")
        print(f"  {info['description']}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Classification Example')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='ResNet50', 
                       help='Model name (default: ResNet50)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions (default: 5)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
    elif args.compare and args.image:
        compare_models(args.image)
    elif args.image:
        classify_image(args.image, args.model, args.top_k)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python classification_example.py --image cat.jpg")
        print("  python classification_example.py --image dog.jpg --model MobileNetV2")
        print("  python classification_example.py --image bird.jpg --top-k 10")
        print("  python classification_example.py --list-models")
        print("  python classification_example.py --image cat.jpg --compare")
