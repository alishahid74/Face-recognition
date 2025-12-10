# Project Structure Documentation

## Overview

The AI Vision Framework is organized into a modular, scalable architecture that separates concerns and makes it easy to extend with new models and features.

## Directory Structure

```
ai_vision_framework/
│
├── config/                      # Configuration files
│   ├── __init__.py
│   └── config.py               # Model configurations, paths, and settings
│
├── models/                      # Model implementations
│   ├── __init__.py
│   ├── classification.py       # Image classification models
│   ├── detection.py            # Object detection models
│   ├── face_recognition.py     # Face recognition models
│   ├── gesture_recognition.py  # Gesture recognition models
│   ├── weights/                # Model weights storage
│   └── cache/                  # Model cache directory
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── image_utils.py          # Image preprocessing and visualization
│
├── gui/                         # Graphical User Interface
│   ├── __init__.py
│   └── main_app.py             # Main GUI application
│
├── examples/                    # Example scripts
│   ├── classification_example.py
│   ├── detection_example.py
│   ├── face_recognition_example.py
│   └── gesture_example.py
│
├── docs/                        # Documentation
│   ├── models/                 # Model-specific documentation
│   ├── api/                    # API reference
│   └── tutorials/              # Step-by-step tutorials
│
├── output/                      # Output directory for results
│
├── __init__.py                  # Package initialization
├── requirements.txt             # Python dependencies
├── run.py                       # Main launcher script
├── setup.sh                     # Setup script
└── README.md                    # Main documentation
```

## Core Components

### 1. Configuration (`config/`)

**config.py**
- Model configurations and metadata
- File paths and directories
- Default settings and thresholds
- Supported model lists
- ImageNet and COCO class definitions

Key configurations:
```python
MODELS_CONFIG = {
    'image_classification': {...},
    'object_detection': {...},
    'face_recognition': {...},
    'gesture_recognition': {...}
}
```

### 2. Models (`models/`)

#### classification.py
Handles image classification models:
- **PyTorchClassifier**: ResNet, MobileNet, VGG, EfficientNet
- **TensorFlowClassifier**: InceptionV3, Keras models
- **ClassificationModelFactory**: Factory pattern for model creation

Key methods:
- `_load_model()`: Load pre-trained weights
- `predict()`: Run inference on images
- `get_model_info()`: Retrieve model metadata

#### detection.py
Handles object detection models:
- **YOLODetector**: YOLOv8 variants (n, s, m, l)
- **DetectionModelFactory**: Factory for detector creation
- Helper functions: `count_objects()`, `filter_detections()`

Key methods:
- `detect()`: Detect objects in images
- `detect_video()`: Real-time video detection
- `set_confidence_threshold()`: Adjust detection sensitivity

#### face_recognition.py
Handles face detection and recognition:
- **MTCNNFaceDetector**: Face detection with landmarks
- **FaceRecognizer**: Face recognition with database
- **FaceModelFactory**: Factory for face models

Key methods:
- `detect_faces()`: Detect faces and landmarks
- `add_known_face()`: Add face to recognition database
- `recognize_faces()`: Recognize faces in images

#### gesture_recognition.py
Handles hand tracking and gesture recognition:
- **MediaPipeGestureRecognizer**: Hand landmark detection
- **GestureModelFactory**: Factory for gesture models

Key methods:
- `detect_hands()`: Detect hands and landmarks
- `detect_hands_video()`: Real-time hand tracking
- `_recognize_gesture()`: Classify hand gestures

### 3. Utilities (`utils/`)

#### image_utils.py
Image processing utilities:
- **ImagePreprocessor**: Image preprocessing for different frameworks
- **VideoProcessor**: Video stream handling
- **Visualization**: Drawing bounding boxes and landmarks

Key functions:
- `preprocess_for_classification()`: Prepare images for classification
- `preprocess_for_detection()`: Prepare images for detection
- `draw_bbox()`: Draw detection results
- `draw_landmarks()`: Draw facial/hand landmarks

### 4. GUI (`gui/`)

#### main_app.py
Main graphical interface:
- **AIVisionGUI**: Main application class
- Task selection (classification, detection, face, gesture)
- Model selection and configuration
- Image/video loading and display
- Real-time inference
- Results visualization

Components:
- Control panel: Task and model selection
- Display panel: Image/video display
- Results panel: Inference results
- Settings: Confidence thresholds, parameters

### 5. Examples (`examples/`)

Demonstration scripts:
- **classification_example.py**: Image classification demos
- **detection_example.py**: Object detection demos
- Benchmark and comparison utilities
- Command-line interfaces

## Design Patterns

### Factory Pattern
Used for model creation to abstract instantiation:
```python
model = ClassificationModelFactory.create_model('ResNet50')
detector = DetectionModelFactory.create_model('YOLOv8n')
```

### Strategy Pattern
Different preprocessing strategies for different frameworks:
```python
preprocessor.preprocess_for_classification(image, framework='pytorch')
preprocessor.preprocess_for_classification(image, framework='tensorflow')
```

### Singleton Pattern (Implicit)
GUI application maintains single model instance per task to avoid reloading.

## Data Flow

### Image Classification Flow
```
User Input (Image)
    ↓
ImagePreprocessor
    ↓
Model-specific preprocessing (PyTorch/TensorFlow)
    ↓
Model inference
    ↓
Post-processing (softmax, top-k)
    ↓
Results display
```

### Object Detection Flow
```
User Input (Image/Video)
    ↓
YOLODetector
    ↓
Image resizing and padding
    ↓
YOLO inference
    ↓
NMS filtering (confidence + IOU)
    ↓
Bounding box visualization
    ↓
Results display
```

### Face Recognition Flow
```
User Input (Image)
    ↓
Face Detector (MTCNN/face_recognition)
    ↓
Face detection
    ↓
Feature extraction
    ↓
Database comparison (if using recognition)
    ↓
Results with landmarks/identities
```

## Extension Points

### Adding New Models

1. **Classification Model**:
```python
# In models/classification.py
class NewClassifier:
    def __init__(self, model_name):
        # Load model
        pass
    
    def predict(self, image_tensor, top_k=5):
        # Run inference
        pass
```

2. **Update Configuration**:
```python
# In config/config.py
MODELS_CONFIG['image_classification']['NewModel'] = {
    'framework': 'pytorch',
    'source': 'custom',
    'input_size': (224, 224),
    'description': 'Custom model description'
}
```

3. **Update Factory**:
```python
# In models/classification.py
@staticmethod
def create_model(model_name):
    if model_name == 'NewModel':
        return NewClassifier(model_name)
```

### Adding New Tasks

1. Create new model handler file (e.g., `models/segmentation.py`)
2. Add configuration to `config/config.py`
3. Create factory class
4. Update GUI to include new task option
5. Add example script to `examples/`

### Custom Preprocessing

```python
# Extend ImagePreprocessor
class CustomPreprocessor(ImagePreprocessor):
    def custom_preprocess(self, image):
        # Your custom preprocessing
        return processed_image
```

## Testing

Recommended testing structure:
```
tests/
├── test_classification.py
├── test_detection.py
├── test_face_recognition.py
├── test_gesture_recognition.py
└── test_utils.py
```

## Performance Optimization

### Model Loading
- Models are loaded once and cached
- Lazy loading: Models loaded only when needed
- GPU acceleration when available

### Inference Optimization
- Batch processing for multiple images
- Video stream buffering
- Asynchronous processing for GUI

### Memory Management
- Proper cleanup of OpenCV resources
- PyTorch no_grad() for inference
- Image resizing to reduce memory footprint

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **TensorFlow**: Alternative framework
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical operations
- **Pillow**: Image handling
- **Ultralytics**: YOLOv8 implementation

### Optional Dependencies
- **MediaPipe**: Gesture recognition
- **face_recognition**: Face recognition
- **MTCNN**: Face detection
- **timm**: PyTorch image models

## Configuration Files

### Model Weights
Stored in `models/weights/`:
- Downloaded automatically on first use
- Cached for subsequent runs
- Can be pre-downloaded for offline use

### Cache Directory
`models/cache/`:
- Temporary files
- Preprocessed data
- Model metadata

## Best Practices

1. **Always use factory methods** for model creation
2. **Close resources properly** (video streams, model instances)
3. **Handle exceptions** gracefully
4. **Use logging** for debugging (future enhancement)
5. **Validate inputs** before processing
6. **Document new features** thoroughly

## Future Enhancements

Planned features:
- [ ] Semantic segmentation models
- [ ] Pose estimation
- [ ] Video processing optimization
- [ ] Batch processing GUI
- [ ] Model fine-tuning interface
- [ ] Export to ONNX
- [ ] Multi-GPU support
- [ ] REST API server
- [ ] Docker containerization
- [ ] Cloud deployment options

## Troubleshooting

### Common Issues

**Issue**: Model fails to load
- Check internet connection (models download on first use)
- Verify disk space in `models/weights/`
- Check CUDA/GPU drivers if using GPU

**Issue**: GUI doesn't start
- Verify all dependencies installed
- Check Python version (3.8+)
- Try running directly: `python gui/main_app.py`

**Issue**: Low FPS on video
- Use smaller model variants (YOLOv8n vs YOLOv8l)
- Reduce input resolution
- Enable GPU acceleration

## Contributing

When contributing:
1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints where possible
5. Follow PEP 8 style guide

## License

MIT License - See LICENSE file for details.
