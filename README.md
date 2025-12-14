# AI Vision Framework

A comprehensive deep learning framework with GUI for running pre-trained models on various computer vision tasks.

## Features

### Supported Tasks
- **Image Classification** - Classify images into 1000 ImageNet categories
- **Object Detection** - Detect and localize objects in images/videos
- **Face Recognition** - Detect and recognize faces
- **Gesture Recognition** - Recognize hand gestures and poses

### Supported Models

#### Image Classification
- **ResNet** (50, 101) - Deep residual networks
- **MobileNet** (V2, V3) - Lightweight models for mobile
- **VGG16** - Classic deep architecture
- **EfficientNet-B0** - State-of-the-art efficient model
- **InceptionV3** - Google's inception architecture

#### Object Detection
- **YOLOv8** (Nano, Small, Medium, Large) - Real-time object detection
- Supports 80 COCO classes
- Adjustable confidence and IOU thresholds

#### Face Recognition
- **MTCNN** - Multi-task cascaded convolutional networks
- **Face Recognition** - Deep learning face recognition library
- Face detection, recognition, and landmark detection

#### Gesture Recognition
- **MediaPipe Hands** - Real-time hand tracking
- Recognizes basic gestures (fist, point, peace, open hand, etc.)
- 21 hand landmarks detection

### Supported Frameworks
- **PyTorch** - For ResNet, MobileNet, VGG, EfficientNet
- **TensorFlow/Keras** - For InceptionV3 and other Keras models
- **Ultralytics** - For YOLOv8 models
- **ONNX** - Cross-platform model format support

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Install Dependencies

```bash
# Clone or download the repository
cd ai_vision_framework

# Install required packages
pip install -r requirements.txt --break-system-packages
```

### Optional: MediaPipe for Gesture Recognition
```bash
pip install mediapipe --break-system-packages
```

## Quick Start

### Launch GUI Application

```bash
python gui/main_app.py
```

### Using the GUI

1. **Select Task**: Choose from Classification, Detection, Face Recognition, or Gesture Recognition
2. **Select Model**: Pick a pre-trained model from the dropdown
3. **Load Image**: Click "Load Image" to select an image file
4. **Run Inference**: Click "Run Inference" to process the image
5. **View Results**: See predictions and visualizations in the output panel

### Keyboard Shortcuts
- **Ctrl+O**: Open image
- **Q**: Quit webcam/video mode

## Programmatic Usage

### Image Classification Example

```python
from models.classification import ClassificationModelFactory
from utils.image_utils import ImagePreprocessor

# Create model
model = ClassificationModelFactory.create_model('ResNet50')
preprocessor = ImagePreprocessor()

# Preprocess image
image_tensor = preprocessor.preprocess_for_classification(
    'path/to/image.jpg',
    input_size=(224, 224),
    framework='pytorch'
)

# Run prediction
results = model.predict(image_tensor, top_k=5)

# Display results
for label, confidence in results:
    print(f"{label}: {confidence*100:.2f}%")
```

### Object Detection Example

```python
from models.detection import DetectionModelFactory

# Create YOLO detector
detector = DetectionModelFactory.create_model('YOLOv8n')

# Run detection
results = detector.detect('path/to/image.jpg', confidence=0.5)

print(f"Detected {results['count']} objects:")
for label, score in zip(results['labels'], results['scores']):
    print(f"  - {label}: {score*100:.2f}%")
```

### Face Recognition Example

```python
from models.face_recognition import FaceModelFactory

# Create face recognizer
recognizer = FaceModelFactory.create_model('FaceRecognition')

# Add known faces
recognizer.add_known_face('john.jpg', 'John Doe')
recognizer.add_known_face('jane.jpg', 'Jane Smith')

# Recognize faces in image
results = recognizer.recognize_faces('group_photo.jpg')

print(f"Found {results['count']} faces:")
for face in results['faces']:
    print(f"  - {face['name']} (confidence: {face['confidence']:.2f})")
```

### Gesture Recognition Example

```python
from models.gesture_recognition import GestureModelFactory

# Create gesture recognizer
recognizer = GestureModelFactory.create_model('MediaPipe')

# Detect hands in image
results = recognizer.detect_hands('hand_photo.jpg')

print(f"Detected {results['count']} hands:")
for hand in results['hands']:
    print(f"  - {hand['handedness']} hand: {hand['gesture']}")
```

### Video/Webcam Processing

```python
# Object detection on video
detector = DetectionModelFactory.create_model('YOLOv8s')

for frame_results in detector.detect_video('video.mp4', display=True):
    print(f"Frame: {frame_results['count']} objects detected")
    # Press 'q' to quit

# Gesture recognition on webcam
recognizer = GestureModelFactory.create_model('MediaPipe')

for frame_results in recognizer.detect_hands_video(0):  # 0 = webcam
    print(f"Hands detected: {frame_results['count']}")
    # Press 'q' to quit
```

##  Configuration

Edit `config/config.py` to customize:
- Model paths and cache directories
- Default confidence thresholds
- Supported file formats
- Class labels

## Model Performance

### Image Classification (ImageNet)
- ResNet50: Top-1: 76.2% | Top-5: 92.9%
- MobileNetV2: Top-1: 71.8% | Top-5: 90.3%
- EfficientNet-B0: Top-1: 77.1% | Top-5: 93.3%

### Object Detection (COCO)
- YOLOv8n: mAP: 37.3 | Speed: ~80 FPS
- YOLOv8s: mAP: 44.9 | Speed: ~60 FPS
- YOLOv8m: mAP: 50.2 | Speed: ~40 FPS
- YOLOv8l: mAP: 52.9 | Speed: ~30 FPS

##  GUI Features

- **Modern Dark Theme**: Easy on the eyes
- **Real-time Preview**: See results as they happen
- **Multiple Tasks**: Switch between different AI tasks seamlessly
- **Model Information**: View details about each model
- **Adjustable Settings**: Confidence thresholds, IOU, etc.
- **Export Results**: Save annotated images

##  Advanced Features

### Custom Model Integration
Add your own models by extending the factory classes:

```python
# In models/classification.py
class CustomClassifier:
    def __init__(self, model_name):
        # Load your model
        pass
    
    def predict(self, image_tensor, top_k=5):
        # Run inference
        pass
```

### Batch Processing
Process multiple images at once:

```python
import os
from pathlib import Path

image_dir = Path('images/')
detector = DetectionModelFactory.create_model('YOLOv8m')

for image_path in image_dir.glob('*.jpg'):
    results = detector.detect(str(image_path))
    print(f"{image_path.name}: {results['count']} objects")
```

### Custom Preprocessing
Extend the ImagePreprocessor class for custom transforms:

```python
from utils.image_utils import ImagePreprocessor

class CustomPreprocessor(ImagePreprocessor):
    def custom_transform(self, image):
        # Your custom preprocessing
        return transformed_image
```

##  System Requirements

### Minimum Requirements
- CPU: Intel i5 or equivalent
- RAM: 8 GB
- Storage: 5 GB for models
- OS: Windows 10/11, Ubuntu 20.04+, macOS 10.15+

### Recommended Requirements
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- GPU: NVIDIA RTX 2060 or better (6GB+ VRAM)
- RAM: 16 GB
- Storage: 10 GB SSD

##  Troubleshooting

### Common Issues

**Issue**: Module not found errors
```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt --break-system-packages
```

**Issue**: CUDA out of memory
```python
# Solution: Use smaller models or batch size
model = DetectionModelFactory.create_model('YOLOv8n')  # Use nano instead of large
```

**Issue**: Webcam not working
```bash
# Solution: Check camera permissions and drivers
# On Linux, ensure user is in video group
sudo usermod -a -G video $USER
```

##  Documentation

- Model Cards: See `docs/models/` for detailed model information
- API Reference: See `docs/api/` for API documentation
- Tutorials: See `docs/tutorials/` for step-by-step guides

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

##  License

This project is licensed under the MIT License.

##  Acknowledgments

- **PyTorch** - Deep learning framework
- **TensorFlow** - Machine learning platform
- **Ultralytics** - YOLOv8 implementation
- **MediaPipe** - Cross-platform ML solutions
- **OpenCV** - Computer vision library

##  Support

For issues and questions:
- GitHub Issues: Report bugs and request features
- Documentation: Check the docs/ folder
- Community: Join discussions

## Updates

### Version 1.0.0 (Current)
- Initial release
- Support for 4 major tasks
- 15+ pre-trained models
- Modern GUI interface
- Comprehensive documentation

---

**Made with ❤️ for the Computer Vision Community**
