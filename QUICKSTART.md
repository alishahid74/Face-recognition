# Quick Start Guide

Get up and running with AI Vision Framework in 5 minutes!

## 1. Installation (2 minutes)

### Using Setup Script (Recommended)
```bash
cd ai_vision_framework
chmod +x setup.sh
./setup.sh
```

### Manual Installation
```bash
cd ai_vision_framework

# Install core dependencies
pip install -r requirements.txt --break-system-packages

# Install optional dependencies (for all features)
pip install mediapipe face-recognition mtcnn --break-system-packages
```

## 2. Launch Application (30 seconds)

### GUI Application
```bash
python run.py
```

Or directly:
```bash
python gui/main_app.py
```

## 3. First Run - Image Classification (2 minutes)

1. **Launch the GUI** (see step 2)

2. **Select Task**: Choose "Image Classification" (default)

3. **Select Model**: Choose "ResNet50" from dropdown

4. **Load Image**: Click "📁 Load Image" button
   - Select any image (jpg, png, etc.)
   - Image will display in the output panel

5. **Run Inference**: Click "▶️ Run Inference"
   - Wait a few seconds for model to load (first time only)
   - View top 5 predictions in the results panel

### Example Output
```
Top 5 Predictions:
1. golden_retriever    ████████████████████████ 87.34%
2. labrador_retriever  ████████████░░░░░░░░░░░░ 8.12%
3. dog                 ████░░░░░░░░░░░░░░░░░░░░ 2.45%
4. puppy               ██░░░░░░░░░░░░░░░░░░░░░░ 1.23%
5. pet                 █░░░░░░░░░░░░░░░░░░░░░░░ 0.86%
```

## 4. Try Object Detection (1 minute)

1. **Select Task**: Choose "Object Detection"

2. **Select Model**: Choose "YOLOv8n" (fastest)

3. **Load Image**: Load an image with objects

4. **Adjust Settings** (optional):
   - Confidence threshold: 0.5 (default)
   - Drag slider to adjust

5. **Run Inference**: Click "▶️ Run Inference"
   - View detected objects with bounding boxes
   - See object counts in results panel

### Example Output
```
Detection Results:
----------------------------------------------------------
Total objects detected: 5

Object counts:
  • person: 2
  • car: 2
  • traffic light: 1

Detailed detections:
  1. person         (92.45%) - Box: [120, 150, 280, 450]
  2. person         (88.12%) - Box: [320, 140, 450, 460]
  3. car            (85.67%) - Box: [50, 200, 200, 300]
  ...
```

## 5. Command Line Examples

### Image Classification
```bash
cd examples

# Basic classification
python classification_example.py --image ../sample_images/cat.jpg

# Use different model
python classification_example.py --image dog.jpg --model MobileNetV2

# Show top 10 predictions
python classification_example.py --image bird.jpg --top-k 10

# List available models
python classification_example.py --list-models

# Compare multiple models
python classification_example.py --image cat.jpg --compare
```

### Object Detection
```bash
cd examples

# Basic detection
python detection_example.py --image ../sample_images/street.jpg

# Use different model and confidence
python detection_example.py --image people.jpg --model YOLOv8m --confidence 0.6

# Detect on webcam
python detection_example.py --video 0

# Detect specific classes only
python detection_example.py --image street.jpg --classes person car

# Benchmark models
python detection_example.py --image test.jpg --benchmark
```

## 6. Tips for Best Results

### Image Classification
- ✓ Use clear, well-lit images
- ✓ Center the object in the frame
- ✓ Avoid heavy filters or distortions
- ✓ Use ResNet50 or EfficientNet for best accuracy
- ✓ Use MobileNet for speed

### Object Detection
- ✓ Higher confidence = fewer false positives
- ✓ Lower confidence = catch more objects
- ✓ YOLOv8n for real-time/webcam (fastest)
- ✓ YOLOv8l for best accuracy (slowest)
- ✓ Adjust IOU threshold to reduce overlapping boxes

### Face Recognition
- ✓ Good lighting is crucial
- ✓ Face should be clearly visible
- ✓ Add multiple photos per person for better recognition
- ✓ Use frontal face images for best results

### Gesture Recognition
- ✓ Ensure hands are clearly visible
- ✓ Use good lighting
- ✓ Avoid cluttered backgrounds
- ✓ Works best with open, spread fingers

## 7. Keyboard Shortcuts

When using webcam/video mode:
- **Q**: Quit video mode
- **ESC**: Exit application

## 8. Common First-Time Issues

### Issue: "Module not found"
```bash
# Solution: Install dependencies
pip install -r requirements.txt --break-system-packages
```

### Issue: "CUDA out of memory"
```python
# Solution: Use smaller models
# GUI: Select YOLOv8n instead of YOLOv8l
# Or add to code:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

### Issue: Models downloading slowly
```bash
# Models auto-download on first use
# Be patient - they're large files
# ResNet50: ~100MB
# YOLOv8n: ~6MB
# YOLOv8l: ~87MB
```

### Issue: Webcam not working
```bash
# Linux: Check permissions
sudo usermod -a -G video $USER
# Then logout and login

# Windows: Check camera privacy settings
# macOS: Grant camera permission in System Preferences
```

## 9. Next Steps

### Try More Features
1. **Face Recognition**: Add known faces and recognize them
2. **Gesture Recognition**: Use webcam for real-time hand tracking
3. **Video Processing**: Process video files
4. **Batch Processing**: Use example scripts for multiple files

### Learn More
- Read full documentation: `README.md`
- Check project structure: `STRUCTURE.md`
- Review example scripts: `examples/`

### Customize
- Edit settings: `config/config.py`
- Adjust thresholds in GUI
- Try different model variants
- Combine multiple tasks

## 10. Getting Help

### Documentation
- Main README: Comprehensive feature guide
- Structure doc: Architecture and design
- Example scripts: Commented code samples

### Support
- Check existing issues
- Review troubleshooting section
- Try example scripts first

## Quick Reference Card

### File Formats Supported
- **Images**: JPG, JPEG, PNG, BMP, TIFF, WebP
- **Videos**: MP4, AVI, MOV, MKV, FLV, WMV

### Available Models
- **Classification**: ResNet50, ResNet101, MobileNetV2, MobileNetV3, VGG16, EfficientNetB0, InceptionV3
- **Detection**: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l
- **Face**: MTCNN, FaceRecognition
- **Gesture**: MediaPipe

### Default Settings
- Confidence threshold: 0.5
- IOU threshold: 0.45
- Top-k predictions: 5
- Input size: Model-dependent (224x224 or 640x640)

### Key Directories
- `models/weights/`: Model files
- `output/`: Saved results
- `examples/`: Demo scripts
- `config/`: Configuration files

---

**Congratulations! You're ready to explore AI Vision Framework! 🚀**

For detailed information, see the main `README.md` file.
