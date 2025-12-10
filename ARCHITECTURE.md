# System Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │     GUI      │  │  CLI Tools   │  │  Python API  │          │
│  │  (Tkinter)   │  │  (Examples)  │  │  (Direct)    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
└─────────┼─────────────────┼──────────────────┼───────────────────┘
          │                 │                  │
          └─────────────────┴──────────────────┘
                            │
┌───────────────────────────┼───────────────────────────────────────┐
│                    Application Layer                              │
├───────────────────────────┴───────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Model Factories (Factory Pattern)               │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │ │
│  │  │ Classification │  │   Detection    │  │     Face     │ │ │
│  │  │    Factory     │  │    Factory     │  │   Factory    │ │ │
│  │  └────────┬───────┘  └────────┬───────┘  └──────┬───────┘ │ │
│  │           │                   │                  │          │ │
│  │  ┌────────▼────────────────────▼──────────────────▼──────┐ │ │
│  │  │            Image Preprocessor & Utilities             │ │ │
│  │  └───────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┼────────────────────────────────────────┐
│                      Model Layer                                   │
├───────────────────────────┴────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  PyTorch     │  │ TensorFlow   │  │ Ultralytics  │            │
│  │              │  │              │  │              │            │
│  │ • ResNet     │  │ • Inception  │  │ • YOLOv8     │            │
│  │ • MobileNet  │  │ • Keras Apps │  │              │            │
│  │ • VGG        │  │              │  │              │            │
│  │ • EfficientNet│ │              │  │              │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                               │
│  │ MediaPipe    │  │Face Recognition                              │
│  │              │  │              │                               │
│  │ • Hands      │  │ • dlib       │                               │
│  │              │  │ • MTCNN      │                               │
│  └──────────────┘  └──────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│                    Framework Layer                                  │
├───────────────────────────┴─────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  OpenCV  │  │  NumPy   │  │  Pillow  │  │  Scipy   │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┼──────────────────────────────────────────┐
│                     Hardware Layer                                   │
├───────────────────────────┴──────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────┐        ┌──────────────────────┐           │
│  │      CPU/GPU         │        │    Storage           │           │
│  │  (Computation)       │        │  (Model Weights)     │           │
│  └──────────────────────┘        └──────────────────────┘           │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagrams

### Image Classification Flow
```
┌──────────┐
│  Image   │
│  Input   │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│  Load & Resize  │
│  (PIL/OpenCV)   │
└────┬────────────┘
     │
     ▼
┌─────────────────────┐
│  Preprocessing      │
│  • Normalize        │
│  • ToTensor         │
│  • Framework-spec   │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│  Model Inference    │
│  • ResNet           │
│  • MobileNet        │
│  • etc.             │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│  Post-processing    │
│  • Softmax          │
│  • Top-K selection  │
└────┬────────────────┘
     │
     ▼
┌─────────────────┐
│  Display        │
│  Results        │
└─────────────────┘
```

### Object Detection Flow
```
┌──────────┐
│  Image   │
│  /Video  │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│  Resize &       │
│  Pad Image      │
└────┬────────────┘
     │
     ▼
┌─────────────────────┐
│  YOLO Inference     │
│  • Detect objects   │
│  • Predict classes  │
│  • Generate boxes   │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│  Non-Max            │
│  Suppression (NMS)  │
│  • Filter by conf   │
│  • Remove overlap   │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│  Draw Bounding      │
│  Boxes & Labels     │
└────┬────────────────┘
     │
     ▼
┌─────────────────┐
│  Display        │
│  Annotated      │
└─────────────────┘
```

### Face Recognition Flow
```
┌──────────┐
│  Image   │
│  Input   │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│  Face           │
│  Detection      │
│  (MTCNN/dlib)   │
└────┬────────────┘
     │
     ▼
┌─────────────────────┐
│  Extract Face       │
│  Encodings          │
│  (128-D vectors)    │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│  Compare with       │
│  Known Faces        │
│  (Distance calc)    │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│  Identify Person    │
│  or "Unknown"       │
└────┬────────────────┘
     │
     ▼
┌─────────────────┐
│  Draw Results   │
│  with Labels    │
└─────────────────┘
```

## Component Interaction

### GUI Application Flow
```
┌─────────────────────────────────────────────────────────┐
│                   GUI Main Window                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐       ┌──────────────────┐       │
│  │  Control Panel   │       │  Display Panel   │       │
│  ├──────────────────┤       ├──────────────────┤       │
│  │                  │       │                  │       │
│  │ • Task Select    │       │ • Image Canvas   │       │
│  │ • Model Select   │◄──────┤ • Video Stream   │       │
│  │ • Settings       │       │                  │       │
│  │ • Buttons        │       └──────────────────┘       │
│  │                  │              │                    │
│  └─────────┬────────┘              │                    │
│            │                       │                    │
│            ▼                       ▼                    │
│  ┌──────────────────────────────────────────┐          │
│  │         Results Panel                     │          │
│  ├──────────────────────────────────────────┤          │
│  │ • Predictions                             │          │
│  │ • Confidence scores                       │          │
│  │ • Object counts                           │          │
│  │ • Processing time                         │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
└──────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              Background Processing Thread                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Load model (if not cached)                          │
│  2. Preprocess input                                    │
│  3. Run inference                                       │
│  4. Post-process results                                │
│  5. Update GUI (thread-safe)                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Class Hierarchy

### Model Classes
```
ModelBase (Abstract)
    │
    ├── ClassificationModel
    │   ├── PyTorchClassifier
    │   │   ├── ResNet50
    │   │   ├── MobileNetV2
    │   │   └── EfficientNetB0
    │   └── TensorFlowClassifier
    │       └── InceptionV3
    │
    ├── DetectionModel
    │   └── YOLODetector
    │       ├── YOLOv8n
    │       ├── YOLOv8s
    │       └── YOLOv8m/l
    │
    ├── FaceModel
    │   ├── MTCNNDetector
    │   └── FaceRecognizer
    │
    └── GestureModel
        └── MediaPipeRecognizer
```

### Factory Classes
```
ModelFactory (Pattern)
    │
    ├── ClassificationModelFactory
    │   ├── create_model()
    │   ├── get_available_models()
    │   └── get_model_info()
    │
    ├── DetectionModelFactory
    │   ├── create_model()
    │   └── ...
    │
    ├── FaceModelFactory
    │   └── ...
    │
    └── GestureModelFactory
        └── ...
```

## Configuration System

```
config/config.py
    │
    ├── MODELS_CONFIG
    │   ├── image_classification
    │   │   └── {model_name: config_dict}
    │   ├── object_detection
    │   ├── face_recognition
    │   └── gesture_recognition
    │
    ├── Paths
    │   ├── MODELS_DIR
    │   ├── CACHE_DIR
    │   └── BASE_DIR
    │
    ├── Thresholds
    │   ├── DEFAULT_CONFIDENCE_THRESHOLD
    │   └── DEFAULT_IOU_THRESHOLD
    │
    └── Classes
        ├── IMAGENET_CLASSES
        └── COCO_CLASSES
```

## Extension Points

```
New Model Integration:

1. Create model class in models/
   └── Inherit from base or create new

2. Update config.py
   └── Add model configuration

3. Update/Create factory
   └── Add creation logic

4. Update GUI
   └── Add to model list (automatic via config)

5. Create example script
   └── Demonstrate usage
```

## Thread Safety

```
Main Thread (GUI)
    │
    ├── UI Updates
    ├── Event Handling
    └── Display Rendering
    
Background Threads
    │
    ├── Model Loading
    ├── Inference
    ├── Video Processing
    └── Results → Queue → Main Thread
```

## Memory Management

```
Model Lifecycle:
    Load → Cache → Reuse → Cleanup
    
Image Lifecycle:
    Load → Process → Display → Release
    
Video Lifecycle:
    Open → Frame Loop → Close
    ├── Read Frame
    ├── Process
    ├── Display
    └── Repeat
```

## Error Handling Flow

```
Try-Catch Hierarchy:
    │
    ├── File I/O Errors
    │   └── Display error message
    │
    ├── Model Loading Errors
    │   └── Fallback or inform user
    │
    ├── Inference Errors
    │   └── Log and continue
    │
    └── GUI Errors
        └── Prevent crash, show dialog
```

---

This architecture provides:
- ✅ Separation of concerns
- ✅ Easy extensibility
- ✅ Framework independence
- ✅ Clean interfaces
- ✅ Thread safety
- ✅ Error resilience
