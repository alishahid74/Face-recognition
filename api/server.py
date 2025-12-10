"""
REST API Server for AI Vision Framework
Provides remote inference endpoints for all models
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import io
import base64
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classification import ClassificationModelFactory
from models.detection import DetectionModelFactory
from models.face_recognition import FaceModelFactory


# Create FastAPI app
app = FastAPI(
    title="AI Vision Framework API",
    description="REST API for computer vision models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ClassificationRequest(BaseModel):
    model_name: str = "ResNet50"
    top_k: int = 5


class DetectionRequest(BaseModel):
    model_name: str = "YOLOv8n"
    confidence: float = 0.5
    iou: float = 0.45


class ClassificationResponse(BaseModel):
    model: str
    predictions: List[Dict[str, float]]
    processing_time: float


class DetectionResponse(BaseModel):
    model: str
    detections: List[Dict]
    count: int
    processing_time: float


# Model cache
model_cache = {}


def get_cached_model(model_type: str, model_name: str):
    """Get or create cached model"""
    cache_key = f"{model_type}_{model_name}"
    
    if cache_key not in model_cache:
        if model_type == "classification":
            model_cache[cache_key] = ClassificationModelFactory.create_model(model_name)
        elif model_type == "detection":
            model_cache[cache_key] = DetectionModelFactory.create_model(model_name)
        elif model_type == "face":
            model_cache[cache_key] = FaceModelFactory.create_model(model_name)
    
    return model_cache[cache_key]


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Vision Framework API",
        "version": "1.0.0",
        "endpoints": {
            "classification": "/api/classify",
            "detection": "/api/detect",
            "face_detection": "/api/detect-faces",
            "models": "/api/models",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(model_cache)}


@app.get("/api/models")
async def list_models():
    """List all available models"""
    return {
        "classification": ClassificationModelFactory.get_available_models(),
        "detection": DetectionModelFactory.get_available_models(),
        "face": FaceModelFactory.get_available_models()
    }


@app.post("/api/classify")
async def classify_image(
    file: UploadFile = File(...),
    model_name: str = Form("ResNet50"),
    top_k: int = Form(5)
):
    """
    Classify an uploaded image
    
    Args:
        file: Image file
        model_name: Classification model name
        top_k: Number of top predictions
    
    Returns:
        Classification results
    """
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Get model
        model = get_cached_model("classification", model_name)
        
        # Preprocess
        from utils.image_utils import ImagePreprocessor
        from config import MODELS_CONFIG
        
        preprocessor = ImagePreprocessor()
        config = MODELS_CONFIG['image_classification'][model_name]
        
        img_tensor = preprocessor.preprocess_for_classification(
            tmp_path,
            input_size=config['input_size'],
            framework=config['framework']
        )
        
        # Predict
        results = model.predict(img_tensor, top_k=top_k)
        
        # Clean up
        os.unlink(tmp_path)
        
        processing_time = time.time() - start_time
        
        return ClassificationResponse(
            model=model_name,
            predictions=[{"class": label, "confidence": conf} for label, conf in results],
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Form("YOLOv8n"),
    confidence: float = Form(0.5),
    iou: float = Form(0.45),
    return_image: bool = Form(False)
):
    """
    Detect objects in an uploaded image
    
    Args:
        file: Image file
        model_name: Detection model name
        confidence: Confidence threshold
        iou: IOU threshold
        return_image: Whether to return annotated image
    
    Returns:
        Detection results
    """
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Get model
        model = get_cached_model("detection", model_name)
        
        # Detect
        results = model.detect(tmp_path, confidence=confidence, iou=iou)
        
        # Prepare response
        detections = []
        for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
            detections.append({
                "class": label,
                "confidence": float(score),
                "bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3])
                }
            })
        
        # Clean up
        os.unlink(tmp_path)
        
        processing_time = time.time() - start_time
        
        response_data = {
            "model": model_name,
            "detections": detections,
            "count": results['count'],
            "processing_time": processing_time
        }
        
        # Return annotated image if requested
        if return_image:
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', results['image'])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data["image"] = img_base64
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-faces")
async def detect_faces(
    file: UploadFile = File(...),
    model_name: str = Form("MTCNN"),
    return_image: bool = Form(False)
):
    """
    Detect faces in an uploaded image
    
    Args:
        file: Image file
        model_name: Face detection model name
        return_image: Whether to return annotated image
    
    Returns:
        Face detection results
    """
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Get model
        model = get_cached_model("face", model_name)
        
        # Detect
        results = model.detect_faces(tmp_path)
        
        # Prepare response
        faces = []
        for face_data in results['faces']:
            faces.append({
                "bbox": face_data['box'],
                "confidence": face_data['confidence'],
                "keypoints": face_data.get('keypoints', {})
            })
        
        # Clean up
        os.unlink(tmp_path)
        
        processing_time = time.time() - start_time
        
        response_data = {
            "model": model_name,
            "faces": faces,
            "count": results['count'],
            "processing_time": processing_time
        }
        
        # Return annotated image if requested
        if return_image:
            _, buffer = cv2.imencode('.jpg', results['image'])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data["image"] = img_base64
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-classify")
async def batch_classify(
    files: List[UploadFile] = File(...),
    model_name: str = Form("ResNet50"),
    top_k: int = Form(5)
):
    """
    Batch classification of multiple images
    
    Args:
        files: List of image files
        model_name: Classification model name
        top_k: Number of top predictions
    
    Returns:
        Batch classification results
    """
    results = []
    
    for file in files:
        try:
            result = await classify_image(file, model_name, top_k)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"batch_results": results, "total": len(files)}


@app.get("/api/model-info/{model_type}/{model_name}")
async def get_model_info(model_type: str, model_name: str):
    """
    Get information about a specific model
    
    Args:
        model_type: Type of model (classification, detection, face)
        model_name: Model name
    
    Returns:
        Model information
    """
    try:
        if model_type == "classification":
            info = ClassificationModelFactory.get_model_info(model_name)
        elif model_type == "detection":
            info = DetectionModelFactory.get_model_info(model_name)
        elif model_type == "face":
            # Face models info
            info = {"name": model_name, "type": "face"}
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        if info is None:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the API server
    
    Args:
        host: Host address
        port: Port number
        reload: Auto-reload on code changes
    """
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║        AI Vision Framework API Server                         ║
║                                                               ║
║        Server running at: http://{host}:{port}            ║
║        Documentation: http://{host}:{port}/docs          ║
║        Health check: http://{host}:{port}/health         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Vision Framework API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--reload', action='store_true', help='Auto-reload on changes')
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port, reload=args.reload)
