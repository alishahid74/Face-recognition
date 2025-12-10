#!/bin/bash

# AI Vision Framework Setup Script
# This script sets up the environment and installs dependencies

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║        AI Vision Framework - Setup Script                     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python found"
echo ""

# Check pip
echo "Checking pip..."
python3 -m pip --version

if [ $? -ne 0 ]; then
    echo "Error: pip is not installed"
    exit 1
fi

echo "✓ pip found"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip --break-system-packages

# Install core dependencies
echo ""
echo "Installing core dependencies..."
echo "This may take several minutes..."
echo ""

python3 -m pip install torch torchvision --break-system-packages
python3 -m pip install tensorflow --break-system-packages
python3 -m pip install opencv-python --break-system-packages
python3 -m pip install numpy pillow scipy --break-system-packages
python3 -m pip install ultralytics --break-system-packages

echo ""
echo "✓ Core dependencies installed"
echo ""

# Install optional dependencies
read -p "Install optional dependencies (MediaPipe, face_recognition)? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing optional dependencies..."
    
    python3 -m pip install mediapipe --break-system-packages
    python3 -m pip install face-recognition mtcnn --break-system-packages
    
    echo "✓ Optional dependencies installed"
fi

echo ""

# Create directories
echo "Creating necessary directories..."
mkdir -p models/weights
mkdir -p models/cache
mkdir -p output
mkdir -p examples

echo "✓ Directories created"
echo ""

# Download sample images (optional)
read -p "Download sample images for testing? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading sample images..."
    mkdir -p sample_images
    
    # You can add URLs to download sample images here
    echo "Note: Add sample images to the sample_images/ directory manually"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║        Setup Complete!                                        ║"
echo "║                                                               ║"
echo "║        To launch the application:                             ║"
echo "║        python3 run.py                                         ║"
echo "║                                                               ║"
echo "║        Or directly:                                           ║"
echo "║        python3 gui/main_app.py                                ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
