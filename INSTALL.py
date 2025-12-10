#!/usr/bin/env python3
"""
AI Vision Framework Pro - Automatic Installer
This script installs all dependencies and sets up the environment
"""

import subprocess
import sys
import os
import platform

def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        AI Vision Framework Pro - Auto Installer              ║
║                                                               ║
║        This will install all required dependencies           ║
║        and set up your environment automatically             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("\n[1/6] Checking Python version...")
    
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ ERROR: Python 3.8+ required, you have {version.major}.{version.minor}")
        print("Please install Python 3.8 or higher from python.org")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available"""
    print("\n[2/6] Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except:
        print("❌ ERROR: pip not found")
        print("Please install pip: python -m ensurepip --upgrade")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("\n[3/6] Upgrading pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                      check=True)
        print("✅ pip upgraded")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not upgrade pip: {e}")
        return True  # Continue anyway

def install_dependencies():
    """Install all required packages"""
    print("\n[4/6] Installing dependencies...")
    print("This may take several minutes...")
    
    # Core dependencies that must work
    core_packages = [
        "torch",
        "torchvision", 
        "opencv-python",
        "numpy",
        "pillow",
        "ultralytics",
    ]
    
    # Optional but recommended
    optional_packages = [
        "tensorflow",
        "transformers",
        "fastapi",
        "uvicorn",
        "matplotlib",
        "pandas",
        "pyyaml",
        "requests",
        "tqdm",
    ]
    
    # Install core packages
    print("\n📦 Installing CORE packages (required)...")
    for package in core_packages:
        print(f"  Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package],
                          check=True, capture_output=True)
            print(f"  ✅ {package} installed")
        except:
            print(f"  ❌ Failed to install {package}")
            print(f"  Try manually: pip install {package}")
            return False
    
    # Install optional packages
    print("\n📦 Installing OPTIONAL packages (recommended)...")
    for package in optional_packages:
        print(f"  Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package],
                          capture_output=True, timeout=300)
            print(f"  ✅ {package} installed")
        except:
            print(f"  ⚠️  {package} not installed (optional)")
    
    print("\n✅ Core dependencies installed successfully!")
    return True

def install_optional():
    """Install optional advanced features"""
    print("\n[5/6] Installing optional advanced features...")
    
    # Ask user
    response = input("\nInstall optional features (face recognition, mediapipe)? (y/n): ")
    
    if response.lower() == 'y':
        optional = [
            "mediapipe",
            "face-recognition", 
            "mtcnn",
        ]
        
        for package in optional:
            print(f"  Installing {package}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package],
                              capture_output=True, timeout=300)
                print(f"  ✅ {package} installed")
            except:
                print(f"  ⚠️  {package} not installed (you can add it later)")
    else:
        print("⏭️  Skipping optional features")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n[6/6] Creating directories...")
    
    dirs = [
        "models/weights",
        "models/cache",
        "output/logs",
        "output/analytics",
        "plugins/models",
        "temp"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ Created {dir_path}")
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("\n" + "="*60)
    print("Testing installation...")
    print("="*60)
    
    tests = {
        "PyTorch": "torch",
        "OpenCV": "cv2",
        "NumPy": "numpy",
        "PIL": "PIL",
        "Ultralytics": "ultralytics",
    }
    
    all_passed = True
    for name, module in tests.items():
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - FAILED")
            all_passed = False
    
    return all_passed

def print_success():
    """Print success message with instructions"""
    success = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        ✅ Installation Complete!                             ║
║                                                               ║
║        To start the application, run:                         ║
║                                                               ║
║        python RUN.py                                          ║
║                                                               ║
║        Or double-click RUN.py                                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

📚 Quick Start:
   1. Run: python RUN.py
   2. Click "Load Image" button
   3. Select an image file
   4. Choose a model (ResNet50 recommended for first try)
   5. Click "Run Inference"
   6. See the results!

🎯 Try Object Detection:
   1. Select "Object Detection" radio button
   2. Choose "YOLOv8n" model
   3. Load an image with objects
   4. Click "Run Inference"
   5. See detected objects with bounding boxes!

📁 Need test images? Download from:
   - https://images.unsplash.com/
   - Or use your own photos!

💡 Tips:
   - First run will download models (may take a few minutes)
   - Use YOLOv8n for fastest object detection
   - Use ResNet50 for image classification
   - Check Analytics tab for performance stats

🆘 Having issues?
   - Read README.md for troubleshooting
   - Check QUICKSTART.md for detailed guide
   - Make sure you have internet for first run (downloads models)

Enjoy! 🚀
"""
    print(success)

def main():
    """Main installation process"""
    print_banner()
    
    # Run all checks and installations
    steps = [
        ("Python Version", check_python_version),
        ("pip", check_pip),
        ("Upgrade pip", upgrade_pip),
        ("Dependencies", install_dependencies),
        ("Optional Features", install_optional),
        ("Directories", create_directories),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Installation failed at: {step_name}")
            print("Please fix the errors above and try again.")
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    # Test installation
    if test_installation():
        print_success()
        
        # Ask if user wants to run now
        print("\n" + "="*60)
        response = input("Would you like to start the application now? (y/n): ")
        if response.lower() == 'y':
            print("\nStarting application...\n")
            try:
                # Import and run
                gui_path = os.path.join(os.path.dirname(__file__), 'gui', 'main_app.py')
                subprocess.run([sys.executable, gui_path])
            except Exception as e:
                print(f"Error starting app: {e}")
                print("You can start it manually with: python RUN.py")
    else:
        print("\n⚠️  Some components failed to install.")
        print("The app may still work with reduced functionality.")
        print("Try running: python RUN.py")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please report this issue.")
        input("\nPress Enter to exit...")
        sys.exit(1)
