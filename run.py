#!/usr/bin/env python3
"""
AI Vision Framework Launcher

Main entry point for the AI Vision Framework application.
This script checks dependencies and launches the GUI.
"""

import sys
import os
import subprocess


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'torchvision',
        'tensorflow',
        'cv2',
        'PIL',
        'numpy',
        'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Error: Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt --break-system-packages")
        return False
    
    return True


def check_optional_dependencies():
    """Check optional dependencies and warn if missing"""
    optional_packages = {
        'mediapipe': 'Required for gesture recognition',
        'face_recognition': 'Required for advanced face recognition',
        'mtcnn': 'Required for MTCNN face detection'
    }
    
    missing_optional = []
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append((package, description))
    
    if missing_optional:
        print("\nOptional packages not installed:")
        for pkg, desc in missing_optional:
            print(f"  - {pkg}: {desc}")
        print("\nInstall with: pip install <package-name> --break-system-packages")
        print()


def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        🚀 AI Vision Framework v1.0                           ║
║                                                               ║
║        Deep Learning Models for Computer Vision               ║
║                                                               ║
║        • Image Classification                                 ║
║        • Object Detection                                     ║
║        • Face Recognition                                     ║
║        • Gesture Recognition                                  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def launch_gui():
    """Launch the GUI application"""
    print("Launching GUI application...")
    print()
    
    gui_path = os.path.join(os.path.dirname(__file__), 'main_app.py')

    
    try:
        subprocess.run([sys.executable, gui_path], check=True)
    except KeyboardInterrupt:
        print("\n\nApplication closed by user")
    except Exception as e:
        print(f"\nError launching GUI: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✓ All required dependencies installed")
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Launch GUI
    launch_gui()


if __name__ == "__main__":
    main()
