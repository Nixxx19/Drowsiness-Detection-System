#!/usr/bin/env python3
"""
Launcher script for the Driver Drowsiness Detection System
"""

import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'cv2', 'tensorflow', 'numpy', 'scipy', 'pygame'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'numpy':
                import numpy
            elif package == 'scipy':
                import scipy
            elif package == 'pygame':
                import pygame
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please run: python setup.py")
        return False
    
    return True

def check_model():
    """Check if CNN models exist"""
    model_paths = [
        "eye_state_cnn_model.h5",
        "sequence_model.h5", 
        "face_analysis_model.h5"
    ]
    
    missing_models = []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        logger.warning(f"CNN models not found: {', '.join(missing_models)}")
        logger.info("Models will be created automatically on first run")
        return True  # Allow to continue, models will be created
    
    return True

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("Could not open camera. Make sure a camera is connected.")
            return False
        cap.release()
        return True
    except Exception as e:
        logger.error(f"Error checking camera: {e}")
        return False

def main():
    """Main launcher function"""
    print("=" * 60)
    print("Driver Drowsiness Detection System")
    print("=" * 60)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    logger.info("✓ All dependencies are installed")
    
    # Check model
    logger.info("Checking facial landmark model...")
    if not check_model():
        sys.exit(1)
    logger.info("✓ Facial landmark model is ready")
    
    # Check camera
    logger.info("Checking camera...")
    if not check_camera():
        logger.warning("Camera check failed, but continuing...")
    else:
        logger.info("✓ Camera is available")
    
    print("\nStarting drowsiness detection system...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to toggle sound alerts")
    print("  - Press 'h' to show this help")
    print("\n" + "=" * 60)
    
    # Launch the main application
    try:
        from cnn_drowsiness_detector import main as detector_main
        detector_main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
