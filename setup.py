"""
Setup script for the Driver Drowsiness Detection System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def download_model():
    """Download the facial landmark predictor model"""
    print("Downloading facial landmark predictor model...")
    try:
        subprocess.check_call([sys.executable, "download_model.py"])
        print("✓ Model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download model: {e}")
        return False

def main():
    """Main setup function"""
    print("Setting up Driver Drowsiness Detection System...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation")
        return False
    
    # Download model
    if not download_model():
        print("Setup failed at model download")
        return False
    
    print("=" * 50)
    print("✓ Setup completed successfully!")
    print("\nTo run the system:")
    print("python drowsiness_detector.py")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 's' to toggle sound alerts")
    
    return True

if __name__ == "__main__":
    main()
