#!/bin/bash

# Driver Drowsiness Detection System - Installation Script
# This script handles installation on macOS and Linux systems

echo "Driver Drowsiness Detection System - Installation"
echo "================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment (optional)
read -p "Create a virtual environment? (recommended) [y/n]: " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv drowsiness_env
    source drowsiness_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Download the facial landmark model
echo "Downloading facial landmark model..."
python3 download_model.py

if [ $? -eq 0 ]; then
    echo "✅ Model downloaded successfully"
else
    echo "❌ Failed to download model"
    exit 1
fi

# Make scripts executable
chmod +x run.py
chmod +x demo.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To run the system:"
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "  source drowsiness_env/bin/activate"
fi
echo "  python3 run.py"
echo ""
echo "To run the demo:"
echo "  python3 demo.py"
echo ""
echo "To test the system:"
echo "  python3 test_detection.py"
