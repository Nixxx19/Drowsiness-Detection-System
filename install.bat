@echo off
REM Driver Drowsiness Detection System - Windows Installation Script

echo Driver Drowsiness Detection System - Installation
echo =================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

echo ✅ pip found

REM Create virtual environment (optional)
set /p create_venv="Create a virtual environment? (recommended) [y/n]: "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv drowsiness_env
    call drowsiness_env\Scripts\activate.bat
    echo ✅ Virtual environment created and activated
)

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

REM Download the facial landmark model
echo Downloading facial landmark model...
python download_model.py

if %errorlevel% neq 0 (
    echo ❌ Failed to download model
    pause
    exit /b 1
)

echo ✅ Model downloaded successfully

echo.
echo 🎉 Installation completed successfully!
echo.
echo To run the system:
if /i "%create_venv%"=="y" (
    echo   drowsiness_env\Scripts\activate.bat
)
echo   python run.py
echo.
echo To run the demo:
echo   python demo.py
echo.
echo To test the system:
echo   python test_detection.py
echo.
pause
