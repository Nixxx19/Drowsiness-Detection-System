# Driver Drowsiness Detection System

A real-time drowsiness detection system that uses computer vision and machine learning to monitor driver alertness through facial landmark analysis. This system helps prevent accidents caused by driver fatigue by providing immediate alerts when drowsiness is detected.

## 🚀 Features

- **Real-time Detection**: Monitors driver's face in real-time using webcam
- **Eye Blink Detection**: Uses Eye Aspect Ratio (EAR) to detect prolonged eye closure
- **Yawning Detection**: Detects yawning through Mouth Aspect Ratio (MAR) analysis
- **Audio Alerts**: Plays warning sounds when drowsiness is detected
- **Visual Feedback**: On-screen indicators showing system status and metrics
- **High Performance**: Optimized for real-time processing (≥10 FPS)
- **Cross-platform**: Works on macOS, Windows, and Linux
- **Privacy-focused**: All processing happens locally on your device

## 🛠 Technology Stack

- **Python 3.x**: Core programming language
- **OpenCV**: Computer vision and image processing
- **dlib**: Facial landmark detection
- **NumPy/SciPy**: Mathematical computations
- **Pygame**: Audio alert system
- **SciPy**: Distance calculations for EAR/MAR

## 📦 Installation

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/Nixxx19/Drowsiness-Detection-System.git
cd drowsiness-detection-system

# Run the setup script
python setup.py
```

### Manual Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the facial landmark predictor model:
```bash
python download_model.py
```

## 🎮 Usage

### Basic Usage
```bash
python drowsiness_detector.py
```

### Test the System
```bash
python test_detection.py
```

### Controls
- Press `q` to quit the application
- Press `s` to toggle sound alerts on/off

## 🧮 Algorithm Details

### Eye Aspect Ratio (EAR)
The system calculates EAR using the formula:
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
Where p1-p6 are specific facial landmarks around the eye.

- **EAR > 0.25**: Eyes are open
- **EAR < 0.25**: Eyes are closed
- **Consecutive frames**: 15-20 frames of low EAR = drowsiness

### Mouth Aspect Ratio (MAR)
Yawning is detected using MAR:
```
MAR = |p3-p9| / |p1-p7|
```
Where p1, p3, p7, p9 are mouth landmarks.

- **MAR > 0.6**: Mouth is open (yawning)
- **MAR < 0.6**: Mouth is closed

## ⚙️ Configuration

You can adjust detection sensitivity by modifying these parameters in `drowsiness_detector.py`:

```python
# Detection thresholds
EAR_THRESHOLD = 0.25        # Lower = more sensitive to eye closure
MAR_THRESHOLD = 0.6         # Higher = more sensitive to yawning
CONSECUTIVE_FRAMES = 15     # Frames required for confirmation
ALERT_DURATION = 3.0        # Alert duration in seconds
```

## 📊 Performance Metrics

The system is optimized for real-time performance:
- **Target FPS**: ≥10 FPS
- **Latency**: ≤300ms
- **Detection Accuracy**: ≥90%
- **False Alarm Rate**: ≤5%

## 🔧 System Requirements

- **Python**: 3.7 or higher
- **Camera**: Webcam or USB camera
- **OS**: macOS, Windows, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor recommended

## 🐛 Troubleshooting

### Camera Issues
- Ensure camera is not being used by another application
- Try different camera indices (0, 1, 2, etc.)
- Check camera permissions in system settings

### Model Download Issues
- Ensure stable internet connection
- Check available disk space (model is ~100MB)
- Verify write permissions in the project directory

### Performance Issues
- Close other applications to free up resources
- Reduce camera resolution if needed
- Ensure good lighting conditions
- Check if antivirus is scanning the application

### Installation Issues
- Update pip: `pip install --upgrade pip`
- Install dependencies individually if batch install fails
- Check Python version compatibility

## 📁 Project Structure

```
drowsiness-detection-system/
├── drowsiness_detector.py      # Main application
├── download_model.py           # Model download script
├── setup.py                    # Setup script
├── test_detection.py           # Test suite
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── shape_predictor_68_face_landmarks.dat  # Facial landmark model
```

## 🔬 Testing

Run the comprehensive test suite:
```bash
python test_detection.py
```

Tests include:
- EAR calculation accuracy
- MAR calculation accuracy
- Camera functionality
- Performance benchmarks

## 🚦 Usage Scenarios

### Individual Drivers
- Long-distance driving
- Night driving
- Commercial vehicle operation

### Fleet Management
- Driver monitoring systems
- Safety compliance
- Accident prevention

### Research & Development
- Driver behavior analysis
- Safety system prototyping
- ML model training

## 🔒 Privacy & Security

- **Local Processing**: All video processing happens on your device
- **No Data Transmission**: No video or personal data is sent to external servers
- **Temporary Storage**: Video frames are processed in memory only
- **Secure**: No network connectivity required for core functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [dlib](http://dlib.net/) library for facial landmark detection
- [OpenCV](https://opencv.org/) for computer vision capabilities
- The research community for EAR and MAR algorithms
- Contributors and testers who helped improve the system

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Run the test suite to identify problems
3. Create an issue on GitHub with detailed information
4. Include system specifications and error messages

---

**⚠️ Safety Notice**: This system is designed to assist drivers but should not replace proper rest and responsible driving practices. Always ensure you are well-rested before driving.