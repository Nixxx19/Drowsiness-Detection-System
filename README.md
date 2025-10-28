#  Driver Drowsiness Detection System

Real-time, camera-based monitoring to detect driver fatigue using facial landmarks and optional deep learning, with on-device processing and immediate alerts.

### 1. Overview
The system monitors facial landmarks from a webcam or in-vehicle camera to detect eye closure, excessive blinking, and yawning. When drowsiness indicators are detected, it triggers visual and/or audio alerts.

### 2. Objectives
- **Primary goal**: Reduce road accidents caused by driver fatigue through real-time detection and alert mechanisms.
- **Secondary goals**:
  - Implement a lightweight, camera-based detection system without external sensors
  - Maintain high accuracy across lighting and face-angle variations
  - Operate in real-time (≥ 10 FPS)

### 3. Target Users
| User Type | Description | Needs |
|---|---|---|
| Individual Drivers | Car and truck drivers who drive for long durations | Alert when drowsiness occurs |
| Fleet Companies | Organizations managing multiple vehicles | Monitoring to prevent fatigue-related accidents |
| Automobile OEMs | Manufacturers seeking safety feature integrations | Embedded ML solution for smart vehicles |

### 4. Problem Statement
Drowsy driving causes thousands of accidents annually. Manual monitoring is impractical, and existing commercial systems are expensive. A low-cost, camera-based ML system is needed to detect early signs of drowsiness in real time.

### 5. Key Features & Requirements
| Category | Feature | Description | Priority |
|---|---|---|---|
| Core Detection | Eye Blink & Closure | Detect prolonged eye closure using Eye Aspect Ratio (EAR) | ⭐⭐⭐⭐ |
|  | Yawning Detection | Identify mouth opening using Mouth Aspect Ratio (MAR) | ⭐⭐⭐ |
|  | Head Pose Tracking | Detect head droop or tilt (optional enhancement) | ⭐⭐ |
| Alerts | Audio Alarm | Play alert sound when drowsiness detected | ⭐⭐⭐⭐ |
|  | Visual Indicator | Display on-screen warning | ⭐⭐⭐ |
| Performance | Real-Time Processing | Process at least 10 FPS | ⭐⭐⭐⭐ |
| Reliability | Offline Functionality | Should not rely on internet connectivity | ⭐⭐⭐ |
| UX/UI | Dashboard View | Display EAR, MAR, and state (alert/drowsy) | ⭐⭐ |
| Scalability | Plug-and-Play | Works with normal webcam or USB camera | ⭐⭐⭐ |
| ML | Deep CNN Eye-State Model | Complements EAR for robustness in low-light/occlusions | ⭐⭐⭐ |
| Integration | Vehicle IoT/Feedback | Trigger seat vibration/buzzer via GPIO/serial when drowsy | ⭐⭐ |

### 6. Technology Stack
| Layer | Technology | Description |
|---|---|---|
| Programming | Python 3.x | Core development language |
| CV Library | OpenCV | Image/video capture and processing |
| ML Library | Dlib | Facial landmark detection |
| Math & Utils | SciPy, NumPy, Imutils | EAR/MAR computation, geometry |
| Alert System | Pygame / Playsound | Audio alerts |
| ML | PyTorch / TensorFlow | CNN model for deep eye-state detection |

### 7. System Workflow
```text
[Start Webcam Feed]
      ↓
[Face Detection → Landmark Extraction]
      ↓
[Calculate EAR (Eyes) & MAR (Mouth)]
      ↓
[Threshold Comparison → Drowsiness Check]
      ↓
[If Drowsy → Trigger Alert]
      ↓
[Log or Display State on Dashboard]
```

### 8. Algorithms
- **Eye Aspect Ratio (EAR)**: `EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)`
  - Threshold: EAR < 0.25 for 15–20 consecutive frames → Eyes closed

- **Mouth Aspect Ratio (MAR)**: `MAR = |p3 - p9| / |p1 - p7|`
  - Threshold: MAR > 0.6 → Yawning detected

- **Optional**: Lightweight CNN for eye-state classification to complement EAR under challenging conditions (low light, occlusions).

### 9. Non-Functional Requirements
- **Performance**: Real-time processing at ≥ 10 FPS on typical laptop hardware
- **Privacy**: All processing on-device; no video frames leave the machine
- **Reliability**: Debouncing and consecutive-frame thresholds to avoid spurious alerts
- **Portability**: Works with built-in webcams and common USB cameras (macOS/Windows/Linux)
- **Usability**: Minimal UI with clear alert feedback and start/stop controls

### 10. Success Metrics
| Metric | Target |
|---|---|
| Detection Accuracy | ≥ 90% on test dataset |
| False Alarm Rate | ≤ 5% |
| Real-Time FPS | ≥ 10 FPS |
| Latency | ≤ 300 ms |
| Power Efficiency | Suitable for laptop/car camera use |

### 11. Risks & Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| Poor lighting affects accuracy | High | Histogram equalization, adaptive thresholding |
| Different head angles | Medium | Add head pose estimation |
| False positives (blink = drowsy) | Medium | Use consecutive-frame logic |
| Processing lag | Low | Optimize with multithreading, smaller frame size |

### Getting Started (High-Level)
1. Install Python 3.x and dependencies: OpenCV, dlib, NumPy, SciPy, imutils, pygame/playsound
2. Connect a webcam/USB camera and run the application
3. Adjust EAR/MAR thresholds as needed for your camera/environment

