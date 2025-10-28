"""
Configuration file for the Driver Drowsiness Detection System
"""

# Detection Parameters
EAR_THRESHOLD = 0.25          # Eye Aspect Ratio threshold for drowsiness detection
MAR_THRESHOLD = 0.6           # Mouth Aspect Ratio threshold for yawning detection
CONSECUTIVE_FRAMES = 15       # Number of consecutive frames required for confirmation
ALERT_DURATION = 3.0          # Duration of alert in seconds

# Camera Settings
CAMERA_INDEX = 0              # Camera index (0 for default camera)
FRAME_WIDTH = 640             # Frame width for processing
FRAME_HEIGHT = 480            # Frame height for processing
TARGET_FPS = 30               # Target FPS for camera

# Performance Settings
MAX_FPS_HISTORY = 30          # Maximum FPS history for averaging
PROCESSING_SCALE = 1.0        # Scale factor for processing (1.0 = full resolution)

# Alert Settings
SOUND_ENABLED = True          # Enable sound alerts by default
VISUAL_ALERTS = True          # Enable visual alerts
ALERT_SOUND_FREQUENCY = 800   # Frequency of alert sound in Hz
ALERT_SOUND_DURATION = 0.5    # Duration of each alert sound in seconds

# UI Settings
SHOW_LANDMARKS = True         # Show facial landmarks on video
SHOW_FPS = True              # Show FPS counter
SHOW_METRICS = True          # Show EAR/MAR values
FONT_SCALE = 0.7             # Font scale for text overlay
FONT_THICKNESS = 2           # Font thickness for text overlay

# Logging Settings
LOG_LEVEL = "INFO"           # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_TO_FILE = False          # Enable logging to file
LOG_FILE = "drowsiness_detection.log"  # Log file name

# Model Settings
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"  # Path to facial landmark model

# Advanced Settings
ENABLE_HEAD_POSE = False     # Enable head pose estimation (experimental)
ENABLE_CNN_MODEL = False     # Enable CNN model for eye state detection (experimental)
CNN_MODEL_PATH = "eye_state_model.pth"  # Path to CNN model

# Debug Settings
DEBUG_MODE = False           # Enable debug mode with additional information
SAVE_FRAMES = False          # Save frames when drowsiness is detected
SAVE_DIRECTORY = "debug_frames"  # Directory to save debug frames
