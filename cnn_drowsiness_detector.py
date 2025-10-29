"""
Advanced CNN-based Drowsiness Detection System
Uses deep learning for accurate eye state classification and drowsiness detection
"""

import cv2
import numpy as np
import time
import logging
import os
from collections import deque
import threading

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                       Dropout, BatchNormalization, GlobalAveragePooling2D,
                                       Input, Concatenate, Reshape, LSTM, TimeDistributed)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Please install: pip install tensorflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedCNNDrowsinessDetector:
    """
    Advanced CNN-based drowsiness detection system using multiple deep learning models
    """
    
    def __init__(self):
        """Initialize the advanced CNN drowsiness detector"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Model parameters
        self.eye_model = None
        self.sequence_model = None
        self.face_model = None
        
        # Detection parameters
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.6
        self.CONSECUTIVE_FRAMES = 15
        self.ALERT_DURATION = 3.0
        
        # State tracking
        self.eye_state_history = deque(maxlen=30)
        self.ear_history = deque(maxlen=self.CONSECUTIVE_FRAMES)
        self.mar_history = deque(maxlen=self.CONSECUTIVE_FRAMES)
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.alert_active = False
        self.alert_start_time = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all CNN models"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot initialize models.")
            return
        
        # Load or create eye state classification model
        self.eye_model = self._create_eye_state_model()
        
        # Load or create sequence model for temporal analysis
        self.sequence_model = self._create_sequence_model()
        
        # Load or create face analysis model
        self.face_model = self._create_face_analysis_model()
        
        logger.info("All CNN models initialized successfully")
    
    def _create_eye_state_model(self):
        """Create advanced CNN model for eye state classification"""
        model = Sequential([
            # Input layer
            Input(shape=(64, 64, 1)),
            
            # First convolutional block with residual connection
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Dense layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')  # 2 classes: open, closed
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_sequence_model(self):
        """Create LSTM model for temporal sequence analysis"""
        model = Sequential([
            Input(shape=(30, 2)),  # 30 timesteps, 2 features (EAR, MAR)
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Drowsiness probability
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_face_analysis_model(self):
        """Create face analysis model using transfer learning"""
        # Use MobileNetV2 as base model
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(3, activation='softmax')(x)  # 3 classes: alert, drowsy, normal
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _extract_eye_region(self, frame, face_roi):
        """Extract eye region from face ROI"""
        x, y, w, h = face_roi
        face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
        
        if len(eyes) >= 2:
            # Use the two largest eyes
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            
            # Extract and combine both eyes
            eye_regions = []
            for (ex, ey, ew, eh) in eyes:
                eye_img = face_gray[ey:ey+eh, ex:ex+ew]
                if eye_img.size > 0:
                    eye_img = cv2.resize(eye_img, (64, 64))
                    eye_img = eye_img.astype(np.float32) / 255.0
                    eye_regions.append(eye_img)
            
            if len(eye_regions) == 2:
                # Combine both eyes
                combined_eye = np.concatenate(eye_regions, axis=1)
                return combined_eye.reshape(64, 128, 1)
        
        return None
    
    def _calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio using facial landmarks"""
        try:
            if len(eye_landmarks) < 6:
                return 0.0
            
            # Extract key points
            p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
            
            # Calculate distances
            vertical_1 = np.linalg.norm(p2 - p6)
            vertical_2 = np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4)
            
            # Calculate EAR
            if horizontal == 0:
                return 0.0
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0
    
    def _calculate_mar(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio using facial landmarks"""
        try:
            if len(mouth_landmarks) < 9:
                return 0.0
            
            # Extract key points
            p1, p3, p7, p9 = mouth_landmarks[0], mouth_landmarks[2], mouth_landmarks[6], mouth_landmarks[8]
            
            # Calculate distances
            vertical = np.linalg.norm(p3 - p7)
            horizontal = np.linalg.norm(p1 - p9)
            
            # Calculate MAR
            if horizontal == 0:
                return 0.0
            mar = vertical / horizontal
            return mar
        except Exception as e:
            logger.error(f"Error calculating MAR: {e}")
            return 0.0
    
    def _predict_eye_state_cnn(self, eye_region):
        """Predict eye state using CNN model"""
        if self.eye_model is None or eye_region is None:
            return 0.5, 0.5  # Default probabilities
        
        try:
            # Preprocess image
            if len(eye_region.shape) == 3:
                eye_region = eye_region.reshape(1, 64, 128, 1)
            else:
                eye_region = eye_region.reshape(1, 64, 64, 1)
            
            # Predict
            prediction = self.eye_model.predict(eye_region, verbose=0)
            open_prob = prediction[0][1]  # Probability of being open
            closed_prob = prediction[0][0]  # Probability of being closed
            
            return open_prob, closed_prob
        except Exception as e:
            logger.error(f"Error in CNN prediction: {e}")
            return 0.5, 0.5
    
    def _predict_drowsiness_sequence(self, ear, mar):
        """Predict drowsiness using sequence model"""
        if self.sequence_model is None:
            return 0.0
        
        try:
            # Add current features to history
            self.ear_history.append(ear)
            self.mar_history.append(mar)
            
            if len(self.ear_history) < 30:
                return 0.0
            
            # Prepare sequence data
            sequence_data = np.array([list(self.ear_history), list(self.mar_history)]).T
            sequence_data = sequence_data.reshape(1, 30, 2)
            
            # Predict
            prediction = self.sequence_model.predict(sequence_data, verbose=0)
            return prediction[0][0]
        except Exception as e:
            logger.error(f"Error in sequence prediction: {e}")
            return 0.0
    
    def _predict_face_analysis(self, face_region):
        """Predict overall face state using face analysis model"""
        if self.face_model is None or face_region is None:
            return 0.0
        
        try:
            # Preprocess face region
            face_resized = cv2.resize(face_region, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_rgb = face_rgb.astype(np.float32) / 255.0
            face_rgb = face_rgb.reshape(1, 224, 224, 3)
            
            # Predict
            prediction = self.face_model.predict(face_rgb, verbose=0)
            drowsy_prob = prediction[0][1]  # Probability of being drowsy
            
            return drowsy_prob
        except Exception as e:
            logger.error(f"Error in face analysis: {e}")
            return 0.0
    
    def process_frame(self, frame):
        """Process a single frame using all CNN models"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        drowsiness_detected = False
        yawn_detected = False
        ear_value = 0.0
        mar_value = 0.0
        cnn_confidence = 0.0
        sequence_confidence = 0.0
        face_confidence = 0.0
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for face in faces:
            x, y, w, h = face
            face_roi = frame[y:y+h, x:x+w]
            
            # Extract eye region for CNN analysis
            eye_region = self._extract_eye_region(frame, face)
            
            # CNN-based eye state prediction
            if eye_region is not None:
                open_prob, closed_prob = self._predict_eye_state_cnn(eye_region)
                cnn_confidence = closed_prob  # Higher closed probability = more drowsy
                
                # Add to history
                self.eye_state_history.append(closed_prob)
            
            # Calculate traditional EAR and MAR
            ear_value = self._calculate_ear_from_eyes(face_roi)
            mar_value = self._calculate_mar_from_face(face_roi)
            
            # Sequence-based drowsiness prediction
            sequence_confidence = self._predict_drowsiness_sequence(ear_value, mar_value)
            
            # Face analysis
            face_confidence = self._predict_face_analysis(face_roi)
            
            # Combined decision making
            combined_confidence = (
                cnn_confidence * 0.4 +           # CNN eye state
                sequence_confidence * 0.3 +       # Temporal sequence
                face_confidence * 0.3             # Face analysis
            )
            
            # Determine drowsiness
            if combined_confidence > 0.7:
                drowsiness_detected = True
            
            # Determine yawning
            if mar_value > self.MAR_THRESHOLD:
                yawn_detected = True
            
            # Draw visualizations
            self._draw_analysis_overlay(frame, face, eye_region, 
                                      cnn_confidence, sequence_confidence, 
                                      face_confidence, combined_confidence)
        
        return frame, drowsiness_detected, yawn_detected, ear_value, mar_value, combined_confidence
    
    def _calculate_ear_from_eyes(self, face_roi):
        """Calculate EAR from detected eyes"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(eyes) >= 2:
            # Use the two largest eyes
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            
            ear_values = []
            for (ex, ey, ew, eh) in eyes:
                # Create eye landmarks
                eye_landmarks = [
                    np.array([ex, ey + eh//2]),           # Left corner
                    np.array([ex + ew//4, ey]),           # Top left
                    np.array([ex + ew//2, ey]),           # Top center
                    np.array([ex + 3*ew//4, ey]),         # Top right
                    np.array([ex + ew, ey + eh//2]),      # Right corner
                    np.array([ex + ew//2, ey + eh])       # Bottom center
                ]
                
                ear = self._calculate_ear(eye_landmarks)
                ear_values.append(ear)
            
            return np.mean(ear_values) if ear_values else 0.0
        
        return 0.0
    
    def _calculate_mar_from_face(self, face_roi):
        """Calculate MAR from face region"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Estimate mouth position (lower third of face)
        mouth_y = int(2.5 * h / 3)
        mouth_x = w // 2
        mouth_w = int(w * 0.4)
        mouth_h = int(h * 0.15)
        
        # Create mouth landmarks
        mouth_landmarks = [
            np.array([mouth_x - mouth_w//2, mouth_y]),           # Left corner
            np.array([mouth_x - mouth_w//4, mouth_y - mouth_h//2]),  # Top left
            np.array([mouth_x, mouth_y - mouth_h//2]),           # Top center
            np.array([mouth_x + mouth_w//4, mouth_y - mouth_h//2]),  # Top right
            np.array([mouth_x + mouth_w//2, mouth_y]),           # Right corner
            np.array([mouth_x + mouth_w//4, mouth_y + mouth_h//2]),  # Bottom right
            np.array([mouth_x, mouth_y + mouth_h//2]),           # Bottom center
            np.array([mouth_x - mouth_w//4, mouth_y + mouth_h//2]),  # Bottom left
            np.array([mouth_x, mouth_y])                         # Center
        ]
        
        return self._calculate_mar(mouth_landmarks)
    
    def _draw_analysis_overlay(self, frame, face, eye_region, cnn_conf, seq_conf, face_conf, combined_conf):
        """Draw analysis overlay on frame"""
        x, y, w, h = face
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw eye region if available
        if eye_region is not None:
            eye_h, eye_w = eye_region.shape[:2]
            eye_display = cv2.resize(eye_region, (eye_w*2, eye_h*2))
            if len(eye_display.shape) == 3:
                eye_display = cv2.cvtColor(eye_display, cv2.COLOR_GRAY2BGR)
            else:
                eye_display = cv2.cvtColor(eye_display, cv2.COLOR_GRAY2BGR)
            
            # Place eye display in top-right corner
            frame[10:10+eye_h*2, frame.shape[1]-eye_w*2-10:frame.shape[1]-10] = eye_display
        
        # Draw confidence bars
        bar_width = 200
        bar_height = 20
        start_x = 10
        start_y = 10
        
        # CNN confidence bar
        cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x, start_y), (start_x + int(bar_width * cnn_conf), start_y + bar_height), (0, 255, 0), -1)
        cv2.putText(frame, f"CNN: {cnn_conf:.2f}", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Sequence confidence bar
        start_y += 30
        cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x, start_y), (start_x + int(bar_width * seq_conf), start_y + bar_height), (255, 0, 0), -1)
        cv2.putText(frame, f"Seq: {seq_conf:.2f}", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Face confidence bar
        start_y += 30
        cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x, start_y), (start_x + int(bar_width * face_conf), start_y + bar_height), (0, 0, 255), -1)
        cv2.putText(frame, f"Face: {face_conf:.2f}", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combined confidence bar
        start_y += 30
        cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x, start_y), (start_x + int(bar_width * combined_conf), start_y + bar_height), (255, 255, 0), -1)
        cv2.putText(frame, f"Combined: {combined_conf:.2f}", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self, camera_index=0):
        """Main application loop"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting Advanced CNN Drowsiness Detection System...")
        logger.info("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            # Process frame
            processed_frame, drowsiness, yawn, ear, mar, confidence = self.process_frame(frame)
            
            # Update FPS
            self.update_fps()
            
            # Handle alerts
            current_time = time.time()
            if drowsiness or yawn:
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = current_time
                    logger.warning(f"DROWSINESS DETECTED! Confidence: {confidence:.2f}")
            else:
                self.alert_active = False
            
            # Check if alert should stop
            if self.alert_active and (current_time - self.alert_start_time) > self.ALERT_DURATION:
                self.alert_active = False
            
            # Draw status information
            status_color = (0, 0, 255) if (drowsiness or yawn) else (0, 255, 0)
            status_text = "DROWSY!" if (drowsiness or yawn) else "AWAKE"
            
            cv2.putText(processed_frame, status_text, (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(processed_frame, f"EAR: {ear:.3f}", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"MAR: {mar:.3f}", (10, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"FPS: {self.current_fps}", (10, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Confidence: {confidence:.2f}", (10, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Advanced CNN Drowsiness Detection System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Advanced CNN drowsiness detection system stopped")

def main():
    """Main function"""
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow not available. Please install: pip install tensorflow")
        return
    
    detector = AdvancedCNNDrowsinessDetector()
    detector.run()

if __name__ == "__main__":
    main()
