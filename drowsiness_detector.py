"""
Driver Drowsiness Detection System
Main application for real-time drowsiness detection using facial landmarks
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import time
import threading
from collections import deque
import logging
import os
from config import *

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

class DrowsinessDetector:
    def __init__(self):
        """Initialize the drowsiness detection system"""
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = None
        self.cap = None
        
        # Detection parameters
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.6
        self.CONSECUTIVE_FRAMES = 15
        self.ALERT_DURATION = 3.0  # seconds
        
        # State tracking
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
        
        # Initialize pygame for audio alerts
        pygame.mixer.init()
        
    def load_landmark_predictor(self, predictor_path):
        """Load the facial landmark predictor model"""
        try:
            self.landmark_predictor = dlib.shape_predictor(predictor_path)
            logger.info("Facial landmark predictor loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load landmark predictor: {e}")
            return False
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        try:
            # Extract eye landmark coordinates
            p1 = np.array([eye_landmarks[0].x, eye_landmarks[0].y])
            p2 = np.array([eye_landmarks[1].x, eye_landmarks[1].y])
            p3 = np.array([eye_landmarks[2].x, eye_landmarks[2].y])
            p4 = np.array([eye_landmarks[3].x, eye_landmarks[3].y])
            p5 = np.array([eye_landmarks[4].x, eye_landmarks[4].y])
            p6 = np.array([eye_landmarks[5].x, eye_landmarks[5].y])
            
            # Calculate distances
            vertical_1 = distance.euclidean(p2, p6)
            vertical_2 = distance.euclidean(p3, p5)
            horizontal = distance.euclidean(p1, p4)
            
            # Calculate EAR
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0
    
    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR)
        MAR = |p3-p9| / |p1-p7|
        """
        try:
            # Extract mouth landmark coordinates
            p1 = np.array([mouth_landmarks[0].x, mouth_landmarks[0].y])
            p3 = np.array([mouth_landmarks[2].x, mouth_landmarks[2].y])
            p7 = np.array([mouth_landmarks[6].x, mouth_landmarks[6].y])
            p9 = np.array([mouth_landmarks[8].x, mouth_landmarks[8].y])
            
            # Calculate distances
            vertical = distance.euclidean(p3, p9)
            horizontal = distance.euclidean(p1, p7)
            
            # Calculate MAR
            mar = vertical / horizontal
            return mar
        except Exception as e:
            logger.error(f"Error calculating MAR: {e}")
            return 0.0
    
    def get_eye_landmarks(self, landmarks):
        """Extract eye landmarks from facial landmarks"""
        # Left eye landmarks (indices 36-41)
        left_eye = [landmarks[i] for i in range(36, 42)]
        # Right eye landmarks (indices 42-47)
        right_eye = [landmarks[i] for i in range(42, 48)]
        return left_eye, right_eye
    
    def get_mouth_landmarks(self, landmarks):
        """Extract mouth landmarks from facial landmarks"""
        # Mouth landmarks (indices 48-67)
        mouth = [landmarks[i] for i in range(48, 68)]
        return mouth
    
    def play_alert_sound(self):
        """Play alert sound in a separate thread"""
        def play_sound():
            try:
                # Generate a simple beep sound
                sample_rate = 22050
                duration = 0.5
                frequency = 800
                
                frames = int(duration * sample_rate)
                arr = np.zeros(frames)
                
                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
                
                # Convert to 16-bit integers
                arr = (arr * 32767).astype(np.int16)
                
                # Play the sound
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(duration)
            except Exception as e:
                logger.error(f"Error playing alert sound: {e}")
        
        # Play sound in a separate thread to avoid blocking
        sound_thread = threading.Thread(target=play_sound)
        sound_thread.daemon = True
        sound_thread.start()
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        drowsiness_detected = False
        yawn_detected = False
        ear_value = 0.0
        mar_value = 0.0
        
        for face in faces:
            if self.landmark_predictor is None:
                continue
                
            # Get facial landmarks
            landmarks = self.landmark_predictor(gray, face)
            landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            landmarks = [dlib.point(x, y) for x, y in landmarks]
            
            # Get eye and mouth landmarks
            left_eye, right_eye = self.get_eye_landmarks(landmarks)
            mouth = self.get_mouth_landmarks(landmarks)
            
            # Calculate EAR and MAR
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear_value = (left_ear + right_ear) / 2.0
            mar_value = self.calculate_mar(mouth)
            
            # Update history
            self.ear_history.append(ear_value)
            self.mar_history.append(mar_value)
            
            # Check for drowsiness
            if len(self.ear_history) >= self.CONSECUTIVE_FRAMES:
                avg_ear = np.mean(self.ear_history)
                if avg_ear < self.EAR_THRESHOLD:
                    self.eye_closed_frames += 1
                    if self.eye_closed_frames >= self.CONSECUTIVE_FRAMES:
                        drowsiness_detected = True
                else:
                    self.eye_closed_frames = 0
            
            # Check for yawning
            if len(self.mar_history) >= self.CONSECUTIVE_FRAMES:
                avg_mar = np.mean(self.mar_history)
                if avg_mar > self.MAR_THRESHOLD:
                    self.yawn_frames += 1
                    if self.yawn_frames >= self.CONSECUTIVE_FRAMES:
                        yawn_detected = True
                else:
                    self.yawn_frames = 0
            
            # Draw landmarks on frame
            for landmark in landmarks:
                cv2.circle(frame, (landmark.x, landmark.y), 1, (0, 255, 0), -1)
            
            # Draw face rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return frame, drowsiness_detected, yawn_detected, ear_value, mar_value
    
    def run(self, camera_index=0):
        """Main application loop"""
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            logger.error("Could not open camera")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting drowsiness detection system...")
        logger.info("Press 'q' to quit, 's' to toggle sound alerts")
        
        sound_enabled = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            # Process frame
            processed_frame, drowsiness, yawn, ear, mar = self.process_frame(frame)
            
            # Update FPS
            self.update_fps()
            
            # Handle alerts
            current_time = time.time()
            if drowsiness or yawn:
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = current_time
                    if sound_enabled:
                        self.play_alert_sound()
            else:
                self.alert_active = False
            
            # Check if alert should stop
            if self.alert_active and (current_time - self.alert_start_time) > self.ALERT_DURATION:
                self.alert_active = False
            
            # Draw status information
            status_color = (0, 0, 255) if (drowsiness or yawn) else (0, 255, 0)
            status_text = "DROWSY!" if (drowsiness or yawn) else "AWAKE"
            
            cv2.putText(processed_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(processed_frame, f"EAR: {ear:.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"MAR: {mar:.3f}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"FPS: {self.current_fps}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Drowsiness Detection System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sound_enabled = not sound_enabled
                logger.info(f"Sound alerts {'enabled' if sound_enabled else 'disabled'}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Drowsiness detection system stopped")

def main():
    """Main function"""
    detector = DrowsinessDetector()
    
    # Try to load the landmark predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not detector.load_landmark_predictor(predictor_path):
        logger.error("Please download the facial landmark predictor model:")
        logger.error("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        logger.error("bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # Start the detection system
    detector.run()

if __name__ == "__main__":
    main()
