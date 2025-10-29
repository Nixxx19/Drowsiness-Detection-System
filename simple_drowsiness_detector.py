"""
Simple Drowsiness Detection System
Uses only OpenCV's built-in face and eye detection
No external dependencies required beyond OpenCV
"""

import cv2
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDrowsinessDetector:
    def __init__(self):
        """Initialize the simple drowsiness detector"""
        # Load OpenCV's cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Detection parameters
        self.EYE_CLOSED_THRESHOLD = 0.3  # Ratio of eye area to face area
        self.CONSECUTIVE_FRAMES = 10     # Frames to confirm drowsiness
        self.ALERT_DURATION = 3.0        # Alert duration in seconds
        
        # State tracking
        self.eye_closed_frames = 0
        self.alert_active = False
        self.alert_start_time = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def calculate_eye_ratio(self, face_roi, eyes):
        """Calculate the ratio of eye area to face area"""
        if len(eyes) == 0:
            return 0.0
        
        face_area = face_roi[2] * face_roi[3]
        total_eye_area = 0
        
        for eye in eyes:
            eye_area = eye[2] * eye[3]
            total_eye_area += eye_area
        
        return total_eye_area / face_area if face_area > 0 else 0.0
    
    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        drowsiness_detected = False
        eye_ratio = 0.0
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for face in faces:
            x, y, w, h = face
            
            # Detect eyes within the face region
            face_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
            
            # Convert eye coordinates to absolute coordinates
            eyes_absolute = []
            for (ex, ey, ew, eh) in eyes:
                eyes_absolute.append((x + ex, y + ey, ew, eh))
            
            # Calculate eye ratio
            eye_ratio = self.calculate_eye_ratio(face, eyes_absolute)
            
            # Check for drowsiness (low eye ratio indicates closed eyes)
            if eye_ratio < self.EYE_CLOSED_THRESHOLD:
                self.eye_closed_frames += 1
                if self.eye_closed_frames >= self.CONSECUTIVE_FRAMES:
                    drowsiness_detected = True
            else:
                self.eye_closed_frames = 0
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw eyes
            for eye in eyes_absolute:
                ex, ey, ew, eh = eye
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Draw eye ratio text
            cv2.putText(frame, f"Eye Ratio: {eye_ratio:.3f}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, drowsiness_detected, eye_ratio
    
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
        
        logger.info("Starting simple drowsiness detection system...")
        logger.info("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            # Process frame
            processed_frame, drowsiness, eye_ratio = self.process_frame(frame)
            
            # Update FPS
            self.update_fps()
            
            # Handle alerts
            current_time = time.time()
            if drowsiness:
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = current_time
                    logger.warning("DROWSINESS DETECTED!")
            else:
                self.alert_active = False
            
            # Check if alert should stop
            if self.alert_active and (current_time - self.alert_start_time) > self.ALERT_DURATION:
                self.alert_active = False
            
            # Draw status information
            status_color = (0, 0, 255) if drowsiness else (0, 255, 0)
            status_text = "DROWSY!" if drowsiness else "AWAKE"
            
            cv2.putText(processed_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(processed_frame, f"Eye Ratio: {eye_ratio:.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"FPS: {self.current_fps}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Closed Frames: {self.eye_closed_frames}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Simple Drowsiness Detection System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Simple drowsiness detection system stopped")

def main():
    """Main function"""
    detector = SimpleDrowsinessDetector()
    detector.run()

if __name__ == "__main__":
    main()
