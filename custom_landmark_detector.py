"""
Custom Facial Landmark Detection System
Uses OpenCV's built-in face detection and geometric calculations
instead of relying on dlib's pre-trained model
"""

import cv2
import numpy as np
from scipy.spatial import distance
import logging

logger = logging.getLogger(__name__)

class CustomLandmarkDetector:
    def __init__(self):
        """Initialize the custom landmark detector"""
        # Load OpenCV's face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_face(self, gray_image):
        """Detect faces in the image"""
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def detect_eyes(self, gray_image, face_roi):
        """Detect eyes within a face region"""
        x, y, w, h = face_roi
        face_gray = gray_image[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # Convert to absolute coordinates
        eyes_absolute = []
        for (ex, ey, ew, eh) in eyes:
            eyes_absolute.append((x + ex, y + ey, ew, eh))
        
        return eyes_absolute
    
    def get_eye_landmarks(self, eye_roi):
        """Extract eye landmarks from eye region"""
        x, y, w, h = eye_roi
        center_x, center_y = x + w//2, y + h//2
        
        # Create 6 landmark points around the eye
        landmarks = []
        
        # Left corner
        landmarks.append((x, center_y))
        # Top left
        landmarks.append((x + w//4, y))
        # Top center
        landmarks.append((center_x, y))
        # Top right
        landmarks.append((x + 3*w//4, y))
        # Right corner
        landmarks.append((x + w, center_y))
        # Bottom center
        landmarks.append((center_x, y + h))
        
        return landmarks
    
    def get_mouth_landmarks(self, face_roi):
        """Estimate mouth landmarks from face region"""
        x, y, w, h = face_roi
        center_x, center_y = x + w//2, y + h//2
        
        # Estimate mouth position (typically in lower third of face)
        mouth_y = y + int(2.5 * h / 3)
        mouth_width = int(w * 0.4)
        mouth_height = int(h * 0.15)
        
        mouth_x = center_x - mouth_width//2
        
        # Create 9 landmark points around the mouth
        landmarks = []
        
        # Left corner
        landmarks.append((mouth_x, mouth_y))
        # Top left
        landmarks.append((mouth_x + mouth_width//4, mouth_y - mouth_height//2))
        # Top center
        landmarks.append((center_x, mouth_y - mouth_height//2))
        # Top right
        landmarks.append((mouth_x + 3*mouth_width//4, mouth_y - mouth_height//2))
        # Right corner
        landmarks.append((mouth_x + mouth_width, mouth_y))
        # Bottom right
        landmarks.append((mouth_x + 3*mouth_width//4, mouth_y + mouth_height//2))
        # Bottom center
        landmarks.append((center_x, mouth_y + mouth_height//2))
        # Bottom left
        landmarks.append((mouth_x + mouth_width//4, mouth_y + mouth_height//2))
        # Center
        landmarks.append((center_x, mouth_y))
        
        return landmarks

class CustomDrowsinessDetector:
    def __init__(self):
        """Initialize the custom drowsiness detector"""
        self.landmark_detector = CustomLandmarkDetector()
        
        # Detection parameters
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.6
        self.CONSECUTIVE_FRAMES = 15
        self.ALERT_DURATION = 3.0
        
        # State tracking
        self.ear_history = []
        self.mar_history = []
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.alert_active = False
        self.alert_start_time = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = 0
        self.current_fps = 0
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        try:
            if len(eye_landmarks) < 6:
                return 0.0
                
            # Extract points
            p1 = np.array(eye_landmarks[0])  # Left corner
            p2 = np.array(eye_landmarks[1])  # Top left
            p3 = np.array(eye_landmarks[2])  # Top center
            p4 = np.array(eye_landmarks[3])  # Top right
            p5 = np.array(eye_landmarks[4])  # Right corner
            p6 = np.array(eye_landmarks[5])  # Bottom center
            
            # Calculate distances
            vertical_1 = distance.euclidean(p2, p6)
            vertical_2 = distance.euclidean(p3, p5)
            horizontal = distance.euclidean(p1, p4)
            
            # Calculate EAR
            if horizontal == 0:
                return 0.0
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0
    
    def calculate_mar(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio (MAR)"""
        try:
            if len(mouth_landmarks) < 9:
                return 0.0
                
            # Extract key points for MAR calculation
            p1 = np.array(mouth_landmarks[0])  # Left corner
            p3 = np.array(mouth_landmarks[2])  # Top center
            p7 = np.array(mouth_landmarks[6])  # Bottom center
            p9 = np.array(mouth_landmarks[8])  # Center
            
            # Calculate distances
            vertical = distance.euclidean(p3, p7)
            horizontal = distance.euclidean(p1, p9)
            
            # Calculate MAR
            if horizontal == 0:
                return 0.0
            mar = vertical / horizontal
            return mar
        except Exception as e:
            logger.error(f"Error calculating MAR: {e}")
            return 0.0
    
    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        drowsiness_detected = False
        yawn_detected = False
        ear_value = 0.0
        mar_value = 0.0
        
        # Detect faces
        faces = self.landmark_detector.detect_face(gray)
        
        for face in faces:
            x, y, w, h = face
            
            # Detect eyes
            eyes = self.landmark_detector.detect_eyes(gray, face)
            
            if len(eyes) >= 2:
                # Use the two largest eyes
                eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                
                # Calculate EAR for each eye
                ear_values = []
                for eye in eyes:
                    eye_landmarks = self.landmark_detector.get_eye_landmarks(eye)
                    ear = self.calculate_ear(eye_landmarks)
                    ear_values.append(ear)
                
                # Average EAR
                if ear_values:
                    ear_value = np.mean(ear_values)
                    self.ear_history.append(ear_value)
                    
                    # Keep only recent history
                    if len(self.ear_history) > self.CONSECUTIVE_FRAMES:
                        self.ear_history.pop(0)
                    
                    # Check for drowsiness
                    if len(self.ear_history) >= self.CONSECUTIVE_FRAMES:
                        avg_ear = np.mean(self.ear_history)
                        if avg_ear < self.EAR_THRESHOLD:
                            self.eye_closed_frames += 1
                            if self.eye_closed_frames >= self.CONSECUTIVE_FRAMES:
                                drowsiness_detected = True
                        else:
                            self.eye_closed_frames = 0
            
            # Detect mouth and calculate MAR
            mouth_landmarks = self.landmark_detector.get_mouth_landmarks(face)
            mar_value = self.calculate_mar(mouth_landmarks)
            
            if mar_value > 0:
                self.mar_history.append(mar_value)
                
                # Keep only recent history
                if len(self.mar_history) > self.CONSECUTIVE_FRAMES:
                    self.mar_history.pop(0)
                
                # Check for yawning
                if len(self.mar_history) >= self.CONSECUTIVE_FRAMES:
                    avg_mar = np.mean(self.mar_history)
                    if avg_mar > self.MAR_THRESHOLD:
                        self.yawn_frames += 1
                        if self.yawn_frames >= self.CONSECUTIVE_FRAMES:
                            yawn_detected = True
                    else:
                        self.yawn_frames = 0
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw eyes
            for eye in eyes:
                ex, ey, ew, eh = eye
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Draw mouth region
            mouth_x = x + w//2 - int(w * 0.2)
            mouth_y = y + int(2.5 * h / 3)
            mouth_w = int(w * 0.4)
            mouth_h = int(h * 0.15)
            cv2.rectangle(frame, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (0, 0, 255), 2)
        
        return frame, drowsiness_detected, yawn_detected, ear_value, mar_value
    
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
        
        logger.info("Starting custom drowsiness detection system...")
        logger.info("Press 'q' to quit")
        
        self.fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
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
                    logger.warning("DROWSINESS DETECTED!")
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
            cv2.imshow('Custom Drowsiness Detection System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Custom drowsiness detection system stopped")

if __name__ == "__main__":
    import time
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    detector = CustomDrowsinessDetector()
    detector.run()
