#!/usr/bin/env python3
"""
Demo script for the Driver Drowsiness Detection System
Shows the system capabilities and provides interactive testing
"""

import cv2
import time
import numpy as np
from cnn_drowsiness_detector import AdvancedCNNDrowsinessDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_ear_calculation():
    """Demonstrate EAR calculation with visual examples"""
    print("\n" + "="*60)
    print("EAR (Eye Aspect Ratio) Calculation Demo")
    print("="*60)
    
    detector = AdvancedCNNDrowsinessDetector()
    
    # Simulate different eye states
    eye_states = [
        ("Fully Open Eye", [
            (0, 0), (10, 10), (20, 10), (30, 0), (20, -10), (10, -10)
        ]),
        ("Partially Closed Eye", [
            (0, 0), (10, 5), (20, 5), (30, 0), (20, -5), (10, -5)
        ]),
        ("Fully Closed Eye", [
            (0, 0), (10, 2), (20, 2), (30, 0), (20, -2), (10, -2)
        ])
    ]
    
    for state_name, landmarks in eye_states:
        # Convert to dlib point objects
        eye_landmarks = []
        for x, y in landmarks:
            point = type('Point', (), {'x': x, 'y': y})()
            eye_landmarks.append(point)
        
        ear = detector.calculate_ear(eye_landmarks)
        status = "OPEN" if ear > detector.EAR_THRESHOLD else "CLOSED"
        
        print(f"{state_name:20} | EAR: {ear:.3f} | Status: {status}")

def demo_mar_calculation():
    """Demonstrate MAR calculation with visual examples"""
    print("\n" + "="*60)
    print("MAR (Mouth Aspect Ratio) Calculation Demo")
    print("="*60)
    
    detector = AdvancedCNNDrowsinessDetector()
    
    # Simulate different mouth states
    mouth_states = [
        ("Closed Mouth", [
            (0, 0), (5, 2), (10, 0), (15, 2), (20, 0), (15, -2), (20, 0), (15, 2), (10, 0)
        ]),
        ("Slightly Open Mouth", [
            (0, 0), (5, 2), (10, 0), (15, 2), (20, 0), (15, -2), (20, 0), (15, 2), (10, 5)
        ]),
        ("Wide Open Mouth (Yawn)", [
            (0, 0), (5, 2), (10, 0), (15, 2), (20, 0), (15, -2), (20, 0), (15, 2), (10, 15)
        ])
    ]
    
    for state_name, landmarks in mouth_states:
        # Convert to dlib point objects
        mouth_landmarks = []
        for x, y in landmarks:
            point = type('Point', (), {'x': x, 'y': y})()
            mouth_landmarks.append(point)
        
        mar = detector.calculate_mar(mouth_landmarks)
        status = "YAWNING" if mar > detector.MAR_THRESHOLD else "NORMAL"
        
        print(f"{state_name:20} | MAR: {mar:.3f} | Status: {status}")

def demo_camera_test():
    """Test camera functionality"""
    print("\n" + "="*60)
    print("Camera Test Demo")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Test reading frames
    print("Testing frame capture...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1}: {frame.shape} - OK")
        else:
            print(f"  Frame {i+1}: Failed")
            cap.release()
            return False
    
    cap.release()
    print("✅ Camera test completed successfully")
    return True

def demo_performance_test():
    """Test system performance"""
    print("\n" + "="*60)
    print("Performance Test Demo")
    print("="*60)
    
    detector = AdvancedCNNDrowsinessDetector()
    
    # Create test frames of different sizes
    test_sizes = [(320, 240), (640, 480), (1280, 720)]
    
    for width, height in test_sizes:
        print(f"\nTesting {width}x{height} resolution:")
        
        # Create dummy frame
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Measure processing time
        start_time = time.time()
        iterations = 50
        
        for i in range(iterations):
            processed_frame, drowsiness, yawn, ear, mar = detector.process_frame(dummy_frame)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        fps = 1.0 / avg_time
        
        print(f"  Average processing time: {avg_time*1000:.2f}ms")
        print(f"  Estimated FPS: {fps:.1f}")
        
        if fps >= 10:
            print(f"  Status: ✅ Good performance (≥10 FPS)")
        else:
            print(f"  Status: ⚠️  Low performance (<10 FPS)")

def interactive_demo():
    """Interactive demo with real camera"""
    print("\n" + "="*60)
    print("Interactive Demo")
    print("="*60)
    print("This will start the actual drowsiness detection system.")
    print("Press 'q' to quit, 's' to toggle sound alerts")
    print("Try different facial expressions to see the system response!")
    
    response = input("\nStart interactive demo? (y/n): ").lower().strip()
    
    if response == 'y':
        detector = AdvancedCNNDrowsinessDetector()
        
        if not detector.load_landmark_predictor():
            print("❌ Could not load facial landmark model")
            print("Please run: python download_model.py")
            return
        
        print("\nStarting interactive demo...")
        detector.run()
    else:
        print("Interactive demo skipped.")

def main():
    """Main demo function"""
    print("Driver Drowsiness Detection System - Demo")
    print("="*60)
    print("This demo showcases the system's capabilities and algorithms.")
    
    # Run component demos
    demo_ear_calculation()
    demo_mar_calculation()
    
    # Test camera
    if demo_camera_test():
        # Test performance
        demo_performance_test()
        
        # Interactive demo
        interactive_demo()
    else:
        print("\n⚠️  Camera test failed. Please check your camera connection.")
        print("You can still run the performance test with dummy data.")
        
        response = input("Run performance test anyway? (y/n): ").lower().strip()
        if response == 'y':
            demo_performance_test()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("To run the full system: python run.py")

if __name__ == "__main__":
    main()
