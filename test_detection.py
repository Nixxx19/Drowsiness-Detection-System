"""
Test script for the drowsiness detection system
"""

import cv2
import numpy as np
import time
from drowsiness_detector import DrowsinessDetector

def test_ear_calculation():
    """Test EAR calculation with known values"""
    print("Testing EAR calculation...")
    
    detector = DrowsinessDetector()
    
    # Create mock eye landmarks for a closed eye (should have low EAR)
    closed_eye_landmarks = [
        type('Point', (), {'x': 0, 'y': 0})(),  # p1
        type('Point', (), {'x': 10, 'y': 5})(),  # p2
        type('Point', (), {'x': 20, 'y': 5})(),  # p3
        type('Point', (), {'x': 30, 'y': 0})(),  # p4
        type('Point', (), {'x': 20, 'y': -5})(), # p5
        type('Point', (), {'x': 10, 'y': -5})(), # p6
    ]
    
    ear = detector.calculate_ear(closed_eye_landmarks)
    print(f"Closed eye EAR: {ear:.3f}")
    
    # Create mock eye landmarks for an open eye (should have higher EAR)
    open_eye_landmarks = [
        type('Point', (), {'x': 0, 'y': 0})(),   # p1
        type('Point', (), {'x': 10, 'y': 10})(), # p2
        type('Point', (), {'x': 20, 'y': 10})(), # p3
        type('Point', (), {'x': 30, 'y': 0})(),  # p4
        type('Point', (), {'x': 20, 'y': -10})(),# p5
        type('Point', (), {'x': 10, 'y': -10})(),# p6
    ]
    
    ear = detector.calculate_ear(open_eye_landmarks)
    print(f"Open eye EAR: {ear:.3f}")
    
    print("✓ EAR calculation test completed\n")

def test_mar_calculation():
    """Test MAR calculation with known values"""
    print("Testing MAR calculation...")
    
    detector = DrowsinessDetector()
    
    # Create mock mouth landmarks for closed mouth (should have low MAR)
    closed_mouth_landmarks = [
        type('Point', (), {'x': 0, 'y': 0})(),   # p1
        type('Point', (), {'x': 5, 'y': 2})(),   # p2
        type('Point', (), {'x': 10, 'y': 0})(),  # p3
        type('Point', (), {'x': 15, 'y': 2})(),  # p4
        type('Point', (), {'x': 20, 'y': 0})(),  # p5
        type('Point', (), {'x': 15, 'y': -2})(), # p6
        type('Point', (), {'x': 20, 'y': 0})(),  # p7
        type('Point', (), {'x': 15, 'y': 2})(),  # p8
        type('Point', (), {'x': 10, 'y': 0})(),  # p9
    ]
    
    mar = detector.calculate_mar(closed_mouth_landmarks)
    print(f"Closed mouth MAR: {mar:.3f}")
    
    # Create mock mouth landmarks for open mouth (should have higher MAR)
    open_mouth_landmarks = [
        type('Point', (), {'x': 0, 'y': 0})(),    # p1
        type('Point', (), {'x': 5, 'y': 2})(),    # p2
        type('Point', (), {'x': 10, 'y': 0})(),   # p3
        type('Point', (), {'x': 15, 'y': 2})(),   # p4
        type('Point', (), {'x': 20, 'y': 0})(),   # p5
        type('Point', (), {'x': 15, 'y': -2})(),  # p6
        type('Point', (), {'x': 20, 'y': 0})(),   # p7
        type('Point', (), {'x': 15, 'y': 2})(),   # p8
        type('Point', (), {'x': 10, 'y': 10})(),  # p9 (open mouth)
    ]
    
    mar = detector.calculate_mar(open_mouth_landmarks)
    print(f"Open mouth MAR: {mar:.3f}")
    
    print("✓ MAR calculation test completed\n")

def test_camera():
    """Test camera functionality"""
    print("Testing camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open camera")
        return False
    
    print("✓ Camera opened successfully")
    
    # Test reading a few frames
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: {frame.shape}")
        else:
            print(f"✗ Failed to read frame {i+1}")
            cap.release()
            return False
    
    cap.release()
    print("✓ Camera test completed\n")
    return True

def test_performance():
    """Test system performance"""
    print("Testing performance...")
    
    detector = DrowsinessDetector()
    
    # Test with a dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Measure processing time
    start_time = time.time()
    for i in range(100):
        processed_frame, drowsiness, yawn, ear, mar = detector.process_frame(dummy_frame)
    
    end_time = time.time()
    processing_time = (end_time - start_time) / 100
    fps = 1.0 / processing_time
    
    print(f"Average processing time per frame: {processing_time*1000:.2f}ms")
    print(f"Estimated FPS: {fps:.1f}")
    
    if fps >= 10:
        print("✓ Performance test passed (≥10 FPS)")
    else:
        print("⚠ Performance test warning (<10 FPS)")
    
    print("✓ Performance test completed\n")

def main():
    """Run all tests"""
    print("Running Drowsiness Detection System Tests")
    print("=" * 50)
    
    # Test individual components
    test_ear_calculation()
    test_mar_calculation()
    
    # Test camera
    if not test_camera():
        print("Camera test failed. Make sure a camera is connected.")
        return
    
    # Test performance
    test_performance()
    
    print("=" * 50)
    print("✓ All tests completed!")
    print("\nTo run the full system:")
    print("python drowsiness_detector.py")

if __name__ == "__main__":
    main()
