"""
CNN Model Training for Drowsiness Detection
Trains a custom CNN model to classify eye states (open/closed)
"""

import cv2
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import logging

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Please install: pip install tensorflow")

# Try to import PyTorch as alternative
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Please install: pip install torch torchvision")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeStateDataset:
    """Dataset class for eye state classification"""
    
    def __init__(self, data_dir, img_size=(64, 64)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.class_names = ['closed', 'open']
        
    def load_data(self):
        """Load and preprocess the dataset"""
        logger.info("Loading dataset...")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory {class_dir} not found. Creating sample data...")
                self.create_sample_data(class_dir, class_name, class_idx)
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, self.img_size)
                            img = img.astype(np.float32) / 255.0
                            self.images.append(img)
                            self.labels.append(class_idx)
                    except Exception as e:
                        logger.error(f"Error loading {img_path}: {e}")
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        logger.info(f"Loaded {len(self.images)} images")
        logger.info(f"Class distribution: {np.bincount(self.labels)}")
        
        return self.images, self.labels
    
    def create_sample_data(self, class_dir, class_name, class_idx):
        """Create sample data for demonstration"""
        os.makedirs(class_dir, exist_ok=True)
        
        # Create synthetic eye images
        for i in range(50):  # Create 50 sample images
            if class_name == 'open':
                # Create open eye pattern
                img = np.random.randint(50, 100, (64, 64), dtype=np.uint8)
                # Add eye shape
                cv2.ellipse(img, (32, 32), (20, 12), 0, 0, 360, 0, -1)
                cv2.ellipse(img, (32, 32), (15, 8), 0, 0, 360, 255, -1)
            else:  # closed
                # Create closed eye pattern
                img = np.random.randint(50, 100, (64, 64), dtype=np.uint8)
                # Add closed eye shape
                cv2.line(img, (12, 32), (52, 32), 0, 3)
            
            img_path = os.path.join(class_dir, f"{class_name}_{i:03d}.png")
            cv2.imwrite(img_path, img)

class CNNEyeStateClassifier:
    """CNN model for eye state classification"""
    
    def __init__(self, input_shape=(64, 64, 1), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build the CNN model architecture"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot build model.")
            return None
            
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the CNN model"""
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            return None
            
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        # Reshape data for CNN
        X_train = X_train.reshape(-1, *self.input_shape)
        X_val = X_val.reshape(-1, *self.input_shape)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
            
        X_test = X_test.reshape(-1, *self.input_shape)
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        # Evaluate
        test_loss, test_acc = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Predictions
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Classification report
        class_names = ['closed', 'open']
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        return test_loss, test_acc, report, y_pred
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            logger.error("Model not trained. Cannot save.")
            return False
            
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot load model.")
            return False
            
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, image):
        """Predict eye state for a single image"""
        if self.model is None:
            logger.error("Model not loaded. Cannot predict.")
            return None
            
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float32) / 255.0
        image = image.reshape(1, 64, 64, 1)
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        
        return class_idx, confidence

def create_training_data():
    """Create training data by capturing eye images"""
    logger.info("Creating training data...")
    
    # Create data directories
    os.makedirs("training_data/closed", exist_ok=True)
    os.makedirs("training_data/open", exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return False
    
    # Load face and eye cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    logger.info("Press 'o' to capture open eye, 'c' to capture closed eye, 'q' to quit")
    
    closed_count = 0
    open_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Detect eyes
            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                
                # Extract eye region
                eye_img = face_gray[ey:ey+eh, ex:ex+ew]
                if eye_img.size > 0:
                    eye_img = cv2.resize(eye_img, (64, 64))
                    
                    # Show eye region
                    cv2.imshow('Eye Region', eye_img)
        
        # Show frame
        cv2.putText(frame, f"Open: {open_count}, Closed: {closed_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Training Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o') and len(eyes) > 0:
            # Save open eye
            eye_img = face_gray[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
            if eye_img.size > 0:
                eye_img = cv2.resize(eye_img, (64, 64))
                cv2.imwrite(f"training_data/open/open_{open_count:03d}.png", eye_img)
                open_count += 1
                logger.info(f"Saved open eye {open_count}")
        elif key == ord('c') and len(eyes) > 0:
            # Save closed eye
            eye_img = face_gray[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
            if eye_img.size > 0:
                eye_img = cv2.resize(eye_img, (64, 64))
                cv2.imwrite(f"training_data/closed/closed_{closed_count:03d}.png", eye_img)
                closed_count += 1
                logger.info(f"Saved closed eye {closed_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Training data collection complete. Open: {open_count}, Closed: {closed_count}")
    return True

def main():
    """Main training function"""
    logger.info("CNN Eye State Classification Training")
    logger.info("=" * 50)
    
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow not available. Please install: pip install tensorflow")
        return
    
    # Check if training data exists
    if not os.path.exists("training_data"):
        logger.info("No training data found. Creating sample data...")
        dataset = EyeStateDataset("training_data")
        dataset.load_data()
    else:
        logger.info("Loading existing training data...")
        dataset = EyeStateDataset("training_data")
        dataset.load_data()
    
    if len(dataset.images) == 0:
        logger.error("No training data available. Please create training data first.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.images, dataset.labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Build and train model
    classifier = CNNEyeStateClassifier()
    model = classifier.build_model()
    
    if model is None:
        logger.error("Failed to build model")
        return
    
    logger.info("Model architecture:")
    model.summary()
    
    # Train model
    logger.info("Starting training...")
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_acc, report, y_pred = classifier.evaluate(X_test, y_test)
    
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info("Classification Report:")
    logger.info(report)
    
    # Save model
    model_path = "eye_state_cnn_model.h5"
    classifier.save_model(model_path)
    
    logger.info("Training completed!")
    logger.info(f"Model saved as: {model_path}")

if __name__ == "__main__":
    main()
