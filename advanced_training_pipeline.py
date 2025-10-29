"""
Advanced Deep Learning Training Pipeline for Drowsiness Detection
Multi-model training with data augmentation, transfer learning, and ensemble methods
"""

import cv2
import numpy as np
import os
import random
import json
import time
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import logging

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                       Dropout, BatchNormalization, GlobalAveragePooling2D,
                                       Input, Concatenate, Reshape, LSTM, TimeDistributed,
                                       Attention, MultiHeadAttention, LayerNormalization)
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                          ModelCheckpoint, TensorBoard, CSVLogger)
    from tensorflow.keras.applications import (MobileNetV2, EfficientNetB0, 
                                             ResNet50, VGG16, DenseNet121)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical, plot_model
    from tensorflow.keras.regularizers import l1, l2
    from tensorflow.keras.metrics import Precision, Recall, AUC
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Please install: pip install tensorflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDataGenerator:
    """Advanced data generator with sophisticated augmentation techniques"""
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['alert', 'drowsy', 'normal']
        
        # Advanced augmentation parameters
        self.augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'brightness_range': [0.8, 1.2],
            'channel_shift_range': 0.1,
            'fill_mode': 'nearest'
        }
        
        self.datagen = ImageDataGenerator(**self.augmentation_params)
        self.val_datagen = ImageDataGenerator(rescale=1./255)
        
    def create_data_generators(self, validation_split=0.2):
        """Create training and validation data generators"""
        train_generator = self.datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            validation_split=validation_split,
            shuffle=True
        )
        
        val_generator = self.val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            validation_split=validation_split,
            shuffle=False
        )
        
        return train_generator, val_generator

class MultiModelEnsemble:
    """Ensemble of multiple CNN models for robust prediction"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        self.weights = {}
        
    def create_efficientnet_model(self):
        """Create EfficientNet-based model"""
        base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Fine-tune last few layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
        
        return model
    
    def create_resnet_model(self):
        """Create ResNet-based model"""
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = True
        for layer in base_model.layers[:-15]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=RMSprop(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
        
        return model
    
    def create_custom_cnn_model(self):
        """Create custom CNN architecture"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            # First block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fifth block
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Dense layers
            Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
        
        return model
    
    def create_attention_model(self):
        """Create attention-based model"""
        inputs = Input(shape=self.input_shape)
        
        # CNN backbone
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Reshape for attention
        x = Reshape((28*28, 256))(x)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8, 
            key_dim=64
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization()(x + attention_output)
        
        # Global average pooling
        x = GlobalAveragePooling2D()(Reshape((28, 28, 256))(x))
        
        # Dense layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
        
        return model

class AdvancedTrainingPipeline:
    """Advanced training pipeline with cross-validation and ensemble methods"""
    
    def __init__(self, data_dir, output_dir="models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize ensemble
        self.ensemble = MultiModelEnsemble()
        
    def setup_callbacks(self, model_name):
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f"{self.output_dir}/{model_name}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=f"logs/{model_name}",
                histogram_freq=1,
                write_graph=True
            ),
            CSVLogger(
                filename=f"{self.output_dir}/{model_name}_training.log"
            )
        ]
        
        return callbacks
    
    def train_single_model(self, model, model_name, train_gen, val_gen, epochs=100):
        """Train a single model"""
        logger.info(f"Training {model_name}...")
        
        callbacks = self.setup_callbacks(model_name)
        
        # Train model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model.save(f"{self.output_dir}/{model_name}_final.h5")
        
        # Evaluate model
        val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(val_gen, verbose=0)
        
        # Store results
        self.results[model_name] = {
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_auc': val_auc,
            'history': history.history
        }
        
        logger.info(f"{model_name} - Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        return model, history
    
    def train_ensemble(self, train_gen, val_gen, epochs=100):
        """Train ensemble of models"""
        logger.info("Training ensemble of models...")
        
        # Create models
        models = {
            'efficientnet': self.ensemble.create_efficientnet_model(),
            'resnet': self.ensemble.create_resnet_model(),
            'custom_cnn': self.ensemble.create_custom_cnn_model(),
            'attention': self.ensemble.create_attention_model()
        }
        
        # Train each model
        for model_name, model in models.items():
            trained_model, history = self.train_single_model(
                model, model_name, train_gen, val_gen, epochs
            )
            self.ensemble.models[model_name] = trained_model
        
        # Calculate ensemble weights based on validation performance
        self.calculate_ensemble_weights()
        
        return self.ensemble
    
    def calculate_ensemble_weights(self):
        """Calculate optimal weights for ensemble"""
        total_auc = sum(result['val_auc'] for result in self.results.values())
        
        for model_name, result in self.results.items():
            self.ensemble.weights[model_name] = result['val_auc'] / total_auc
        
        logger.info("Ensemble weights calculated:")
        for model_name, weight in self.ensemble.weights.items():
            logger.info(f"  {model_name}: {weight:.4f}")
    
    def evaluate_ensemble(self, test_gen):
        """Evaluate ensemble performance"""
        logger.info("Evaluating ensemble...")
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.ensemble.models.items():
            pred = model.predict(test_gen, verbose=0)
            predictions[model_name] = pred
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for model_name, pred in predictions.items():
            ensemble_pred += pred * self.ensemble.weights[model_name]
        
        # Get true labels
        true_labels = test_gen.classes
        predicted_labels = np.argmax(ensemble_pred, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_labels == true_labels)
        
        # Classification report
        class_names = ['alert', 'drowsy', 'normal']
        report = classification_report(true_labels, predicted_labels, 
                                    target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Store ensemble results
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        
        return ensemble_pred, true_labels
    
    def generate_visualizations(self):
        """Generate training visualizations"""
        logger.info("Generating visualizations...")
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (model_name, result) in enumerate(self.results.items()):
            if 'history' in result:
                history = result['history']
                
                # Plot accuracy
                axes[0, 0].plot(history['accuracy'], label=f'{model_name}_train')
                axes[0, 0].plot(history['val_accuracy'], label=f'{model_name}_val')
                
                # Plot loss
                axes[0, 1].plot(history['loss'], label=f'{model_name}_train')
                axes[0, 1].plot(history['val_loss'], label=f'{model_name}_val')
        
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot model comparison
        model_names = list(self.results.keys())
        accuracies = [result.get('val_accuracy', 0) for result in self.results.values()]
        
        axes[1, 0].bar(model_names, accuracies)
        axes[1, 0].set_title('Model Comparison - Validation Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot ensemble weights
        if 'ensemble' in self.results:
            ensemble_models = list(self.ensemble.weights.keys())
            ensemble_weights = list(self.ensemble.weights.values())
            
            axes[1, 1].bar(ensemble_models, ensemble_weights)
            axes[1, 1].set_title('Ensemble Weights')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrix
        if 'ensemble' in self.results:
            cm = np.array(self.results['ensemble']['confusion_matrix'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['alert', 'drowsy', 'normal'],
                       yticklabels=['alert', 'drowsy', 'normal'])
            plt.title('Ensemble Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self):
        """Save training results"""
        logger.info("Saving results...")
        
        # Save results as JSON
        with open(f"{self.output_dir}/training_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model_name, result in self.results.items():
                json_results[model_name] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_results[model_name][key] = value.tolist()
                    else:
                        json_results[model_name][key] = value
            
            json.dump(json_results, f, indent=2)
        
        # Save model architecture diagrams
        for model_name, model in self.ensemble.models.items():
            try:
                plot_model(model, to_file=f"{self.output_dir}/{model_name}_architecture.png", 
                          show_shapes=True, show_layer_names=True)
            except Exception as e:
                logger.warning(f"Could not save architecture for {model_name}: {e}")
    
    def run_training_pipeline(self, epochs=100):
        """Run the complete training pipeline"""
        logger.info("Starting Advanced Training Pipeline...")
        logger.info("=" * 60)
        
        # Create data generator
        data_gen = AdvancedDataGenerator(self.data_dir)
        train_gen, val_gen = data_gen.create_data_generators()
        
        # Train ensemble
        ensemble = self.train_ensemble(train_gen, val_gen, epochs)
        
        # Evaluate ensemble
        ensemble_pred, true_labels = self.evaluate_ensemble(val_gen)
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")

def create_sample_dataset():
    """Create sample dataset for demonstration"""
    logger.info("Creating sample dataset...")
    
    # Create directories
    os.makedirs("dataset/alert", exist_ok=True)
    os.makedirs("dataset/drowsy", exist_ok=True)
    os.makedirs("dataset/normal", exist_ok=True)
    
    # Create synthetic images
    for class_name in ['alert', 'drowsy', 'normal']:
        for i in range(100):  # 100 images per class
            # Create synthetic face image
            img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            
            # Add class-specific features
            if class_name == 'alert':
                # Wide open eyes
                cv2.ellipse(img, (80, 100), (15, 25), 0, 0, 360, (255, 255, 255), -1)
                cv2.ellipse(img, (144, 100), (15, 25), 0, 0, 360, (255, 255, 255), -1)
            elif class_name == 'drowsy':
                # Half-closed eyes
                cv2.ellipse(img, (80, 100), (15, 10), 0, 0, 360, (100, 100, 100), -1)
                cv2.ellipse(img, (144, 100), (15, 10), 0, 0, 360, (100, 100, 100), -1)
            else:  # normal
                # Normal eyes
                cv2.ellipse(img, (80, 100), (15, 20), 0, 0, 360, (200, 200, 200), -1)
                cv2.ellipse(img, (144, 100), (15, 20), 0, 0, 360, (200, 200, 200), -1)
            
            # Save image
            img_path = f"dataset/{class_name}/{class_name}_{i:03d}.png"
            cv2.imwrite(img_path, img)
    
    logger.info("Sample dataset created successfully!")

def main():
    """Main function"""
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow not available. Please install: pip install tensorflow")
        return
    
    # Create sample dataset if it doesn't exist
    if not os.path.exists("dataset"):
        create_sample_dataset()
    
    # Run training pipeline
    pipeline = AdvancedTrainingPipeline("dataset")
    pipeline.run_training_pipeline(epochs=50)  # Reduced for demo

if __name__ == "__main__":
    main()
