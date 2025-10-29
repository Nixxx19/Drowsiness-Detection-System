"""
Dataset Downloader for Drowsiness Detection
Downloads real datasets from various sources and prepares them for training
"""

import os
import requests
import zipfile
import json
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Downloads and prepares drowsiness detection datasets"""
    
    def __init__(self, data_dir="dataset"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_huggingface_dataset(self):
        """Download dataset from Hugging Face"""
        logger.info("Downloading dataset from Hugging Face...")
        
        try:
            # Try to use huggingface_hub if available
            try:
                from huggingface_hub import hf_hub_download
                
                # Download the dataset
                dataset_path = hf_hub_download(
                    repo_id="akahana/Driver-Drowsiness-Dataset",
                    filename="data.zip",
                    repo_type="dataset"
                )
                
                # Extract the dataset
                self._extract_zip(dataset_path)
                logger.info("Hugging Face dataset downloaded successfully!")
                return True
                
            except ImportError:
                logger.warning("huggingface_hub not available. Installing...")
                os.system("pip install huggingface_hub")
                
                from huggingface_hub import hf_hub_download
                
                dataset_path = hf_hub_download(
                    repo_id="akahana/Driver-Drowsiness-Dataset",
                    filename="data.zip",
                    repo_type="dataset"
                )
                
                self._extract_zip(dataset_path)
                logger.info("Hugging Face dataset downloaded successfully!")
                return True
                
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            return False
    
    def download_roboflow_dataset(self):
        """Download dataset from Roboflow"""
        logger.info("Downloading dataset from Roboflow...")
        
        # Roboflow dataset URL (you'll need to get the actual download URL)
        roboflow_url = "https://universe.roboflow.com/ltomic01/driver-drowsiness-detection-gk0ws"
        
        try:
            # This would require API key and proper authentication
            logger.info("Roboflow dataset requires API key. Please download manually from:")
            logger.info(roboflow_url)
            return False
            
        except Exception as e:
            logger.error(f"Failed to download from Roboflow: {e}")
            return False
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing"""
        logger.info("Creating sample dataset...")
        
        # Create class directories
        classes = ['drowsy', 'awake']
        for class_name in classes:
            (self.data_dir / class_name).mkdir(exist_ok=True)
        
        # Create sample images
        for class_name in classes:
            for i in range(50):  # 50 images per class
                img = self._create_sample_image(class_name)
                img_path = self.data_dir / class_name / f"{class_name}_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)
        
        logger.info("Sample dataset created successfully!")
        return True
    
    def _create_sample_image(self, class_name):
        """Create a sample image for the given class"""
        # Create a 224x224 image
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add face-like features
        cv2.rectangle(img, (50, 50), (174, 174), (220, 180, 140), -1)  # Face
        
        if class_name == 'drowsy':
            # Half-closed eyes
            cv2.ellipse(img, (80, 100), (15, 8), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(img, (144, 100), (15, 8), 0, 0, 360, (50, 50, 50), -1)
            # Slightly open mouth
            cv2.ellipse(img, (112, 140), (20, 8), 0, 0, 360, (100, 100, 100), -1)
        else:  # awake
            # Wide open eyes
            cv2.ellipse(img, (80, 100), (15, 20), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(img, (144, 100), (15, 20), 0, 0, 360, (255, 255, 255), -1)
            # Closed mouth
            cv2.line(img, (92, 140), (132, 140), (100, 100, 100), 2)
        
        return img
    
    def _extract_zip(self, zip_path):
        """Extract zip file to dataset directory"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
    
    def organize_dataset(self):
        """Organize the dataset into proper structure"""
        logger.info("Organizing dataset...")
        
        # Look for common patterns in the downloaded data
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = Path(root) / file
                    
                    # Determine class based on filename or directory
                    if 'drowsy' in file.lower() or 'sleep' in file.lower():
                        target_dir = self.data_dir / 'drowsy'
                    elif 'awake' in file.lower() or 'alert' in file.lower() or 'normal' in file.lower():
                        target_dir = self.data_dir / 'awake'
                    else:
                        continue
                    
                    # Create target directory if it doesn't exist
                    target_dir.mkdir(exist_ok=True)
                    
                    # Move file to appropriate directory
                    target_path = target_dir / file
                    if not target_path.exists():
                        file_path.rename(target_path)
        
        logger.info("Dataset organized successfully!")
    
    def validate_dataset(self):
        """Validate the dataset structure"""
        logger.info("Validating dataset...")
        
        required_classes = ['drowsy', 'awake']
        total_images = 0
        
        for class_name in required_classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.error(f"Missing class directory: {class_name}")
                return False
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            count = len(images)
            total_images += count
            logger.info(f"Class '{class_name}': {count} images")
        
        logger.info(f"Total images: {total_images}")
        
        if total_images < 100:
            logger.warning("Dataset has very few images. Consider using a larger dataset.")
        
        return True
    
    def download_and_prepare(self):
        """Download and prepare the dataset"""
        logger.info("Starting dataset download and preparation...")
        
        # Try different sources
        success = False
        
        # Try Hugging Face first
        if self.download_huggingface_dataset():
            success = True
        # Try Roboflow
        elif self.download_roboflow_dataset():
            success = True
        # Fall back to sample dataset
        else:
            logger.info("Falling back to sample dataset...")
            if self.create_sample_dataset():
                success = True
        
        if success:
            # Organize the dataset
            self.organize_dataset()
            
            # Validate the dataset
            if self.validate_dataset():
                logger.info("Dataset preparation completed successfully!")
                return True
        
        logger.error("Failed to prepare dataset")
        return False

def main():
    """Main function"""
    downloader = DatasetDownloader()
    
    if downloader.download_and_prepare():
        print("✅ Dataset ready for training!")
        print("Run: python advanced_training_pipeline.py")
    else:
        print("❌ Dataset preparation failed")
        print("Using sample dataset for testing...")

if __name__ == "__main__":
    main()
