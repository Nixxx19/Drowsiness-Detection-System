"""
Script to download the required facial landmark predictor model
"""

import os
import urllib.request
import bz2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """Download and extract the facial landmark predictor model"""
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
    extracted_file = "shape_predictor_68_face_landmarks.dat"
    
    # Check if model already exists
    if os.path.exists(extracted_file):
        logger.info("Facial landmark predictor model already exists")
        return True
    
    try:
        logger.info("Downloading facial landmark predictor model...")
        urllib.request.urlretrieve(model_url, compressed_file)
        logger.info("Download completed")
        
        logger.info("Extracting compressed file...")
        with bz2.BZ2File(compressed_file, 'rb') as source:
            with open(extracted_file, 'wb') as target:
                target.write(source.read())
        
        # Clean up compressed file
        os.remove(compressed_file)
        logger.info("Model extraction completed")
        
        # Verify file exists and has reasonable size
        if os.path.exists(extracted_file) and os.path.getsize(extracted_file) > 1000000:  # > 1MB
            logger.info("Facial landmark predictor model ready to use")
            return True
        else:
            logger.error("Downloaded file appears to be corrupted")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    if success:
        print("Model download completed successfully!")
    else:
        print("Model download failed. Please try again.")
