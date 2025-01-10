# main.py
import logging
from pathlib import Path
import sys
from config import OUTPUT_DIR, IN_COLAB
from preprocessing import DatasetPreprocessor
from utils import setup_logging, visualize_sample_images

## Logger set up
logger = setup_logging()

## Colab set up
def setup_colab_env():
    """Setup Google Colab environment if needed"""
    try:
        from google.colab import drive
        from IPython import get_ipython
        if get_ipython() and hasattr(get_ipython(), 'kernel'):
            print("Running in Google Colab. Mounting Drive...")
            drive.mount('/content/drive')
        else:
            print("Not running in Google Colab. Skipping drive.mount().")
    except ModuleNotFoundError:
        print("Google Colab module not found. Skipping drive.mount().")

def main():
    """Main preprocessing pipeline"""
    try:
        # Setup environment
        setup_colab_env()
        
        # Initialize preprocessor
        logger.info("Initializing data preprocessor...")
        preprocessor = DatasetPreprocessor()
        
        # Visualize sample images from both datasets
        visualize_sample_images(preprocessor)
        
        # Process TrashNet dataset
        logger.info("Processing TrashNet dataset...")
        preprocessor.process_trashnet()
        preprocessor.create_cross_validation_folds('trashnet')
        preprocessor.log_dataset_stats('trashnet')
        preprocessor.save_dataset_metadata('trashnet')
        
        # Process TACO dataset
        logger.info("Processing TACO dataset...")
        preprocessor.process_taco()
        preprocessor.create_cross_validation_folds('taco')
        preprocessor.log_dataset_stats('taco')
        preprocessor.save_dataset_metadata('taco')
        
        logger.info(f"Preprocessing complete. Output saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
