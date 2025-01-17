from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Tuple
from utils import (
    setup_logging, create_cross_validation_folds, log_dataset_stats,
    save_dataset_metadata, setup_directories, process_split_files,
    create_yaml_file
)
import cv2
import numpy as np
from config import *

logger = setup_logging()

class DatasetPreprocessor:
    def __init__(self):
        """Initialize the dataset preprocessor"""
        self.processed_dir = OUTPUT_DIR
        self.trashnet_dir = self.processed_dir / "trashnet"

        # Set up directories
        self.trashnet_train_dir, self.trashnet_val_dir, self.trashnet_test_dir = setup_directories(self.trashnet_dir)

    def is_fully_processed(self) -> bool:
        """Check if dataset and its folds are already processed"""
        cv_dir = self.trashnet_dir / "cv_splits"
        
        # Check basic preprocessing
        if not self.process_dataset():
            return False
            
        # Check cross-validation folds
        if not cv_dir.exists():
            return False
            
        fold_yamls = list(cv_dir.glob('fold_*/dataset.yaml'))
        if len(fold_yamls) != CV_FOLDS:
            return False
            
        return True

    def process_dataset(self) -> bool:
        """Check if dataset is already processed"""
        if self.trashnet_dir.exists():
            required_dirs = [
                self.trashnet_dir / "train" / "images",
                self.trashnet_dir / "train" / "labels",
                self.trashnet_dir / "val" / "images",
                self.trashnet_dir / "val" / "labels",
                self.trashnet_dir / "test" / "images",
                self.trashnet_dir / "test" / "labels"
            ]
            yaml_file = self.trashnet_dir / "dataset.yaml"
            
            if all(d.exists() for d in required_dirs) and yaml_file.exists():
                if len(list((self.trashnet_dir / "train" / "images").glob('*.jpg'))) > 0:
                    return True
                    
        return False

    def process_image_batch(self, image_paths: List[Path], labels: List[str], 
                          split_dir: Path) -> None:
        """Process a batch of images"""
        processed_images = []
        processed_labels = []
        
        for img_path, label in zip(image_paths, labels):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append((img_path.stem, img))
            processed_labels.append(label)
        
        process_split_files(processed_images, processed_labels, split_dir, 
                          IMG_SIZE)

    def process_trashnet(self):
        """Process TrashNet dataset"""
        if self.is_fully_processed():
            logger.info("TrashNet dataset already fully processed")
            return

        image_paths, labels = [], []
        
        # Collect data
        for category in TRASHNET_CLASSES:
            category_dir = TRASHNET_DIR / category
            if not category_dir.exists():
                continue

            for img_path in category_dir.glob('*.jpg'):
                class_id = TRASHNET_CLASSES.index(category)
                label = f"{class_id} 0.5 0.5 1.0 1.0\n"
                image_paths.append(img_path)
                labels.append(label)

        # Split data
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, train_size=TRAIN_RATIO, random_state=RANDOM_STATE, shuffle=True
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_STATE, shuffle=True
        )

        # Process splits
        for split_name, (paths, labels, output_dir) in {
            'train': (train_paths, train_labels, self.trashnet_train_dir),
            'val': (val_paths, val_labels, self.trashnet_val_dir),
            'test': (test_paths, test_labels, self.trashnet_test_dir)
        }.items():
            logger.info(f"Processing {split_name} split...")
            self.process_image_batch(paths, labels, output_dir)

        # Save configuration
        yaml_data = {
            'path': str(self.trashnet_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(TRASHNET_CLASSES),
            'names': TRASHNET_CLASSES
        }
        create_yaml_file(self.trashnet_dir / 'dataset.yaml', yaml_data)

    def create_cross_validation_folds(self):
        """Create k-fold cross-validation splits"""
        create_cross_validation_folds(self.trashnet_dir, CV_FOLDS)

    def log_dataset_stats(self):
        """Log dataset statistics"""
        log_dataset_stats(self.trashnet_dir)

    def save_dataset_metadata(self):
        """Save dataset statistics"""
        save_dataset_metadata(self.trashnet_dir)