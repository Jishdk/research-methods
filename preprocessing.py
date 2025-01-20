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
        
        # Set up directories for all datasets
        self.dataset_dirs = {}
        self.split_dirs = {}
        
        for dataset in DATASET_DIRS.keys():
            self.dataset_dirs[dataset] = self.processed_dir / dataset
            self.split_dirs[dataset] = setup_directories(self.dataset_dirs[dataset])

    def is_fully_processed(self, dataset_type: str) -> bool:
        """Check if dataset and its folds are already processed"""
        dataset_dir = self.dataset_dirs[dataset_type]
        cv_dir = dataset_dir / "cv_splits"
        
        # Check basic preprocessing
        if not self.process_dataset(dataset_type):
            return False
            
        # Check cross-validation folds
        if not cv_dir.exists():
            return False
            
        fold_yamls = list(cv_dir.glob('fold_*/dataset.yaml'))
        if len(fold_yamls) != CV_FOLDS:
            return False
            
        return True

    def process_dataset(self, dataset_type: str) -> bool:
        """Check if dataset is already processed"""
        dataset_dir = self.dataset_dirs[dataset_type]
        
        if dataset_dir.exists():
            required_dirs = [
                dataset_dir / "train" / "images",
                dataset_dir / "train" / "labels",
                dataset_dir / "val" / "images",
                dataset_dir / "val" / "labels",
                dataset_dir / "test" / "images",
                dataset_dir / "test" / "labels"
            ]
            yaml_file = dataset_dir / "dataset.yaml"
            
            if all(d.exists() for d in required_dirs) and yaml_file.exists():
                if len(list((dataset_dir / "train" / "images").glob('*.jpg'))) > 0:
                    return True
                    
        return False

    def process_image_batch(self, image_paths: List[Path], labels: List[str], 
                          split_dir: Path) -> None:
        """Process a batch of images
        
        Args:
            image_paths: List of paths to images
            labels: List of label strings
            split_dir: Directory to save processed files
        """
        processed_images = []
        processed_labels = []
        
        for img_path, label in zip(image_paths, labels):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed_images.append((img_path.stem, img))
                processed_labels.append(label)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        process_split_files(processed_images, processed_labels, split_dir, IMG_SIZE)

    def process_trashnet(self):
        """Process TrashNet dataset"""
        if self.is_fully_processed('trashnet'):
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
        splits = {
            'train': (train_paths, train_labels, self.split_dirs['trashnet'][0]),
            'val': (val_paths, val_labels, self.split_dirs['trashnet'][1]),
            'test': (test_paths, test_labels, self.split_dirs['trashnet'][2])
        }
        
        for split_name, (paths, labels, output_dir) in splits.items():
            logger.info(f"Processing {split_name} split...")
            self.process_image_batch(paths, labels, output_dir)

        # Save configuration
        self._save_dataset_config('trashnet', TRASHNET_CLASSES)

    def process_taco(self):
        """Process TACO dataset"""
        if self.is_fully_processed('taco'):
            logger.info("TACO dataset already fully processed")
            return

        # Process each split from original TACO dataset
        split_mapping = {'train': 'train', 'valid': 'val', 'test': 'test'}
        for input_split, output_split in split_mapping.items():
            split_dir = TACO_DIR / input_split
            if not split_dir.exists():
                continue

            output_dir = self.split_dirs['taco'][list(split_mapping.values()).index(output_split)]
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'

            # Collect and process data
            image_paths, labels = [], []
            for img_path in images_dir.glob('*.jpg'):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                try:
                    with open(label_path, "r") as f:
                        labels.append(f.read())
                        image_paths.append(img_path)
                except Exception as e:
                    logger.error(f"Error reading label file {label_path}: {e}")
                    continue

            self.process_image_batch(image_paths, labels, output_dir)

        # Save configuration
        self._save_dataset_config('taco', TACO_CLASSES)

    def process_trashnet_annotated(self):
        """Process annotated TrashNet dataset"""
        if self.is_fully_processed('trashnet_annotated'):
            logger.info("Annotated TrashNet dataset already fully processed")
            return

        # Process existing split structure
        for split in ['train', 'val', 'test']:
            split_dir = TRASHNET_ANNOTATED_DIR / split
            if not split_dir.exists():
                continue

            output_dir = self.split_dirs['trashnet_annotated'][
                ['train', 'val', 'test'].index(split)
            ]
            
            # Copy files to processed directory
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            image_paths, labels = [], []
            for img_path in images_dir.glob('*.jpg'):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                try:
                    with open(label_path, "r") as f:
                        labels.append(f.read())
                        image_paths.append(img_path)
                except Exception as e:
                    logger.error(f"Error reading label file {label_path}: {e}")
                    continue

            self.process_image_batch(image_paths, labels, output_dir)

        # Save configuration
        self._save_dataset_config('trashnet_annotated', TRASHNET_CLASSES)

    def _save_dataset_config(self, dataset_type: str, class_names: List[str]):
        """Save dataset configuration YAML file"""
        yaml_data = {
            'path': str(self.dataset_dirs[dataset_type].absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        create_yaml_file(self.dataset_dirs[dataset_type] / 'dataset.yaml', yaml_data)

    def create_cross_validation_folds(self, dataset_type: str):
        """Create k-fold cross-validation splits"""
        dataset_dir = self.dataset_dirs[dataset_type]
        class_names = (TRASHNET_CLASSES if dataset_type in ['trashnet', 'trashnet_annotated'] 
                      else TACO_CLASSES)
        create_cross_validation_folds(dataset_dir, class_names, CV_FOLDS)

    def log_dataset_stats(self, dataset_type: str):
        """Log dataset statistics"""
        dataset_dir = self.dataset_dirs[dataset_type]
        log_dataset_stats(dataset_dir, dataset_type)

    def save_dataset_metadata(self, dataset_type: str):
        """Save dataset statistics"""
        dataset_dir = self.dataset_dirs[dataset_type]
        save_dataset_metadata(dataset_dir, dataset_type)