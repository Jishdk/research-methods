from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import shutil
from config import *
from utils import (
    setup_logging, create_cross_validation_folds, log_dataset_stats,
    save_dataset_metadata, setup_directories, process_split_files,
    create_yaml_file
)

logger = setup_logging()

class DatasetPreprocessor:
    def __init__(self):
        """Initialize dataset preprocessor with necessary directories"""
        self.processed_dir = OUTPUT_DIR
        self.taco_dir = self.processed_dir / "taco"
        self.trashnet_dir = self.processed_dir / "trashnet"
        self.trashnet_annotated_dir = self.processed_dir / "trashnet_annotated"
        self.example_dir = EXAMPLES_DIR
        
        # Set up directories for all datasets
        self.taco_train_dir, self.taco_val_dir, self.taco_test_dir = setup_directories(self.taco_dir)
        self.trashnet_train_dir, self.trashnet_val_dir, self.trashnet_test_dir = setup_directories(self.trashnet_dir)
        self.trashnet_annotated_train_dir, self.trashnet_annotated_val_dir, self.trashnet_annotated_test_dir = setup_directories(self.trashnet_annotated_dir)

    def is_fully_processed(self, dataset_type: str) -> bool:
        """Check if dataset and its folds are already processed"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        cv_dir = dataset_dir / "cv_splits"
        
        if not self.process_dataset(dataset_type):
            return False
            
        if not cv_dir.exists():
            return False
            
        fold_yamls = list(cv_dir.glob('fold_*/dataset.yaml'))
        if len(fold_yamls) != CV_FOLDS:
            return False
            
        return True

    def process_dataset(self, dataset_type: str) -> bool:
        """Check if dataset is already processed"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        
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

    def save_dataset_examples(self, dataset_type: str) -> None:
        """Save example images from each dataset class"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        examples_dir = self.example_dir / dataset_type
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        classes = (TRASHNET_CLASSES if dataset_type == 'trashnet' else 
                  TRASHNET_CLASSES if dataset_type == 'trashnet_annotated' else 
                  TACO_CLASSES)
        
        for class_name in classes:
            class_images = list((dataset_dir / "train" / "images").glob('*.jpg'))
            if class_images:
                for i, img_path in enumerate(class_images[:NUM_EXAMPLES]):
                    shutil.copy2(img_path, examples_dir / f"{class_name}_{i+1}.jpg")

    def process_trashnet(self):
        """Process TrashNet dataset"""
        if self.is_fully_processed('trashnet'):
            logger.info("TrashNet dataset already fully processed")
            return True

        image_paths, labels = [], []
        
        for category in TRASHNET_CLASSES:
            category_dir = TRASHNET_DIR / category
            if not category_dir.exists():
                continue

            for img_path in category_dir.glob('*.jpg'):
                class_id = TRASHNET_CLASSES.index(category)
                label = f"{class_id} 0.5 0.5 1.0 1.0\n"
                image_paths.append(img_path)
                labels.append(label)

        if not image_paths:
            logger.error("No images found in TrashNet dataset")
            return False

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
            logger.info(f"Processing {split_name} split: {len(paths)} images")
            process_split_files(paths, labels, output_dir)

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
        return True

    def process_taco(self):
        """Process TACO dataset with 70/15/15 split"""
        if self.is_fully_processed('taco'):
            logger.info("TACO dataset already fully processed")
            return True

        # Collect all images and labels first
        image_paths, labels = [], []
        total_images = 0

        # Original TACO directories
        original_splits = ['train', 'valid', 'test']
        
        # Collect all data from original splits
        for split in original_splits:
            split_dir = TACO_DIR / split
            if not split_dir.exists():
                logger.error(f"Missing {split} directory: {split_dir}")
                continue

            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'

            if not images_dir.exists() or not labels_dir.exists():
                logger.error(f"Missing images or labels directory in {split}")
                continue

            # Process all files in this split
            current_images = list(images_dir.glob('*.jpg'))
            logger.info(f"Found {len(current_images)} images in {split} directory")
            
            for img_path in current_images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    logger.warning(f"Missing label file for {img_path.name}")
                    continue

                try:
                    with open(label_path, "r") as f:
                        label_content = f.read()
                        if label_content.strip():  # Check if label file is not empty
                            labels.append(label_content)
                            image_paths.append(img_path)
                            total_images += 1
                except Exception as e:
                    logger.error(f"Error reading label file {label_path}: {e}")
                    continue

            logger.info(f"Processed {split} directory: {len(current_images)} images found")

        logger.info(f"Total valid images collected: {total_images}")

        if total_images == 0:
            logger.error("No valid images found in the dataset")
            return False

        # Split into new proportions (70/15/15)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, train_size=0.7, random_state=RANDOM_STATE, shuffle=True
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_STATE, shuffle=True
        )

        # Process new splits
        splits_data = {
            'train': (train_paths, train_labels, self.taco_train_dir),
            'val': (val_paths, val_labels, self.taco_val_dir),
            'test': (test_paths, test_labels, self.taco_test_dir)
        }

        for split_name, (paths, labels, output_dir) in splits_data.items():
            logger.info(f"Processing new {split_name} split: {len(paths)} images ({len(paths)/total_images*100:.1f}%)")
            process_split_files(paths, labels, output_dir)

        # Save configuration
        yaml_data = {
            'path': str(self.taco_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(TACO_CLASSES),
            'names': TACO_CLASSES
        }
        create_yaml_file(self.taco_dir / 'dataset.yaml', yaml_data)
        logger.info(f"TACO dataset processed: {total_images} total images")
        return True

    def process_trashnet_annotated(self):
        """Process annotated TrashNet dataset"""
        if self.is_fully_processed('trashnet_annotated'):
            logger.info("Annotated TrashNet dataset already fully processed")
            return True

        image_paths, labels = [], []
        
        for category in TRASHNET_CLASSES:
            category_dir = TRASHNET_ANNOTATED_DIR / category
            if not category_dir.exists():
                continue

            # Process annotated data
            for img_path in category_dir.glob('*.jpg'):
                label_path = category_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            label_content = f.read()
                            if label_content.strip():  # Check if label file is not empty
                                labels.append(label_content)
                                image_paths.append(img_path)
                    except Exception as e:
                        logger.error(f"Error reading label file {label_path}: {e}")
                        continue

        logger.info(f"Found {len(image_paths)} annotated images")

        if not image_paths:
            logger.error("No valid annotated images found")
            return False

        # Split data using same ratios as other datasets
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, train_size=TRAIN_RATIO, random_state=RANDOM_STATE, shuffle=True
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_STATE, shuffle=True
        )

        # Process splits
        for split_name, (paths, labels, output_dir) in {
            'train': (train_paths, train_labels, self.trashnet_annotated_train_dir),
            'val': (val_paths, val_labels, self.trashnet_annotated_val_dir),
            'test': (test_paths, test_labels, self.trashnet_annotated_test_dir)
        }.items():
            logger.info(f"Processing {split_name} split: {len(paths)} images")
            process_split_files(paths, labels, output_dir)

        # Save configuration
        yaml_data = {
            'path': str(self.trashnet_annotated_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(TRASHNET_CLASSES),
            'names': TRASHNET_CLASSES
        }
        create_yaml_file(self.trashnet_annotated_dir / 'dataset.yaml', yaml_data)
        logger.info("TrashNet annotated dataset processed successfully")
        return True

    def create_cross_validation_folds(self, dataset_type: str):
        """Create k-fold cross-validation splits"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        create_cross_validation_folds(dataset_dir, dataset_type, CV_FOLDS)

    def log_dataset_stats(self, dataset_type: str):
        """Log dataset statistics"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        log_dataset_stats(dataset_dir, dataset_type)

    def save_dataset_metadata(self, dataset_type: str):
        """Save dataset statistics"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        save_dataset_metadata(dataset_dir, dataset_type)