# preprocessing.py
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Tuple
from utils import (
    setup_logging, create_cross_validation_folds, log_dataset_stats,
    save_dataset_metadata, setup_directories, process_split_files,
    create_yaml_file, augment_image_and_label
)
import cv2
import numpy as np
from config import *

logger = setup_logging()

class DatasetPreprocessor:
    def __init__(self, save_examples: int = 5):
        """Initialize the dataset preprocessor"""
        self.processed_dir = OUTPUT_DIR
        self.taco_dir = self.processed_dir / "taco"
        self.trashnet_dir = self.processed_dir / "trashnet"
        self.example_dir = self.processed_dir / "examples"
        self.example_dir.mkdir(parents=True, exist_ok=True)
        self.save_examples = save_examples

        # Set up directories for both datasets
        self.taco_train_dir, self.taco_val_dir, self.taco_test_dir = setup_directories(self.taco_dir)
        self.trashnet_train_dir, self.trashnet_val_dir, self.trashnet_test_dir = setup_directories(self.trashnet_dir)

    def is_fully_processed(self, dataset_type: str) -> bool:
        """Check if dataset and its folds are already processed"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
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

    def augment_data(self, image: np.ndarray, label: str) -> List[Tuple[np.ndarray, str]]:
        """Generate augmented versions of an image and its label"""
        augmented_pairs = []
        for _ in range(AUGMENTATION_FACTOR):
            aug_img, aug_label = augment_image_and_label(image, label)
            augmented_pairs.append((aug_img, aug_label))
        return augmented_pairs if AUGMENTATION_ENABLED else []

    def process_image_batch(self, image_paths: List[Path], labels: List[str], 
                          split_dir: Path, is_training: bool = False) -> None:
        """Process a batch of images with optional augmentation for training"""
        processed_images = []
        processed_labels = []
        
        for img_path, label in zip(image_paths, labels):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append((img_path.stem, img))
            processed_labels.append(label)
            
            if is_training and AUGMENTATION_ENABLED:
                try:
                    augmented_pairs = self.augment_data(img, label)
                    for idx, (aug_img, aug_label) in enumerate(augmented_pairs):
                        processed_images.append((f"{img_path.stem}_aug_{idx}", aug_img))
                        processed_labels.append(aug_label)
                except Exception as e:
                    logger.error(f"Error augmenting {img_path}: {e}")
                    continue
        
        process_split_files(processed_images, processed_labels, split_dir, 
                          IMG_SIZE, COLOR_MEAN, COLOR_STD)

    def process_trashnet(self):
        """Process TrashNet dataset with augmentation"""
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
        for split_name, (paths, labels, output_dir) in {
            'train': (train_paths, train_labels, self.trashnet_train_dir),
            'val': (val_paths, val_labels, self.trashnet_val_dir),
            'test': (test_paths, test_labels, self.trashnet_test_dir)
        }.items():
            logger.info(f"Processing {split_name} split...")
            self.process_image_batch(paths, labels, output_dir, is_training=(split_name=='train'))

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

    def process_taco(self):
        """Process TACO dataset with augmentation"""
        if self.is_fully_processed('taco'):
            logger.info("TACO dataset already fully processed")
            return

        split_mapping = {'train': 'train', 'valid': 'val', 'test': 'test'}

        for input_split, output_split in split_mapping.items():
            split_dir = TACO_DIR / input_split
            if not split_dir.exists():
                continue

            output_dir = getattr(self, f'taco_{output_split}_dir')
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'

            # Collect data
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

            # Process split
            is_training = (input_split == 'train')
            self.process_image_batch(image_paths, labels, output_dir, is_training=is_training)

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