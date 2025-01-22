# utils.py

import cv2
import numpy as np
import pandas as pd
import yaml
import json
import logging
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from config import *

def setup_logging():
    """
    Set up logging configuration
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def setup_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Create and return necessary data directories
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Tuple[Path, Path, Path]: Paths to train, val, and test directories
    """
    splits = ['train', 'val', 'test']
    split_dirs = []
    
    for split in splits:
        split_dir = base_dir / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        split_dirs.append(split_dir)
    
    return tuple(split_dirs)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image maintaining aspect ratio with padding
    
    Args:
        image: Input image array
        target_size: Desired output size (width, height)
        
    Returns:
        np.ndarray: Resized and padded image
    """
    if image is None or len(image.shape) != 3:
        raise ValueError(f"Invalid image shape: {getattr(image, 'shape', None)}")
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    # Resize and pad
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded

def normalize_image(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """
    Normalize image using provided mean and standard deviation
    
    Args:
        image: Input image array
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        np.ndarray: Normalized image
    """
    return ((image.astype(np.float32) / 255.0) - np.array(mean)) / np.array(std)

def process_split_files(files: List[Path], labels: List[str], 
                       split_dir: Path) -> None:
    """
    Process and save image files with labels
    
    Args:
        files: List of image file paths
        labels: List of label strings
        split_dir: Output directory for the split
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    for img_path, label in zip(files, labels):
        try:
            # Copy image
            shutil.copy2(img_path, images_dir / f"{img_path.stem}.jpg")
            
            # Save label
            with open(labels_dir / f"{img_path.stem}.txt", 'w') as f:
                f.write(label)
                
        except Exception as e:
            logger.error(f"Error processing {img_path.stem}: {e}")

def create_yaml_file(path: Path, data: Dict) -> None:
    """
    Create YAML file with dataset configuration
    
    Args:
        path: Output path for YAML file
        data: Configuration data to save
    """
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def count_images_per_class(labels_dir: Path) -> Dict[int, int]:
    """
    Count number of images per class in labels directory
    
    Args:
        labels_dir: Directory containing label files
        
    Returns:
        Dict[int, int]: Dictionary mapping class IDs to counts
    """
    class_counts = {}
    for label_file in labels_dir.glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {e}")
    return class_counts

def save_dataset_metadata(dataset_dir: Path, dataset_type: str) -> None:
    """
    Save comprehensive dataset statistics
    
    Args:
        dataset_dir: Dataset directory path
        dataset_type: Type of dataset
    """
    metadata = {
        'total_images': 0,
        'class_distribution': {},
        'splits': {}
    }
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        n_images = len(list((split_dir / "images").glob('*.jpg')))
        metadata['splits'][split] = {
            'images': n_images,
            'class_distribution': count_images_per_class(split_dir / "labels")
        }
        metadata['total_images'] += n_images
    
    output_file = dataset_dir / f"{dataset_type}_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def log_dataset_stats(dataset_dir: Path, dataset_type: str) -> None:
    """
    Log basic dataset statistics
    
    Args:
        dataset_dir: Dataset directory path
        dataset_type: Type of dataset
    """
    stats = {split: len(list((dataset_dir / split / "images").glob("*.jpg")))
            for split in ["train", "val", "test"]}
    
    total_images = sum(stats.values())
    logger.info(f"{dataset_type} dataset processed:")
    logger.info(f"Total images: {total_images}")
    for split, count in stats.items():
        logger.info(f"{split} split: {count} images")

def create_cross_validation_folds(dataset_dir: Path, dataset_type: str, cv_folds: int) -> None:
    """
    Create k-fold cross-validation splits
    
    Args:
        dataset_dir: Dataset directory path
        dataset_type: Type of dataset
        cv_folds: Number of folds to create
    """
    cv_dir = dataset_dir / "cv_splits"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    train_images_dir = dataset_dir / "train" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        logger.error(f"Training directories not found in {dataset_dir}")
        return
        
    image_files = sorted(list(train_images_dir.glob('*.jpg')))
    if not image_files:
        logger.error(f"No training images found in {train_images_dir}")
        return
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        logger.info(f"Processing fold {fold_idx + 1}/{cv_folds}")
        
        fold_dir = cv_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]
        
        metadata = {}
        for split_name, files in [('train', train_files), ('val', val_files)]:
            split_dir = fold_dir / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
            
            for img_file in files:
                label_file = train_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(img_file, split_dir / "images" / img_file.name)
                    shutil.copy2(label_file, split_dir / "labels" / f"{img_file.stem}.txt")
        
        # Create fold configuration
        class_names = (TRASHNET_CLASSES if dataset_type == 'trashnet' else 
                      TRASHNET_ANNOTATED_CLASSES if dataset_type == 'trashnet_annotated' else 
                      TACO_CLASSES)
        
        yaml_data = {
            'path': str(fold_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(class_names),
            'names': class_names
        }
        create_yaml_file(fold_dir / "dataset.yaml", yaml_data)
    
    logger.info(f"Created {cv_folds} cross-validation folds in {cv_dir}")

def load_training_config(dataset_type: str) -> Dict:
    """
    Load training configuration for specified dataset
    
    Args:
        dataset_type: Type of dataset
        
    Returns:
        Dict: Training configuration
    """
    if dataset_type == 'trashnet':
        return TRASHNET_TRAINING_CONFIG
    elif dataset_type == 'trashnet_annotated':
        return TRASHNET_ANNOTATED_TRAINING_CONFIG
    elif dataset_type == 'taco':
        return TACO_TRAINING_CONFIG
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def plot_learning_curves(metrics: Dict, save_dir: Path) -> None:
    """
    Plot training and validation learning curves
    
    Args:
        metrics: Dictionary containing training metrics
        save_dir: Directory to save plots
    """
    plt.figure(figsize=FIGURE_SIZES['learning_curves'])
    
    for metric in ['mAP50', 'precision', 'recall']:
        if metric in metrics:
            plt.plot(metrics['epoch'], metrics[metric], label=metric)
    
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_dir / 'learning_curves.png')
    plt.close()