# utils.py
import cv2
import numpy as np
import pandas as pd
import yaml
import json
import logging
import random
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from config import *

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

### Directory Setup ###
def setup_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Create and return necessary data directories"""
    splits = ['train', 'val', 'test']
    split_dirs = []
    
    for split in splits:
        split_dir = base_dir / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        split_dirs.append(split_dir)
    
    return tuple(split_dirs)

### Image Processing ###
def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image maintaining aspect ratio with padding"""
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
    """Normalize image using provided mean and std"""
    return ((image.astype(np.float32) / 255.0) - np.array(mean)) / np.array(std)

### Data Augmentation ###
def rotate_box(box_params: Tuple[float, float, float, float], 
              angle: float, image_dims: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """Rotate bounding box coordinates"""
    x_center, y_center, width, height = box_params
    image_width, image_height = image_dims
    
    # Convert to pixel coordinates
    x = x_center * image_width
    y = y_center * image_height
    
    # Rotate around image center
    angle_rad = np.radians(angle)
    cx, cy = image_width / 2, image_height / 2
    
    x_shifted = x - cx
    y_shifted = y - cy
    
    new_x = cx + (x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad))
    new_y = cy + (x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad))
    
    # Convert back to normalized coordinates
    return (new_x / image_width, new_y / image_height, width, height)

def augment_image_and_label(image: np.ndarray, label: str) -> Tuple[np.ndarray, str]:
    """Apply augmentation to image and its label according to proposal specifications"""
    # Parse label
    parts = label.strip().split()
    class_id = int(parts[0])
    box_params = tuple(map(float, parts[1:]))
    
    # Apply augmentations from config
    angle = random.uniform(*AUGMENTATION_CONFIG['rotation_range'])
    
    # Rotate image and box
    matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    new_box = rotate_box(box_params, angle, (image.shape[1], image.shape[0]))
    
    # Apply additional augmentations
    if AUGMENTATION_CONFIG['flip_horizontal'] and random.random() > 0.5:
        image = cv2.flip(image, 1)
        new_box = (1 - new_box[0], new_box[1], new_box[2], new_box[3])
        
    if AUGMENTATION_CONFIG['flip_vertical'] and random.random() > 0.5:
        image = cv2.flip(image, 0)
        new_box = (new_box[0], 1 - new_box[1], new_box[2], new_box[3])
    
    # Apply color jittering
    jitter = AUGMENTATION_CONFIG['color_jitter']
    alpha = random.uniform(*jitter['brightness'])
    beta = random.uniform(-30, 30)  # Contrast adjustment
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Create new label string
    new_label = f"{class_id} {' '.join(f'{x:.6f}' for x in new_box)}\n"
    return image, new_label

### File Processing ###
def process_split_files(files: List[Tuple[str, np.ndarray]], labels: List[str], 
                       split_dir: Path, img_size: int, 
                       mean: List[float], std: List[float]) -> None:
    """Process and save image files with labels"""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    for (stem, img), label in zip(files, labels):
        try:
            # Process image
            target_size = (img_size, img_size)
            img = resize_image(img, target_size)
            
            # Save processed image
            cv2.imwrite(
                str(images_dir / f"{stem}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            
            # Save label
            with open(labels_dir / f"{stem}.txt", 'w') as f:
                f.write(label)
                
        except Exception as e:
            logger.error(f"Error processing {stem}: {e}")

def create_yaml_file(path: Path, data: Dict) -> None:
    """Create YAML file with dataset configuration"""
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

### Dataset Statistics ###
def count_images_per_class(labels_dir: Path) -> Dict[int, int]:
    """Count number of images per class in labels directory"""
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
    """Save comprehensive dataset statistics"""
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
    """Log basic dataset statistics"""
    stats = {split: len(list((dataset_dir / split / "images").glob("*.jpg")))
            for split in ["train", "val", "test"]}
    
    total_images = sum(stats.values())
    logger.info(f"{dataset_type} dataset processed:")
    logger.info(f"Total images: {total_images}")
    for split, count in stats.items():
        logger.info(f"{split} split: {count} images")

### Cross Validation ###
def create_cross_validation_folds(dataset_dir: Path, dataset_type: str, cv_folds: int) -> None:
    """Create k-fold cross-validation splits"""
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
            
            split_metadata = {}
            for img_file in files:
                label_file = train_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(img_file, split_dir / "images" / img_file.name)
                    shutil.copy2(label_file, split_dir / "labels" / f"{img_file.stem}.txt")
                    
                    with open(label_file, 'r') as f:
                        class_id = int(f.readline().split()[0])
                        split_metadata[class_id] = split_metadata.get(class_id, 0) + 1
            
            metadata[f"fold_{fold_idx}_{split_name}"] = split_metadata
        
        yaml_data = {
            'path': str(fold_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(TRASHNET_CLASSES if dataset_type == 'trashnet' else TACO_CLASSES),
            'names': TRASHNET_CLASSES if dataset_type == 'trashnet' else TACO_CLASSES
        }
        create_yaml_file(fold_dir / "dataset.yaml", yaml_data)
    
    distribution_file = cv_dir / "fold_distribution.csv"
    df = pd.DataFrame.from_dict(metadata, orient='index')
    df.index.name = 'Fold'
    df.to_csv(distribution_file)
    
    logger.info(f"Created {cv_folds} cross-validation folds in {cv_dir}")

### Training Utilities ###
def load_training_config(dataset_type: str) -> Dict:
    """Load training configuration for specified dataset"""
    if dataset_type == 'trashnet':
        return TRASHNET_TRAINING_CONFIG
    elif dataset_type == 'taco':
        return TACO_TRAINING_CONFIG
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def save_training_metadata(results_dir: Path, dataset_type: str, metrics: Dict) -> None:
    """Save training results metadata"""
    metadata = {
        'dataset': dataset_type,
        'metrics': metrics,
        'date': str(datetime.now())
    }
    
    metadata_file = results_dir / dataset_type / "training_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def plot_training_curves(results_dir: Path, dataset_type: str, metrics: Dict) -> None:
    """Plot and save training curves"""
    plot_dir = results_dir / dataset_type / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Training Loss Plot
    plt.figure(figsize=FIGURE_SIZES['learning_curves'])
    for key in ['train_loss', 'val_loss']:
        if key in metrics:
            plt.plot(metrics['epoch'], metrics[key], label=key.replace('_', ' ').title())
    plt.title(f"{dataset_type} Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_dir / 'training_loss.png')
    plt.close()
    
    # Metrics Plot
    plt.figure(figsize=FIGURE_SIZES['learning_curves'])
    for metric in METRICS:
        if metric in metrics:
            plt.plot(metrics['epoch'], metrics[metric], label=metric)
    plt.title(f"{dataset_type} Training Metrics")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(plot_dir / 'training_metrics.png')
    plt.close()