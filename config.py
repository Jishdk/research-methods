import os
from pathlib import Path
import torch

# Base Directory Structure
CODE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = CODE_DIR / "data"
OUTPUT_DIR = CODE_DIR / "preprocessed_data"
RESULTS_DIR = CODE_DIR / "results"

# Dataset Source Directories
TRASHNET_DIR = DATA_DIR / "data_trashnet"
TACO_DIR = DATA_DIR / "data_taco"
TRASHNET_ANNOTATED_DIR = DATA_DIR / "data_trashnet_annotated"

# Results Directory Structure
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
TRAINED_RESULTS_DIR = RESULTS_DIR / "trained"

# Dataset-specific Results Directories
TRAINED_TRASHNET_DIR = TRAINED_RESULTS_DIR / "trashnet"
TRAINED_TACO_DIR = TRAINED_RESULTS_DIR / "taco"
TRAINED_TRASHNET_ANNOTATED_DIR = TRAINED_RESULTS_DIR / "trashnet_annotated"

# Dataset Classes
TRASHNET_CLASSES = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash'
]

TACO_CLASSES = [
    'Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can',
    'Carton', 'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic',
    'Paper', 'Plastic bag - wrapper', 'Plastic container', 'Pop tab',
    'Straw', 'Styrofoam piece', 'Unlabeled litter'
]

# Category Mapping (TACO to TrashNet)
CATEGORY_MAPPING = {
    # Plastic items
    'Bottle': 'plastic',
    'Plastic container': 'plastic',
    'Plastic bag - wrapper': 'plastic',
    'Other plastic': 'plastic',
    # Metal items
    'Can': 'metal',
    'Bottle cap': 'metal',
    'Pop tab': 'metal',
    'Aluminium foil': 'metal',
    # Glass items
    'Broken glass': 'glass',
    # Paper items
    'Paper': 'paper',
    # Cardboard items
    'Carton': 'cardboard',
    # General trash
    'Cup': 'trash',
    'Lid': 'trash',
    'Other litter': 'trash',
    'Cigarette': 'trash',
    'Straw': 'trash',
    'Styrofoam piece': 'trash',
    'Unlabeled litter': 'trash'
}

# Dataset Processing Settings
IMG_SIZE = 640  # YOLOv8 optimal size

# Dataset Split Ratios 
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Cross Validation Settings
CV_FOLDS = 3
RANDOM_STATE = 42

# Model Configurations
YOLO_MODELS = {
    'n': 'nano',      # 3.2M parameters
    's': 'small',     # 11.2M parameters
    'm': 'medium',    # 25.9M parameters
    'l': 'large',     # 43.8M parameters
    'x': 'extra'      # 68.2M parameters
}
DEFAULT_MODEL = 'n'  # Using nano for faster experimentation

# Training Settings
TRAINING_CONFIG = {
    'batch_size': 16,
    'epochs': 100,
    'optimizer': 'AdamW',
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'save_period': 10,  # Save checkpoints every N epochs
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'workers': 8,  # Number of worker threads
    'patience': 20,  # Early stopping patience
    'resume': True,  # Resume training from last checkpoint if available
}

# Evaluation Settings
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# Metrics to Compute
METRICS = [
    'mAP50',        # mean Average Precision at IoU=0.50
    'mAP50-95',     # mean Average Precision at IoU=0.50:0.95
    'precision',    # Precision score
    'recall',       # Recall score
    'f1'           # F1 score
]

# Prediction Settings
PREDICTION_CONFIG = {
    'batch_size': 16,
    'save_txt': True,       # Save predictions as text files
    'save_conf': True,      # Save confidence scores
    'iou': IOU_THRESHOLD,   # IOU threshold for NMS
    'conf': CONF_THRESHOLD, # Confidence threshold
    'save_json': True,      # Save results to JSON
    'plots': True,         # Generate plots
}

# Visualization Settings
FIGURE_SIZES = {
    'confusion_matrix': (12, 10),
    'class_performance': (12, 6),
    'learning_curves': (10, 6),
    'examples_grid': (15, 10),  # Size for example images grid
    'precision_recall': (10, 6)  # Size for precision-recall curves
}

# Number of example images to show per class
EXAMPLES_PER_CLASS = 6

# Plot Settings
PLOT_CONFIG = {
    'dpi': 300,
    'font_size': 10,
    'line_width': 2,
    'grid': True,
    'color_palette': 'deep'  # Seaborn color palette
}

# Create Required Directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BASELINE_RESULTS_DIR, exist_ok=True)
os.makedirs(TRAINED_RESULTS_DIR, exist_ok=True)

# Create Dataset-specific Results Directories
DATASET_DIRS = {
    'trashnet': TRAINED_TRASHNET_DIR,
    'taco': TRAINED_TACO_DIR,
    'trashnet_annotated': TRAINED_TRASHNET_ANNOTATED_DIR
}

for dir_name, dir_path in DATASET_DIRS.items():
    os.makedirs(BASELINE_RESULTS_DIR / dir_name, exist_ok=True)
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(dir_path / "plots", exist_ok=True)
    os.makedirs(dir_path / "examples", exist_ok=True)