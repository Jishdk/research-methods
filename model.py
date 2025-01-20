from pathlib import Path
import logging
import torch
from ultralytics import YOLO
from config import *
from utils import setup_logging

logger = setup_logging()

class BaselineModel:
    """Baseline YOLO model without training"""
    
    def __init__(self, model_size: str = DEFAULT_MODEL):
        """Initialize baseline model with pretrained weights
        
        Args:
            model_size: YOLOv8 model size (n, s, m, l, x)
        """
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading pretrained YOLOv8 model: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def predict(self, data_yaml: Path, split: str = 'test') -> object:
        """Run predictions using the baseline model
        
        Args:
            data_yaml: Path to dataset YAML file
            split: Dataset split to evaluate ('train', 'val', 'test')
            
        Returns:
            YOLO Results object containing evaluation metrics
        """
        try:
            dataset_name = data_yaml.parent.name
            logger.info(f"Running predictions on {dataset_name} {split} set")
            
            # Use prediction config from config.py
            results = self.model.val(
                data=str(data_yaml),
                split=split,
                imgsz=IMG_SIZE,
                batch=PREDICTION_CONFIG['batch_size'],
                save_txt=PREDICTION_CONFIG['save_txt'],
                save_conf=PREDICTION_CONFIG['save_conf'],
                conf=PREDICTION_CONFIG['conf'],
                iou=PREDICTION_CONFIG['iou'],
                project=str(BASELINE_RESULTS_DIR),
                name=dataset_name,
                exist_ok=True,
                plots=PREDICTION_CONFIG['plots'],
                save_json=PREDICTION_CONFIG['save_json']
            )
            
            logger.info(f"Predictions completed for {dataset_name} {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

class TrainedModel:
    """YOLO model for training on garbage detection"""
    
    def __init__(self, model_size: str = DEFAULT_MODEL, dataset: str = 'trashnet'):
        """Initialize model for training
        
        Args:
            model_size: YOLOv8 model size (n, s, m, l, x)
            dataset: Dataset to train on ('trashnet', 'taco', 'trashnet_annotated')
        """
        if dataset not in DATASET_DIRS:
            raise ValueError(f"Dataset must be one of {list(DATASET_DIRS.keys())}, got {dataset}")
            
        self.dataset = dataset
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading YOLOv8 model for {dataset} training: {model_name}")
        
        # Select appropriate results directory
        self.results_dir = DATASET_DIRS[dataset]
        
        try:
            self.model = YOLO(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train(self, data_yaml: Path) -> object:
        """Train the model on specified dataset
        
        Args:
            data_yaml: Path to dataset YAML file
            
        Returns:
            YOLO Results object containing training metrics
        """
        try:
            logger.info(f"Starting training on {self.dataset}")
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, training on CPU")
            
            # Train model using config
            results = self.model.train(
                data=str(data_yaml),
                epochs=TRAINING_CONFIG['epochs'],
                imgsz=IMG_SIZE,
                batch=TRAINING_CONFIG['batch_size'],
                optimizer=TRAINING_CONFIG['optimizer'],
                lr0=TRAINING_CONFIG['learning_rate'],
                weight_decay=TRAINING_CONFIG['weight_decay'],
                device=TRAINING_CONFIG['device'],
                project=str(self.results_dir),
                name=self.dataset,
                save_period=TRAINING_CONFIG['save_period'],
                exist_ok=True,
                patience=TRAINING_CONFIG['patience'],
                workers=TRAINING_CONFIG['workers'],
                resume=TRAINING_CONFIG['resume'],
                plots=True,  # Enable training plots
                save=True,  # Save best model
                val=True    # Run validation during training
            )
            
            logger.info(f"Training completed for {self.dataset}")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, data_yaml: Path, split: str = 'test') -> object:
        """Run predictions using the trained model
        
        Args:
            data_yaml: Path to dataset YAML file
            split: Dataset split to evaluate ('train', 'val', 'test')
            
        Returns:
            YOLO Results object containing evaluation metrics
        """
        try:
            logger.info(f"Running predictions on {self.dataset} {split} set")
            
            # Use prediction config from config.py
            results = self.model.val(
                data=str(data_yaml),
                split=split,
                imgsz=IMG_SIZE,
                batch=PREDICTION_CONFIG['batch_size'],
                save_txt=PREDICTION_CONFIG['save_txt'],
                save_conf=PREDICTION_CONFIG['save_conf'],
                conf=PREDICTION_CONFIG['conf'],
                iou=PREDICTION_CONFIG['iou'],
                project=str(self.results_dir),
                name=f"{self.dataset}_predictions",
                exist_ok=True,
                plots=PREDICTION_CONFIG['plots'],
                save_json=PREDICTION_CONFIG['save_json']
            )
            
            logger.info(f"Predictions completed for {self.dataset} {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise