# model.py

from pathlib import Path
import logging
from ultralytics import YOLO
from config import *
from utils import setup_logging

logger = setup_logging()

class BaselineModel:
    """Baseline YOLOv8 model without training, used only for TrashNet evaluation"""
    
    def __init__(self, model_size: str = DEFAULT_MODEL):
        """Initialize baseline model with pretrained weights
        
        Args:
            model_size: Size of YOLOv8 model to use (n, s, m, l, x)
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
            data_yaml: Path to dataset configuration file
            split: Dataset split to use (train, val, test)
            
        Returns:
            object: YOLOv8 validation results
        """
        try:
            dataset_name = data_yaml.parent.name
            logger.info(f"Running predictions on {dataset_name} {split} set")
            
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
                exist_ok=True
            )
            
            logger.info(f"Predictions completed for {dataset_name} {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

class TrainedModel:
    """YOLOv8 model for training and prediction on waste detection datasets"""
    
    def __init__(self, model_size: str = DEFAULT_MODEL, dataset: str = 'trashnet'):
        """Initialize model for specific dataset
        
        Args:
            model_size: YOLOv8 model size (n, s, m, l, x)
            dataset: Dataset to use ('trashnet', 'trashnet_annotated', or 'taco')
        """
        if dataset not in ['trashnet', 'trashnet_annotated', 'taco']:
            raise ValueError(f"Dataset must be 'trashnet', 'trashnet_annotated', or 'taco', got {dataset}")
            
        self.dataset = dataset
        
        # Select appropriate training config and results directory
        if dataset == 'trashnet':
            self.training_config = TRASHNET_TRAINING_CONFIG
            self.results_dir = TRAINED_TRASHNET_DIR
            self.data_yaml = OUTPUT_DIR / "trashnet" / "dataset.yaml"
        elif dataset == 'trashnet_annotated':
            self.training_config = TRASHNET_ANNOTATED_TRAINING_CONFIG
            self.results_dir = TRAINED_TRASHNET_ANNOTATED_DIR
            self.data_yaml = OUTPUT_DIR / "trashnet_annotated" / "dataset.yaml"
        else:  # taco
            self.training_config = TACO_TRAINING_CONFIG
            self.results_dir = TRAINED_TACO_DIR
            self.data_yaml = OUTPUT_DIR / "taco" / "dataset.yaml"
        
        try:
            # For training, start with base model
            if model_size:
                model_name = f'yolov8{model_size}.pt'
                logger.info(f"Loading YOLOv8 model for {dataset} training: {model_name}")
                self.model = YOLO(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train(self, data_yaml: Path = None) -> object:
        """Train the model on its specific dataset
        
        Args:
            data_yaml: Optional override for dataset configuration file
            
        Returns:
            object: YOLOv8 training results
        """
        try:
            logger.info(f"Starting training on {self.dataset}")
            
            # Use class-specific data_yaml if none provided
            data_yaml = data_yaml if data_yaml else self.data_yaml
            
            results = self.model.train(
                data=str(data_yaml),
                epochs=self.training_config['epochs'],
                imgsz=IMG_SIZE,
                batch=self.training_config['batch_size'],
                optimizer=self.training_config['optimizer'],
                lr0=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay'],
                device=self.training_config['device'],
                project=str(self.results_dir),
                name=self.dataset,
                save_period=self.training_config['save_period'],
                exist_ok=True,
                plots=True
            )
            
            logger.info(f"Training completed for {self.dataset}")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, data_yaml: Path = None, split: str = 'test') -> object:
        """Run predictions using the trained model
        
        Args:
            data_yaml: Optional override for dataset configuration file
            split: Dataset split to use (train, val, test)
            
        Returns:
            object: YOLOv8 validation results
        """
        try:
            logger.info(f"Running predictions on {self.dataset} {split} set")
            
            # Use class-specific data_yaml if none provided
            data_yaml = data_yaml if data_yaml else self.data_yaml
            
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
                plots=True
            )
            
            logger.info(f"Predictions completed for {self.dataset} {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise