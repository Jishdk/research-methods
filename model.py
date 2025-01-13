# model.py
from pathlib import Path
import logging
from ultralytics import YOLO
from config import *
from utils import setup_logging

logger = setup_logging()

class BaselineModel:
    """Baseline YOLO model without training"""
    
    def __init__(self, model_size: str = 'n'):
        """Initialize baseline model with pretrained weights"""
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading pretrained YOLOv8 model: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def predict(self, data_yaml: Path, split: str = 'test') -> object:
        """Run predictions using the baseline model"""
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
                exist_ok=True
            )
            
            logger.info(f"Predictions completed for {dataset_name} {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

class TrainedModel:
    """YOLO model for training on garbage detection"""
    
    def __init__(self, model_size: str = 'n', dataset: str = 'trashnet'):
        """Initialize model for training
        
        Args:
            model_size: YOLOv8 model size (n, s, m, l, x)
            dataset: Dataset to train on ('trashnet' or 'taco')
        """
        if dataset not in ['trashnet', 'taco']:
            raise ValueError(f"Dataset must be 'trashnet' or 'taco', got {dataset}")
            
        self.dataset = dataset
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading YOLOv8 model for {dataset} training: {model_name}")
        
        # Select appropriate training config
        self.training_config = (TRASHNET_TRAINING_CONFIG 
                              if dataset == 'trashnet' 
                              else TACO_TRAINING_CONFIG)
        
        # Select appropriate results directory
        self.results_dir = (TRAINED_TRASHNET_DIR 
                          if dataset == 'trashnet' 
                          else TRAINED_TACO_DIR)
        
        try:
            self.model = YOLO(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train(self, data_yaml: Path) -> object:
        """Train the model on specified dataset"""
        try:
            logger.info(f"Starting training on {self.dataset}")
            
            # Train model using dataset-specific config
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
                exist_ok=True
            )
            
            logger.info(f"Training completed for {self.dataset}")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, data_yaml: Path, split: str = 'test') -> object:
        """Run predictions using the trained model"""
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
                exist_ok=True
            )
            
            logger.info(f"Predictions completed for {self.dataset} {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise