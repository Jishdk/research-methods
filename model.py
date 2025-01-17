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
            logger.info(f"Running predictions on TrashNet {split} set")
            
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
                name="trashnet",
                exist_ok=True
            )
            
            logger.info(f"Predictions completed for TrashNet {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

class TrainedModel:
    """YOLO model for training on TrashNet"""
    
    def __init__(self, model_size: str = 'n'):
        """Initialize model for training
        
        Args:
            model_size: YOLOv8 model size (n, s, m, l, x)
        """
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading YOLOv8 model for TrashNet training: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train(self, data_yaml: Path) -> object:
        """Train the model on TrashNet dataset"""
        try:
            logger.info("Starting training on TrashNet")
            
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
                project=str(TRAINED_TRASHNET_DIR),
                name="trashnet",
                save_period=TRAINING_CONFIG['save_period'],
                exist_ok=True
            )
            
            logger.info("Training completed for TrashNet")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, data_yaml: Path, split: str = 'test') -> object:
        """Run predictions using the trained model"""
        try:
            logger.info(f"Running predictions on TrashNet {split} set")
            
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
                project=str(TRAINED_TRASHNET_DIR),
                name="trashnet_predictions",
                exist_ok=True
            )
            
            logger.info(f"Predictions completed for TrashNet {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise