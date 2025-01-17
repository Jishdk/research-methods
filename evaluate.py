import logging
from pathlib import Path
import json
from typing import Dict
from config import BASELINE_RESULTS_DIR, TRAINED_RESULTS_DIR, METRICS

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for YOLO detection model"""
    
    def __init__(self):
        """Initialize model evaluator"""
        BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        TRAINED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
    def evaluate_baseline(self, results: object) -> Dict:
        """Extract basic metrics from baseline evaluation
        
        Args:
            results: YOLO Results object from model validation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Extract basic metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
            
            # Create dataset-specific directory
            dataset_dir = BASELINE_RESULTS_DIR / "trashnet"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics to JSON
            metrics_file = dataset_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            logger.info(f"Saved TrashNet baseline metrics to {metrics_file}")
            
            # Save confusion matrix if available
            if hasattr(results, 'confusion_matrix'):
                matrix_file = dataset_dir / "confusion_matrix.json"
                matrix_data = {
                    'matrix': results.confusion_matrix.matrix.tolist(),
                    'names': results.names
                }
                with open(matrix_file, 'w') as f:
                    json.dump(matrix_data, f, indent=4)
                logger.info(f"Saved confusion matrix to {matrix_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating TrashNet baseline: {e}")
            return {'error': str(e)}
            
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics for TrashNet
        
        Returns:
            Dictionary containing all available metrics
        """
        metrics_file = BASELINE_RESULTS_DIR / "trashnet" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return {}

class TrainingEvaluator:
    """Evaluator for trained TrashNet detection model"""
    
    def __init__(self):
        """Initialize training evaluator"""
        self.results_dir = TRAINED_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_training(self, results: object) -> Dict:
        """Evaluate training results
        
        Args:
            results: YOLO training Results object
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Extract training metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'training_epochs': results.epoch,
                'training_time': results.t,
                'best_epoch': results.best_epoch
            }
            
            # Save metrics
            metrics_dir = self.results_dir / "trashnet"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / "training_metrics.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Saved TrashNet training metrics to {metrics_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating TrashNet training: {e}")
            return {'error': str(e)}
    
    def evaluate_predictions(self, results: object) -> Dict:
        """Evaluate model predictions after training
        
        Args:
            results: YOLO validation Results object
            
        Returns:
            Dictionary containing prediction metrics
        """
        try:
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
            
            # Save metrics
            metrics_dir = self.results_dir / "trashnet"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / "prediction_metrics.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Saved TrashNet prediction metrics to {metrics_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {'error': str(e)}