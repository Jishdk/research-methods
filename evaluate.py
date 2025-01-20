import logging
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from config import (
    BASELINE_RESULTS_DIR, TRAINED_RESULTS_DIR, METRICS,
    PLOT_CONFIG, FIGURE_SIZES, EXAMPLES_PER_CLASS
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for object detection models"""
    
    def __init__(self):
        """Initialize model evaluator"""
        BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        TRAINED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn')
        sns.set_palette(PLOT_CONFIG['color_palette'])
        plt.rcParams['figure.figsize'] = FIGURE_SIZES['learning_curves']
        plt.rcParams['font.size'] = PLOT_CONFIG['font_size']
        
    def evaluate_baseline(self, results: object, dataset_name: str) -> Dict:
        """Extract metrics from baseline evaluation
        
        Args:
            results: YOLO Results object from model validation
            dataset_name: Name of dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Extract metrics
            metrics = self._extract_metrics(results)
            
            # Create dataset-specific directory
            dataset_dir = BASELINE_RESULTS_DIR / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            self._save_metrics(metrics, dataset_dir / "metrics.json")
            
            # Save confusion matrix if available
            if hasattr(results, 'confusion_matrix'):
                self._save_confusion_matrix(results, dataset_dir)
                self._plot_confusion_matrix(results, dataset_dir)
            
            # Plot precision-recall curves
            self._plot_precision_recall_curves(results, dataset_dir)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name} baseline: {e}")
            return {'error': str(e)}
            
    def get_metrics_summary(self, dataset_name: str) -> Dict:
        """Get summary of metrics for a dataset"""
        metrics_file = BASELINE_RESULTS_DIR / dataset_name / "metrics.json"
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error reading metrics for {dataset_name}: {e}")
            return {'error': str(e)}
    
    def _extract_metrics(self, results: object) -> Dict:
        """Extract all relevant metrics from results object"""
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr)
        }
        
        # Calculate F1 score
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (
            metrics['precision'] + metrics['recall']
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        return metrics
    
    def _save_metrics(self, metrics: Dict, file_path: Path) -> None:
        """Save metrics to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {file_path}")
    
    def _save_confusion_matrix(self, results: object, output_dir: Path) -> None:
        """Save confusion matrix data"""
        matrix_data = {
            'matrix': results.confusion_matrix.matrix.tolist(),
            'names': results.names
        }
        matrix_file = output_dir / "confusion_matrix.json"
        with open(matrix_file, 'w') as f:
            json.dump(matrix_data, f, indent=4)
    
    def _plot_confusion_matrix(self, results: object, output_dir: Path) -> None:
        """Plot and save confusion matrix"""
        plt.figure(figsize=FIGURE_SIZES['confusion_matrix'])
        sns.heatmap(
            results.confusion_matrix.matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=results.names,
            yticklabels=results.names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=PLOT_CONFIG['dpi'])
        plt.close()
    
    def _plot_precision_recall_curves(self, results: object, output_dir: Path) -> None:
        """Plot precision-recall curves"""
        plt.figure(figsize=FIGURE_SIZES['precision_recall'])
        
        if hasattr(results, 'pr_curve'):
            for i, class_name in enumerate(results.names):
                precision = results.pr_curve[i, :, 0]
                recall = results.pr_curve[i, :, 1]
                plt.plot(recall, precision, linewidth=PLOT_CONFIG['line_width'],
                        label=f'{class_name} (AP={results.box.ap50[i]:.2f})')
        
        plt.grid(PLOT_CONFIG['grid'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=PLOT_CONFIG['dpi'])
        plt.close()

class TrainingEvaluator:
    """Evaluator for trained models"""
    
    def __init__(self):
        """Initialize training evaluator"""
        self.results_dir = TRAINED_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_training(self, results: object, dataset_name: str) -> Dict:
        """Evaluate training results and create visualizations
        
        Args:
            results: YOLO training Results object
            dataset_name: Name of dataset
            
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
                'f1': 2 * float(results.box.mp) * float(results.box.mr) / (
                    float(results.box.mp) + float(results.box.mr)
                ) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0,
                'training_epochs': len(results.epoch) if hasattr(results, 'epoch') else None,
                'training_time': results.t if hasattr(results, 't') else None,
                'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else None
            }
            
            # Save metrics
            metrics_dir = self.results_dir / dataset_name
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / "training_metrics.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Plot training curves if epoch data is available
            if hasattr(results, 'epoch'):
                self._plot_training_history(results, metrics_dir)
            
            logger.info(f"Saved {dataset_name} training metrics to {metrics_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name} training: {e}")
            return {'error': str(e)}
    
    def evaluate_predictions(self, results: object, dataset_name: str) -> Dict:
        """Evaluate model predictions and create visualizations"""
        try:
            # Extract prediction metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
            
            # Calculate F1 score
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (
                metrics['precision'] + metrics['recall']
            ) if (metrics['precision'] + metrics['recall']) > 0 else 0
            
            # Save metrics
            metrics_dir = self.results_dir / dataset_name
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / "prediction_metrics.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Create visualizations
            if hasattr(results, 'confusion_matrix'):
                self._plot_confusion_matrix(results, metrics_dir)
            self._plot_precision_recall_curves(results, metrics_dir)
            
            logger.info(f"Saved {dataset_name} prediction metrics to {metrics_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {'error': str(e)}
    
    def _plot_training_history(self, results: object, output_dir: Path) -> None:
        """Plot training history curves"""
        epochs = range(1, len(results.epoch) + 1)
        
        # Plot training and validation loss
        plt.figure(figsize=FIGURE_SIZES['learning_curves'])
        plt.plot(epochs, results.loss, 'b-', label='Training Loss', 
                linewidth=PLOT_CONFIG['line_width'])
        if hasattr(results, 'val_loss'):
            plt.plot(epochs, results.val_loss, 'r-', label='Validation Loss',
                    linewidth=PLOT_CONFIG['line_width'])
        plt.grid(PLOT_CONFIG['grid'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_curves.png', dpi=PLOT_CONFIG['dpi'])
        plt.close()
        
        # Plot metrics
        plt.figure(figsize=FIGURE_SIZES['learning_curves'])
        for metric in METRICS:
            if hasattr(results, metric.lower()):
                values = getattr(results, metric.lower())
                plt.plot(epochs, values, label=metric, linewidth=PLOT_CONFIG['line_width'])
        plt.grid(PLOT_CONFIG['grid'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_curves.png', dpi=PLOT_CONFIG['dpi'])
        plt.close()
    
    def _plot_confusion_matrix(self, results: object, output_dir: Path) -> None:
        """Plot confusion matrix"""
        plt.figure(figsize=FIGURE_SIZES['confusion_matrix'])
        sns.heatmap(
            results.confusion_matrix.matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=results.names,
            yticklabels=results.names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=PLOT_CONFIG['dpi'])
        plt.close()
    
    def _plot_precision_recall_curves(self, results: object, output_dir: Path) -> None:
        """Plot precision-recall curves"""
        plt.figure(figsize=FIGURE_SIZES['precision_recall'])
        
        if hasattr(results, 'pr_curve'):
            for i, class_name in enumerate(results.names):
                precision = results.pr_curve[i, :, 0]
                recall = results.pr_curve[i, :, 1]
                plt.plot(recall, precision, linewidth=PLOT_CONFIG['line_width'],
                        label=f'{class_name} (AP={results.box.ap50[i]:.2f})')
        
        plt.grid(PLOT_CONFIG['grid'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=PLOT_CONFIG['dpi'])
        plt.close()