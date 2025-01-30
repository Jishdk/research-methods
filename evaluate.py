# evaluate.py

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from utils import setup_logging

logger = setup_logging()

class ModelEvaluator:
    """Handles evaluation of baseline model performance"""
    
    def __init__(self):
        """Initialize evaluator with baseline results directory"""
        self.results_dir = BASELINE_RESULTS_DIR
        
    def evaluate_baseline(self, results: Any, dataset: str) -> Dict:
        """Evaluate baseline model performance on TrashNet dataset only
        
        Args:
            results: YOLO Results object from validation
            dataset: Name of dataset (only 'trashnet' supported for baseline)
            
        Returns:
            Dict containing evaluation metrics
        """
        if dataset != 'trashnet':
            logger.info(f"Skipping baseline evaluation for {dataset} (baseline only for TrashNet)")
            return {}
            
        try:
            # Extract comprehensive metrics including per-class performance
            metrics = {
                'dataset': dataset,
                'model_type': 'baseline',
                'overall': {
                    'mAP50': float(results.maps[0]),
                    'mAP50-95': float(results.maps[1]),
                    'precision': float(results.results_dict['metrics/precision(B)']),
                    'recall': float(results.results_dict['metrics/recall(B)'])
                },
                'per_class': {}
            }
            
            # Add per-class metrics if available
            if hasattr(results, 'names') and hasattr(results, 'metrics'):
                for i, name in enumerate(results.names):
                    metrics['per_class'][name] = {
                        'precision': float(results.metrics.precision[i]),
                        'recall': float(results.metrics.recall[i]),
                        'mAP50': float(results.metrics.map50[i])
                    }
            
            # Save metrics in both JSON and CSV formats
            self._save_metrics(metrics, dataset)
            
            # Create visualization
            self._plot_baseline_metrics(metrics['overall'], dataset)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating baseline model: {e}")
            return {}
    
    def _save_metrics(self, metrics: Dict, dataset: str) -> None:
        """Save metrics in both JSON and CSV formats
        
        Args:
            metrics: Dictionary containing evaluation metrics
            dataset: Name of dataset being evaluated
        """
        save_dir = self.results_dir / dataset
        
        # Save detailed metrics as JSON
        json_file = save_dir / "baseline_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Save summary metrics as CSV for compatibility
        csv_file = save_dir / "baseline_metrics.csv"
        pd.DataFrame([metrics['overall']]).to_csv(csv_file, index=False)
            
    def _plot_baseline_metrics(self, metrics: Dict, dataset: str) -> None:
        """Create visualization of baseline metrics
        
        Args:
            metrics: Dictionary of overall metrics to plot
            dataset: Name of dataset being evaluated
        """
        plt.figure(figsize=FIGURE_SIZES['class_performance'])
        
        # Create bar plot
        metrics_values = [metrics[m] for m in METRICS]
        plt.bar(METRICS, metrics_values)
        
        plt.title(f'Baseline Model Performance on {dataset.capitalize()}')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.results_dir / dataset / "baseline_metrics.png")
        plt.close()

class TrainingEvaluator:
    """Handles evaluation of trained model performance"""
    
    def __init__(self):
        """Initialize evaluator with results directory for trained models"""
        self.results_dir = TRAINED_RESULTS_DIR
        
    def evaluate_training(self, results: Any, dataset: str) -> Dict:
        """Evaluate training performance and create visualizations
        
        Args:
            results: YOLO Results object from training
            dataset: Name of dataset used for training
            
        Returns:
            Dict containing training metrics
        """
        try:
            # Extract comprehensive training metrics
            metrics = {
                'dataset': dataset,
                'training_history': {
                    'epoch': list(range(1, len(results.metrics['train/box_loss']) + 1)),
                    'train_loss': results.metrics['train/box_loss'],
                    'val_loss': results.metrics['val/box_loss'],
                    'mAP50': results.metrics['metrics/mAP50(B)'],
                    'mAP50-95': results.metrics['metrics/mAP50-95(B)'],
                    'precision': results.metrics['metrics/precision(B)'],
                    'recall': results.metrics['metrics/recall(B)']
                },
                'final_metrics': {
                    'mAP50': float(results.metrics['metrics/mAP50(B)'][-1]),
                    'mAP50-95': float(results.metrics['metrics/mAP50-95(B)'][-1]),
                    'precision': float(results.metrics['metrics/precision(B)'][-1]),
                    'recall': float(results.metrics['metrics/recall(B)'][-1])
                }
            }
            
            # Save both detailed JSON and summary CSV
            save_dir = self._get_save_dir(dataset)
            
            # Save JSON with full training history
            json_file = save_dir / "training_metrics.json"
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            # Save CSV with final metrics for compatibility
            csv_file = save_dir / "training_metrics.csv"
            pd.DataFrame([metrics['final_metrics']]).to_csv(csv_file, index=False)
            
            # Create training visualizations
            self._plot_training_curves(metrics['training_history'], dataset)
            self._plot_metrics_evolution(metrics['training_history'], dataset)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating training: {e}")
            return {}
            
    def evaluate_predictions(self, results: Any, dataset: str) -> Dict:
        """Evaluate model predictions on test set
        
        Args:
            results: YOLO Results object from validation
            dataset: Name of dataset being evaluated
            
        Returns:
            Dict containing prediction metrics
        """
        try:
            # Extract comprehensive metrics
            metrics = {
                'dataset': dataset,
                'overall': {
                    'mAP50': float(results.maps[0]),
                    'mAP50-95': float(results.maps[1]),
                    'precision': float(results.results_dict['metrics/precision(B)']),
                    'recall': float(results.results_dict['metrics/recall(B)'])
                },
                'per_class': {}
            }
            
            # Add per-class metrics if available
            if hasattr(results, 'names') and hasattr(results, 'metrics'):
                for i, name in enumerate(results.names):
                    metrics['per_class'][name] = {
                        'precision': float(results.metrics.precision[i]),
                        'recall': float(results.metrics.recall[i]),
                        'mAP50': float(results.metrics.map50[i])
                    }
            
            # Save both JSON and CSV formats
            save_dir = self._get_save_dir(dataset)
            
            # Save detailed metrics as JSON
            json_file = save_dir / "prediction_metrics.json"
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            # Save summary metrics as CSV for compatibility
            csv_file = save_dir / "test_metrics.csv"
            pd.DataFrame([metrics['overall']]).to_csv(csv_file, index=False)
            
            logger.info(f"Saved prediction metrics for {dataset}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {}
            
    def _get_save_dir(self, dataset: str) -> Path:
        """Get appropriate save directory for each dataset
        
        Args:
            dataset: Name of dataset
            
        Returns:
            Path to dataset-specific results directory
        """
        if dataset == 'trashnet':
            return TRAINED_TRASHNET_DIR
        elif dataset == 'trashnet_annotated':
            return TRAINED_TRASHNET_ANNOTATED_DIR
        else:  # taco
            return TRAINED_TACO_DIR
            
    def _plot_training_curves(self, metrics: Dict, dataset: str) -> None:
        """Plot training and validation loss curves
        
        Args:
            metrics: Dictionary containing training metrics history
            dataset: Name of dataset being evaluated
        """
        plt.figure(figsize=FIGURE_SIZES['learning_curves'])
        
        plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
        
        plt.title(f'Training Curves - {dataset.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Save plot
        save_dir = self._get_save_dir(dataset)
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png")
        plt.close()
        
    def _plot_metrics_evolution(self, metrics: Dict, dataset: str) -> None:
        """Plot evolution of evaluation metrics during training
        
        Args:
            metrics: Dictionary containing training metrics history
            dataset: Name of dataset being evaluated
        """
        plt.figure(figsize=FIGURE_SIZES['learning_curves'])
        
        for metric in ['mAP50', 'precision', 'recall']:
            if metric in metrics:
                plt.plot(metrics['epoch'], metrics[metric], label=metric)
        
        plt.title(f'Metrics Evolution - {dataset.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.ylim(0, 1)
        
        # Save plot
        save_dir = self._get_save_dir(dataset)
        plt.tight_layout()
        plt.savefig(save_dir / "metrics_evolution.png")
        plt.close()