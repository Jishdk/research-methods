# evaluate.py

import logging
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
        self.results_dir = BASELINE_RESULTS_DIR
        
    def evaluate_baseline(self, results: Any, dataset: str) -> Dict:
        """Evaluate baseline model performance on TrashNet dataset only"""
        if dataset != 'trashnet':
            logger.info(f"Skipping baseline evaluation for {dataset} (baseline only for TrashNet)")
            return {}
            
        try:
            # Extract metrics from results
            metrics = {
                'mAP50': results.maps[0],  # mAP at IoU=0.50
                'mAP50-95': results.maps[1],  # mAP at IoU=0.50:0.95
                'precision': results.results_dict['metrics/precision(B)'],
                'recall': results.results_dict['metrics/recall(B)']
            }
            
            # Save baseline metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(self.results_dir / dataset / "baseline_metrics.csv", index=False)
            
            # Create visualization
            self._plot_baseline_metrics(metrics, dataset)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating baseline model: {e}")
            return {}
            
    def _plot_baseline_metrics(self, metrics: Dict, dataset: str) -> None:
        """Create bar plot of baseline metrics"""
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
    """Handles evaluation of training performance"""
    
    def __init__(self):
        self.results_dir = TRAINED_RESULTS_DIR
        
    def evaluate_training(self, results: Any, dataset: str) -> Dict:
        """Evaluate training performance and create visualizations"""
        try:
            # Extract training history
            metrics = {
                'epoch': list(range(1, len(results.metrics['train/box_loss']) + 1)),
                'train_loss': results.metrics['train/box_loss'],
                'val_loss': results.metrics['val/box_loss'],
                'mAP50': results.metrics['metrics/mAP50(B)'],
                'mAP50-95': results.metrics['metrics/mAP50-95(B)'],
                'precision': results.metrics['metrics/precision(B)'],
                'recall': results.metrics['metrics/recall(B)']
            }
            
            # Save training metrics
            metrics_df = pd.DataFrame(metrics)
            save_dir = self._get_save_dir(dataset)
            metrics_df.to_csv(save_dir / "training_metrics.csv", index=False)
            
            # Create training visualizations
            self._plot_training_curves(metrics, dataset)
            self._plot_metrics_evolution(metrics, dataset)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating training: {e}")
            return {}
            
    def evaluate_predictions(self, results: Any, dataset: str) -> Dict:
        """Evaluate model predictions on test set"""
        try:
            metrics = {
                'mAP50': results.maps[0],
                'mAP50-95': results.maps[1],
                'precision': results.results_dict['metrics/precision(B)'],
                'recall': results.results_dict['metrics/recall(B)']
            }
            
            # Save test metrics
            save_dir = self._get_save_dir(dataset)
            pd.DataFrame([metrics]).to_csv(save_dir / "test_metrics.csv", index=False)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {}
            
    def _get_save_dir(self, dataset: str) -> Path:
        """Get appropriate save directory based on dataset"""
        if dataset == 'trashnet':
            return TRAINED_TRASHNET_DIR
        elif dataset == 'trashnet_annotated':
            return TRAINED_TRASHNET_ANNOTATED_DIR
        else:
            return TRAINED_TACO_DIR
            
    def _plot_training_curves(self, metrics: Dict, dataset: str) -> None:
        """Plot training and validation loss curves"""
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
        """Plot evolution of evaluation metrics during training"""
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