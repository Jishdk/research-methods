import logging
from pathlib import Path
from typing import Dict, Tuple
from config import (
    OUTPUT_DIR, RESULTS_DIR, BASELINE_RESULTS_DIR, TRAINED_RESULTS_DIR, 
    TRAINING_CONFIG
)
from preprocessing import DatasetPreprocessor
from utils import setup_logging, plot_training_curves
from model import BaselineModel, TrainedModel
from evaluate import ModelEvaluator, TrainingEvaluator

logger = setup_logging()

def check_dataset_exists() -> bool:
    """Check if TrashNet dataset is already processed"""
    yaml_path = OUTPUT_DIR / "trashnet" / "dataset.yaml"
    return yaml_path.exists()

def run_preprocessing() -> bool:
    """Run preprocessing pipeline for TrashNet dataset"""
    try:
        logger.info("Initializing data preprocessor...")
        preprocessor = DatasetPreprocessor()
        
        if not check_dataset_exists():
            logger.info("Processing TrashNet dataset...")
            preprocessor.process_trashnet()
            preprocessor.create_cross_validation_folds()
            preprocessor.log_dataset_stats()
            preprocessor.save_dataset_metadata()
        else:
            logger.info("TrashNet dataset already processed, skipping...")
        
        logger.info(f"Preprocessing complete. Output saved to {OUTPUT_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False

def run_baseline_evaluation() -> Tuple[Dict, bool]:
    """Run baseline model evaluation"""
    try:
        logger.info("Starting baseline model evaluation")
        model = BaselineModel()
        evaluator = ModelEvaluator()
        
        data_yaml = OUTPUT_DIR / "trashnet" / "dataset.yaml"
        
        try:
            results = model.predict(data_yaml)
            metrics = evaluator.evaluate_baseline(results)
            
            logger.info(f"TrashNet baseline mAP50: {metrics.get('mAP50', 'N/A')}")
            return metrics, True
            
        except Exception as e:
            logger.error(f"Error evaluating TrashNet: {e}")
            return {'error': str(e)}, False
        
    except Exception as e:
        logger.error(f"Error during baseline evaluation: {str(e)}")
        return {'error': str(e)}, False

def run_model_training() -> Tuple[Dict, bool]:
    """Run model training on TrashNet dataset"""
    try:
        logger.info("Starting model training")
        training_evaluator = TrainingEvaluator()
        
        data_yaml = OUTPUT_DIR / "trashnet" / "dataset.yaml"
        
        try:
            # Initialize and train model
            model = TrainedModel()
            training_results = model.train(data_yaml)
            
            # Evaluate training results
            metrics = training_evaluator.evaluate_training(training_results)
            
            # Plot training curves
            plot_training_curves(TRAINED_RESULTS_DIR, metrics)
            
            # Run predictions on test set
            test_results = model.predict(data_yaml)
            test_metrics = training_evaluator.evaluate_predictions(test_results)
            
            logger.info(f"TrashNet training mAP50: {metrics.get('mAP50', 'N/A')}")
            
            return metrics, True
            
        except Exception as e:
            logger.error(f"Error training TrashNet: {e}")
            return {'error': str(e)}, False
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return {'error': str(e)}, False

def save_run_config():
    """Save current configuration for reproducibility"""
    config_file = RESULTS_DIR / "run_config.txt"
    with open(config_file, 'w') as f:
        f.write("TRAINING CONFIGURATION:\n")
        for key, value in TRAINING_CONFIG.items():
            f.write(f"{key}: {value}\n")

def main():
    """Main execution pipeline
    
    Pipeline steps:
    1. Preprocess TrashNet dataset (if needed)
    2. Run baseline evaluation 
    3. Run model training 
    4. Save configuration
    """
    try:
        # Step 1: Preprocessing
        if not check_dataset_exists():
            if not run_preprocessing():
                raise RuntimeError("Preprocessing failed")
        
        # Step 2: Baseline Evaluation
        baseline_metrics, success = run_baseline_evaluation()
        if not success:
            raise RuntimeError("Baseline evaluation failed")
            
        # Step 3: Model Training
        training_metrics, success = run_model_training()
        if not success:
            raise RuntimeError("Model training failed")
        
        # Step 4: Save Configuration
        save_run_config()
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()