# main.py

import logging
from pathlib import Path
from typing import Dict, Tuple
from config import (
    OUTPUT_DIR, RESULTS_DIR, BASELINE_RESULTS_DIR, TRAINED_RESULTS_DIR,
    EXAMPLES_DIR
)
from preprocessing import DatasetPreprocessor
from utils import setup_logging, load_training_config
from model import BaselineModel, TrainedModel
from evaluate import ModelEvaluator, TrainingEvaluator

logger = setup_logging()

def check_dataset_exists(dataset_name: str) -> bool:
    """
    Check if dataset is already processed and available
    
    Args:
        dataset_name: Name of the dataset to check
        
    Returns:
        bool: True if dataset exists, False otherwise
    """
    yaml_path = OUTPUT_DIR / dataset_name / "dataset.yaml"
    return yaml_path.exists()

def run_preprocessing() -> bool:
    """
    Run preprocessing pipeline for all datasets and save examples
    
    Returns:
        bool: True if preprocessing successful, False otherwise
    """
    try:
        logger.info("Initializing data preprocessor...")
        preprocessor = DatasetPreprocessor()
        
        # Process each dataset
        for dataset in ['trashnet', 'trashnet_annotated', 'taco']:
            if not check_dataset_exists(dataset):
                logger.info(f"Processing {dataset} dataset...")
                getattr(preprocessor, f'process_{dataset}')()
                
                # Save dataset examples
                preprocessor.save_dataset_examples(dataset)
                
                # Create folds and save metadata
                preprocessor.create_cross_validation_folds(dataset)
                preprocessor.log_dataset_stats(dataset)
                preprocessor.save_dataset_metadata(dataset)
            else:
                logger.info(f"{dataset} dataset already processed, skipping...")
        
        logger.info(f"Preprocessing complete. Output saved to {OUTPUT_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False

def run_baseline_evaluation() -> Tuple[Dict, bool]:
    """
    Run baseline model evaluation on TrashNet dataset only
    
    Returns:
        Tuple[Dict, bool]: Dictionary of metrics and success flag
    """
    try:
        logger.info("Starting baseline model evaluation on TrashNet")
        model = BaselineModel()
        evaluator = ModelEvaluator()
        
        # Evaluate baseline model only on TrashNet
        data_yaml = OUTPUT_DIR / "trashnet" / "dataset.yaml"
        results = model.predict(data_yaml)
        metrics = evaluator.evaluate_baseline(results, 'trashnet')
        
        logger.info(f"TrashNet baseline mAP50: {metrics.get('mAP50', 'N/A')}")
        logger.info(f"Baseline evaluation complete. Results saved to {BASELINE_RESULTS_DIR}")
        
        return metrics, True
        
    except Exception as e:
        logger.error(f"Error during baseline evaluation: {str(e)}")
        return {'error': str(e)}, False

def run_model_training() -> Tuple[Dict, bool]:
    """
    Run model training on all datasets
    
    Returns:
        Tuple[Dict, bool]: Dictionary of training metrics and success flag
    """
    try:
        logger.info("Starting model training")
        training_evaluator = TrainingEvaluator()
        training_metrics = {}
        
        # Train on each dataset separately
        for dataset in ['trashnet', 'trashnet_annotated', 'taco']:
            logger.info(f"Training model on {dataset}...")
            data_yaml = OUTPUT_DIR / dataset / "dataset.yaml"
            
            try:
                # Initialize and train model
                model = TrainedModel(dataset=dataset)
                training_results = model.train(data_yaml)
                
                # Evaluate training results
                metrics = training_evaluator.evaluate_training(training_results, dataset)
                training_metrics[dataset] = metrics
                
                # Run predictions on test set
                test_results = model.predict(data_yaml)
                test_metrics = training_evaluator.evaluate_predictions(test_results, dataset)
                
                logger.info(f"{dataset} training mAP50: {metrics.get('mAP50', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Error training on {dataset}: {e}")
                training_metrics[dataset] = {'error': str(e)}
        
        logger.info(f"Training complete. Results saved to {TRAINED_RESULTS_DIR}")
        return training_metrics, True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return {'error': str(e)}, False

def save_run_config():
    """Save current configuration settings for reproducibility"""
    config_file = RESULTS_DIR / "run_config.txt"
    with open(config_file, 'w') as f:
        for dataset in ['trashnet', 'trashnet_annotated', 'taco']:
            config = load_training_config(dataset)
            f.write(f"\n{dataset.upper()} Training Config:\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

def main():
    """
    Main execution pipeline for waste detection project
    
    Pipeline steps:
    1. Preprocess datasets (if needed) and save examples
    2. Run baseline evaluation on TrashNet only
    3. Run model training on all datasets
    4. Save configuration
    """
    try:
        # Step 1: Preprocessing
        if not all(check_dataset_exists(d) for d in ['trashnet', 'trashnet_annotated', 'taco']):
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