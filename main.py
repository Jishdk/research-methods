import logging
from pathlib import Path
from typing import Dict, Tuple, List
from config import (
    OUTPUT_DIR, RESULTS_DIR, BASELINE_RESULTS_DIR, TRAINED_RESULTS_DIR,
    DATASET_DIRS
)
from preprocessing import DatasetPreprocessor
from utils import setup_logging, save_example_images
from model import BaselineModel, TrainedModel
from evaluate import ModelEvaluator, TrainingEvaluator

logger = setup_logging()

def check_dataset_exists(dataset_name: str) -> bool:
    """Check if dataset is already processed"""
    yaml_path = OUTPUT_DIR / dataset_name / "dataset.yaml"
    return yaml_path.exists()

def run_preprocessing() -> bool:
    """Run preprocessing pipeline for all datasets"""
    try:
        logger.info("Initializing data preprocessor...")
        preprocessor = DatasetPreprocessor()
        
        # List of all datasets to process
        datasets = ['trashnet', 'taco', 'trashnet_annotated']
        
        for dataset in datasets:
            if not check_dataset_exists(dataset):
                logger.info(f"Processing {dataset} dataset...")
                
                # Process dataset based on type
                if dataset == 'trashnet':
                    preprocessor.process_trashnet()
                elif dataset == 'taco':
                    preprocessor.process_taco()
                else:  # trashnet_annotated
                    preprocessor.process_trashnet_annotated()
                
                # Create folds and save metadata
                preprocessor.create_cross_validation_folds(dataset)
                preprocessor.log_dataset_stats(dataset)
                preprocessor.save_dataset_metadata(dataset)
                
                # Save example images
                save_example_images(dataset)
            else:
                logger.info(f"{dataset} dataset already processed, skipping...")
        
        logger.info(f"Preprocessing complete. Output saved to {OUTPUT_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False

def run_baseline_evaluation() -> Tuple[Dict, bool]:
    """Run baseline model evaluation on all datasets"""
    try:
        logger.info("Starting baseline model evaluation")
        model = BaselineModel()
        evaluator = ModelEvaluator()
        
        baseline_metrics = {}
        success = True
        
        # Evaluate all datasets
        for dataset in DATASET_DIRS.keys():
            logger.info(f"Evaluating baseline model on {dataset}...")
            data_yaml = OUTPUT_DIR / dataset / "dataset.yaml"
            
            try:
                results = model.predict(data_yaml)
                metrics = evaluator.evaluate_baseline(results, dataset)
                baseline_metrics[dataset] = metrics
                
                logger.info(f"{dataset} baseline mAP50: {metrics.get('mAP50', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Error evaluating {dataset}: {e}")
                baseline_metrics[dataset] = {'error': str(e)}
                success = False
        
        logger.info(f"Baseline evaluation complete. Results saved to {BASELINE_RESULTS_DIR}")
        return baseline_metrics, success
        
    except Exception as e:
        logger.error(f"Error during baseline evaluation: {str(e)}")
        return {'error': str(e)}, False

def run_model_training() -> Tuple[Dict, bool]:
    """Run model training on all datasets"""
    try:
        logger.info("Starting model training")
        training_evaluator = TrainingEvaluator()
        training_metrics = {}
        success = True
        
        # Train on each dataset separately
        for dataset in DATASET_DIRS.keys():
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
                success = False
        
        logger.info(f"Training complete. Results saved to {TRAINED_RESULTS_DIR}")
        return training_metrics, success
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return {'error': str(e)}, False

def save_run_config():
    """Save current configuration for reproducibility"""
    config_file = RESULTS_DIR / "run_config.txt"
    with open(config_file, 'w') as f:
        f.write("TRAINING CONFIGURATION:\n")
        from config import TRAINING_CONFIG
        for key, value in TRAINING_CONFIG.items():
            f.write(f"{key}: {value}\n")

def main():
    """Main execution pipeline
    
    Pipeline steps:
    1. Preprocess all datasets (if needed)
    2. Run baseline evaluation
    3. Run model training
    4. Save configuration
    """
    try:
        # Step 1: Preprocessing
        if not all(check_dataset_exists(d) for d in DATASET_DIRS.keys()):
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
        
        # Print summary of results
        for dataset in DATASET_DIRS.keys():
            print(f"\nResults for {dataset}:")
            print(f"Baseline mAP50: {baseline_metrics[dataset].get('mAP50', 'N/A')}")
            print(f"Training mAP50: {training_metrics[dataset].get('mAP50', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()