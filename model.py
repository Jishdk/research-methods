import os
import json
from typing import Dict, List, Tuple
import re
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from config import OUTPUT_DIR, IMG_SIZE, BATCH, N_EPOCHS, MODEL

# Define paths (update these paths based on your file structure)
def train_model(dataset, cross_val: False, output_dir = OUTPUT_DIR / 'models', 
                yolo_mod = MODEL, epochs = N_EPOCHS, batch = BATCH, imgsz = IMG_SIZE):

  ### Define directories
  if not dataset in ['taco', 'trashnet']:
    return dataset + " dataset not supported"
  
  dataset_dir = OUTPUT_DIR / dataset
 
  #check cross validation folders
  if cross_val:
    folds = os.listdir(dataset_dir / "cv_splits")
    folds = [sp for sp in folds if not(re.search('\.csv', sp))]

    #check structure of all folds
    for fold in folds:
      if os.path.isdir(dataset_dir / "cv_splits"):
        if not(check_train_folder(dataset_dir / "cv_splits" / fold)):
          return f"fold {fold} is not in valid yolo train format"
  else:
    if os.path.isdir(dataset_dir):
      if not(check_train_folder(dataset_dir)):
        return "dataset is not in valid yolo train format"

  #set and check test folder
  #test_set    = dataset_dir / "test"
  #test_yaml   = dataset_dir / 'dataset.yaml'

  #if not(os.path.isdir(test_set)):
  #  return f'testset is missing in {dataset_dir}'
  #if not(os.path.isdir(test_yaml)):
  #  return f'yaml file is missing in {dataset_dir}'

  #set output dir
  output_dir.mkdir(exist_ok=True, parents=True) #make model file if does not excist
  output_dir = output_dir / dataset
  output_dir.mkdir(exist_ok=True, parents=True) #make dataset file if does not excist
  output_dir = output_dir / f"yolo_{cross_val}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
  output_dir.mkdir(exist_ok=True, parents=True)

  ### Train model
  results = {}

  for idx, fold in enumerate(folds):

    print(f"Training for {fold}...")
    fold_path = dataset_dir / "cv_splits" / fold

    # Paths to train and validation YAML files
    fold_yaml = fold_path / 'dataset.yaml'

    # Initialize and train the YOLOv8 model
    model = YOLO(yolo_mod)  # Using YOLOv8 Nano, change to 'yolov8s.pt' or others if needed

    model.train(
        data=str(fold_yaml),        # Path to the training YAML file
        epochs = epochs,            # Number of epochs to train
        imgsz  = imgsz,             # Image size
        batch  = batch,             # Batch size
        project=str(output_dir),    # Directory to save results
        name=f"{fold}_results",     # Name of the folder for this fold
        val=True                    # Path to the validation YAML file
    )

    # Evaluate model on the validation set
    metrics = model.val(data=str(fold_yaml))
    results[fold] = metrics  # Store metrics for this fold

    # Save the model
    model_path = output_dir / f"{model}_{fold}.pt"
    model.export(format='torchscript', path=str(model_path))  # Save the model as .pt file

  # Save results
  results_file = output_dir / "results.json"
  with open(results_file, 'w') as f:
      json.dump(results, f, indent=4)

  return "Training was succesfull"
