import os
import json
from typing import Dict, List, Tuple
import re
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from config import OUTPUT_DIR, IMG_SIZE, BATCH, N_EPOCHS, MODEL
import yaml

def check_train_folder(folder_path: Path) -> bool:
    """
    Check if a folder has the correct YOLO train format.

    Args:
        folder_path (Path): Path to the folder to check.

    Returns:
        bool: True if the folder structure and YAML file format are correct, False otherwise.
    """
    folders = os.listdir(folder_path)

    # Check for the YAML file
    yaml_file = folder_path / 'dataset.yaml'  # Assuming the file is named 'dataset.yaml'
    if not yaml_file.exists():
        print("Error: Missing dataset.yaml file.")
        return False

    # Validate the YAML file format
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        # Check if required keys exist
        required_keys = ['train', 'val', 'nc', 'names']
        if not all(key in data for key in required_keys):
            print(f"Error: dataset.yaml is missing one or more required keys: {required_keys}")
            return False
    except Exception as e:
        print(f"Error: Failed to parse dataset.yaml. {e}")
        return False

    # Check for the presence of 'train' and 'val' folders
    if not all(subfolder in folders for subfolder in ['train', 'val']):
        print("Error: Missing 'train' or 'val' folders.")
        return False

    # Optionally check for 'test' folder
    if 'test' in folders:
        test_folder = folder_path / 'test'
        if not test_folder.is_dir():
            print("Error: 'test' is not a directory.")
            return False

    # Check contents of 'train' and 'val' folders
    for subfolder in ['train', 'val']:
        subfolder_path = folder_path / subfolder
        if not subfolder_path.is_dir():
            print(f"Error: '{subfolder}' is not a directory.")
            return False
        # Check for 'images' and 'labels' subfolders
        if not all((subfolder_path / sub).is_dir() for sub in ['images', 'labels']):
            print(f"Error: '{subfolder}' folder is missing 'images' or 'labels' subfolders.")
            return False

    print("Folder structure and YAML file are valid.")
    return True

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
