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
def train_model(dataset,
                model,
                train_params,
                project = f"yolo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                output_dir        = OUTPUT_DIR / 'models'):

  # Define dataset directory
  dataset_dir = OUTPUT_DIR / dataset

  # Check if yolo format
  if not check_train_folder(dataset_dir):
    print("Folder structure and YAML file are not valid.")
    return

  yaml_dir    = dataset_dir / 'dataset.yaml'

  # Set output dir
  output_dir.mkdir(exist_ok=True, parents=True) #make model file if does not excist
  output_dir = output_dir / dataset
  output_dir.mkdir(exist_ok=True, parents=True) #make dataset file if does not excist
  output_dir = output_dir / project
  output_dir.mkdir(exist_ok=True, parents=True)

  # Load model
  model = YOLO(model)

  # Train model
  model.train(data = str(yaml_dir),
              project = str(output_dir),
              name = f"train_results",
              **train_params)

  # Evaluate model on the validation set
  metrics = model.val(data=str(yaml_dir))

  # Save the model
  model_path = output_dir / f"{model}.pt"
  model.export(format='torchscript')  # Save the model as .pt file

  print(f"Trained model and saved to {model_path}")
  print(f"Results saved to {output_dir}")

  return metrics  # Return metrics


# Define paths (update these paths based on your file structure)
def tune_model(dataset,
               model,
               train_params,
               project = f"yolo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
               output_dir        = OUTPUT_DIR / 'models'):

  # Define dataset directory
  dataset_dir = OUTPUT_DIR / dataset

  # Check if yolo format
  if not check_train_folder(dataset_dir):
    print("Folder structure and YAML file are not valid.")
    return

  yaml_dir    = dataset_dir / 'dataset.yaml'

  # Set output dir
  output_dir.mkdir(exist_ok=True, parents=True) #make model file if does not excist
  output_dir = output_dir / dataset
  output_dir.mkdir(exist_ok=True, parents=True) #make dataset file if does not excist
  output_dir = output_dir / project
  output_dir.mkdir(exist_ok=True, parents=True)

  # Load model
  model = YOLO(model)  # Using YOLOv8 Nano, change to 'yolov8s.pt' or others if needed

  # Tune model
  model.tune(data = str(yaml_dir),
              project = str(output_dir),
              name = f"tune_results",
              **train_params, verbose = True)

  best_params = model.best  # Access the best hyperparameters (depending on model version)

  # Save best parameters to a YAML file
  #best_params_path = output_dir / "best_params.yaml"
  #with open(best_params_path, "w") as file:
  #    yaml.dump(best_params, file)

  #print(f"Best hyperparameters saved to {best_params_path}")

  return model.best  # Return the best hyperparameters

#def cross_val_model(dataset_path, model, train_params, 
#                    project = f"yolo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", 
#                    output_dir = OUTPUT_DIR / 'models'):
