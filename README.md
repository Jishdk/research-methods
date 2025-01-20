# Assessing YOLOv8’s Capability in Multi-Class Household Waste Detection and Classification Under Real-World Conditions

# Multi-Class Waste Detection Using YOLOv8

This project implements a waste detection and classification system using YOLOv8. We evaluate the model's performance on the TrashNet, TACO, and annotated TrashNet datasets to assess its capability in real-world conditions.

## Directory Structure
```
.
├── config.py          # Configuration settings
├── evaluate.py        # Evaluation scripts
├── main.py           # Main execution script
├── model.py          # Model definitions
├── preprocessing.py   # Data preprocessing
├── utils.py          # Utility functions
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Place datasets in the following structure:
```
data/
├── data_trashnet/           # Original TrashNet dataset
├── data_trashnet_annotated/ # Annotated TrashNet dataset
└── data_taco/              # TACO dataset
```

2. The preprocessed data will be stored in:
```
preprocessed_data/
├── trashnet/
├── trashnet_annotated/
└── taco/
```

## Running the Code

1. Run the complete pipeline:
```bash
python main.py
```

This will:
- Preprocess all datasets
- Run baseline evaluations
- Train models
- Generate visualizations

## Results

Results are stored in:
```
results/
├── baseline/      # Baseline model results
└── trained/       # Trained model results
    ├── trashnet/
    ├── trashnet_annotated/
    └── taco/
```

Each results directory contains:
- Metrics (mAP, precision, recall)
- Training curves
- Example predictions
- Class distribution plots

## Authors
- Eva Koenders
- Jishnu Harinandansingh
- Michel Marien

Research Methods for AI
Open University of Netherlands
2024
