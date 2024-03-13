#!/bin/bash

# Activate conda environment
conda env list
conda activate llmtrain

#pip install scikit-learn
#pip install torchmetrics
# Run the Python script
python merge_and_evaluate.py
