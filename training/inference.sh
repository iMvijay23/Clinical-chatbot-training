#!/bin/bash

# Activate conda environment
conda env list
conda activate llmtrain

# Hardcode the model path
MODEL_PATH="/data/solr/models/tobaccowatcher/checkpoints_oct16/checkpoint-16999"

# Check if --quantize option is provided
USE_QUANTIZE=0
if [ "$#" -eq 1 ]; then
    if [ "$1" == "--quantize" ]; then
        USE_QUANTIZE=1
    else
        echo "Invalid option $1"
        exit 1
    fi
fi

# Run the Python script
python inference.py --model_path "$MODEL_PATH" --use_quantize $USE_QUANTIZE
