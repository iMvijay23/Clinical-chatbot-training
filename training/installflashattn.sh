#!/bin/bash

conda env list
conda activate llmtrain
# Check if running in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Please activate your conda environment first!"
    exit 1
fi

# Install required dependencies
pip install packaging

# Check ninja
ninja --version
if [ $? -ne 0 ]; then
    pip uninstall -y ninja
    pip install ninja
fi

# Install flash-attn
# Assuming average specs for MAX_JOBS, adjust if needed
MAX_JOBS=4 pip install flash-attn --no-build-isolation

echo "Installation complete!"
