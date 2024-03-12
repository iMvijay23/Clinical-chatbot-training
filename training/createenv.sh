#!/bin/bash -l

# Source conda initialization script from your miniconda installation
. /home/vtiyyal1/miniconda3/etc/profile.d/conda.sh

# Check if the Conda environment exists. If not, create it.
env_name="trainllm"
if [[ $(conda info --envs | grep $env_name) == "" ]]; then
    conda create --name $env_name python=3.8 -y
fi

# Activate the environment
conda activate $env_name

# Install cudatoolkit for version 12.1
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit

# Install PyTorch, torchvision, and torchaudio for CUDA 12.1
pip3 install torch torchvision torchaudio

# Install your requirements
pip3 install -r requirements.txt

# Uninstall the original bitsandbytes
pip uninstall bitsandbytes -y

# Clone your fork (adjust the URL if needed, since I'm using the one from the readme)
git clone https://github.com/KaiserWhoLearns/bitsandbytes.git

# Change into the cloned directory
cd bitsandbytes

# Build the CUDA code for version 12.1
CUDA_VERSION=121 make cuda11x

# Install using setup.py
python setup.py install

# Change back to the original directory
cd ..

# Print status
echo "Environment set up and dependencies installed."
