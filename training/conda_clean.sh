#!/bin/bash -l

#SBATCH --job-name=llama2askdocs
#SBATCH --time=48:00:00
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=vtiyyal1@jh.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH -A mdredze80_gpu
#SBATCH --job-name="finetune_askdocs"
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/finetune_askdocs_latest.out"
#SBATCH --export=ALL

module load anaconda


conda info --envs
echo "conda config --show-sources"
conda config --show
echo "Showing conda environment path..."
echo $CONDA_ENVS_PATH
# Remove the existing Conda environment if it exists
conda env remove --name llmtrain_env -y
# Create a new Conda environment and install PyTorch with GPU support
#conda create --name llmtrain_env python=3.8 -y
#conda activate llmtrain_env

# Install PyTorch with CUDA support (adjust cuda version if needed)
#conda install pytorch==2.0.1 cudatoolkit=11.3 -c pytorch
#conda install -c conda-forge accelerate
# Update PATH to include .local/bin
#export PATH="${CONDA_PREFIX}/bin:$PATH"

#pip install --upgrade -r requirements.txt
# Clean up any unused packages and caches
#conda clean --all -y


python -c "import torch; print(torch.__version__)"


