#!/bin/bash -l

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

# Source conda initialization script from your miniconda installation
. /home/vtiyyal1/miniconda3/etc/profile.d/conda.sh
conda activate trainllm3

echo $LD_LIBRARY_PATH | tr ':' '\n'
echo $CONDA_PREFIX

#NVIDIA SMI
nvidia-smi

python -m bitsandbytes
