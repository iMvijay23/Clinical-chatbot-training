#!/bin/bash -l

# Job settings
JOB_NAME="llama2baseaskdocs"
TIME_LIMIT="48:00:00"
PARTITION="a100"
GPU_COUNT="1"
NODE_COUNT="1"
TASKS_PER_NODE="12"
QOS="qos_gpu"
ACCOUNT="mdredze1_gpu"
MAIL_USER="vtiyyal1@jh.edu"
MAIL_TYPE="BEGIN,END,FAIL"
OUTPUT_PATH="/home/vtiyyal1/askdocs/outputs/llamabase_finetune_askdocs_latest.out"

# Script settings
CONDA_ENV="llmtrain_env"
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
DATA_DIR="/scratch4/mdredze1/vtiyyal1/data/askdocs/test_data/preprocessed_test_data_llama2_mar13.json"
SCRIPT_NAME="getbasemodeloutputs.py"

# SLURM directives
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --nodes=${NODE_COUNT}
#SBATCH --ntasks-per-node=${TASKS_PER_NODE}
#SBATCH --qos=${QOS}
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --mail-type=${MAIL_TYPE}
#SBATCH -A ${ACCOUNT}
#SBATCH --output=${OUTPUT_PATH}
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

echo "Available Conda environments:"
conda info --envs

echo "Activating Conda environment: ${CONDA_ENV}"
conda activate ${CONDA_ENV}

echo "Running python script..."
python ${SCRIPT_NAME} --model_path "${MODEL_PATH}" --data_dir "${DATA_DIR}"
