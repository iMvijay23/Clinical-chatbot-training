#!/bin/bash -l

#SBATCH --job-name=gemmabase-askdocs
#SBATCH --time=48:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=vtiyyal1@jh.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH -A mdredze1_gpu
#SBATCH --job-name="gemma7bbase_finetune_askdocs"
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/gemmabase_finetune_askdocs_latest.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs

conda activate llmtrain_env


echo "Running python script..."
python getbasemodeloutputs.py --model_path "google/gemma-7b-it" --data_dir "/scratch4/mdredze1/vtiyyal1/data/askdocs/test_data/preprocessed_test_data_gemma_mar13.json" --output_dir "/home/vtiyyal1/askdocs/Clinical-chatbot-training/evaluation/base_outputs"
