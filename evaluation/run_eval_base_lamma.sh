#!/bin/bash -l

#SBATCH --job-name=llama2baseaskdocs
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
#SBATCH --job-name="llama2chat_evalbase_askdocs"
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/llamabase_finetune_askdocs_latest.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs

conda activate llmtrain_env


echo "Running python script..."

python getbasemodeloutputs.py --model_path "meta-llama/Llama-2-7b-chat-hf" --data_dir "/scratch4/mdredze1/vtiyyal1/data/askdocs/test_data/preprocessed_test_data_llama2_mar13.json" --output_dir "/home/vtiyyal1/askdocs/Clinical-chatbot-training/evaluation/base_outputs"
