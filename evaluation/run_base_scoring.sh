#!/bin/bash -l

#SBATCH --job-name=score_empathy_quality
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
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/base_scoring_empathy_quality.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs
conda activate llmtrain_env

echo "Running scoring script..."

python getbasemodelscores.py --file_paths /home/vtiyyal1/askdocs/Clinical-chatbot-training/evaluation/base_outputs/outputs_test_data_gemma-7b-it_base.json /home/vtiyyal1/askdocs/Clinical-chatbot-training/evaluation/base_outputs/outputs_test_data_Llama-2-7b-chat-hf_base.json --empathy_model_name vtiyyal1/empathy_model --quality_model_name vtiyyal1/quality_model
