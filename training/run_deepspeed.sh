#!/bin/bash

#SBATCH -A mdredze1_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=10GB
#SBATCH --job-name="finetune_askdocs"
#SBATCH --output="/home/vtiyyal1/tobaccowatcher/llama2/scripts/finetune_tobacco.out"
#SBATCH --export=ALL

echo "Printing Conda env info..."
module load anaconda
conda info --envs
conda activate faiss_2

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

#pip install -r requirements.txt # install Python dependencies
pip install -r requirements.txt
pip install nvidia-ml-py3

# Set your Weights & Biases API key
echo "Setting W&B API key..."
export WANDB_API_KEY='777501c1a468cab3359a9d2ee89293c06605a76e'
export HUGGINGFACE_TOKEN='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'
accelerate config
accelerate env
# runs your code
echo "Running python script..."



# Running the training script
accelerate launch --config_file "configs/deepspeed_config.yaml"  train.py \
--model_name "meta-llama/Llama-2-7b-chat-hf" \
--dataset_name "vtiyyal1/AskDocs-53k" \
--max_seq_len 2048 \
--num_train_epochs 2 \
--max_steps 50000 \
--logging_steps 25 \
--eval_steps 100 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "/scratch4/mdredze1/vtiyyal1/models/askdocsproject" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--dataset_text_field "content" \
--use_gradient_checkpointing \
--use_peft_lora True \
--use_8bit_qunatization False \
--learning_rate 5e-5  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--optim "paged_adamw_8bit" \
--use_flash_attn True