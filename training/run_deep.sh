#!/bin/bash -l

#SBATCH --job-name=llama2askdocs
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
#SBATCH --job-name="llama2chat finetune_askdocs"
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/llama_finetune_askdocs_latest.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs

conda activate llmtrain_env


echo "Running python script..."


# Running the training script 
#  
accelerate launch --config_file "configs/deepspeed_config.yaml" train.py \
--model_name "meta-llama/Llama-2-7b-chat-hf" \
--dataset_name "vtiyyal1/AskDocsEmpathy_llama2_it" \
--max_seq_len 2048 \
--num_train_epochs 3 \
--max_steps 20000 \
--logging_steps 25 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "/scratch4/mdredze1/vtiyyal1/models/askdocsproject/checkpoints_apr3/llama2chat/" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--dataset_text_field "content" \
--use_gradient_checkpointing \
--use_peft_lora True \
--use_8bit_qunatization False \
--learning_rate 1e-5  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--optim "paged_adamw_8bit" \
--use_flash_attn True