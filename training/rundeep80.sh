#!/bin/bash -l
#SBATCH --job-name=llama2askdocs
#SBATCH --time=48:00:00
#SBATCH --partition ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH -A mdredze80_gpu
#SBATCH --job-name="finetune_askdocs"
#SBATCH --output="/home/vtiyyal1/askdocs/output/finetune_askdocs_latest_ica.out"
#SBATCH --export=ALL

module load cuda/12.1.0
echo "Printing Conda env info..."
module load anaconda
conda info --envs
conda activate trainllm

# init virtual environment if needed for ica100 its 64 ntasks-per-node if 1 node then 12 tasks per node
# conda create -n toy_classification_env python=3.7


#pip install -r requirements.txt

#added to avoid cuda error 


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