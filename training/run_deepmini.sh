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



# Initialize Conda from Miniconda directly
source /home/vtiyyal1/miniconda3/etc/profile.d/conda.sh

# Remove the existing Conda environment if it exists
/home/vtiyyal1/miniconda3/bin/conda env remove --name trainllm -y

# Create a new Conda environment and install PyTorch with GPU support
/home/vtiyyal1/miniconda3/bin/conda create --name llmtrain_env python=3.8 -y

# Activate the environment
source /home/vtiyyal1/miniconda3/bin/activate llmtrain_env

# Install PyTorch with CUDA support
/home/vtiyyal1/miniconda3/bin/conda install pytorch==2.0.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install other requirements
pip install --upgrade -r requirements.txt

# Confirmation of PyTorch installation
python -c "import torch; print(torch.__version__)"
# Set your Weights & Biases API key
echo "Setting W&B API key..."
export WANDB_API_KEY='777501c1a468cab3359a9d2ee89293c06605a76e'
export HUGGINGFACE_TOKEN='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'



echo "Running python script..."

which python
python -c "import torch; print(torch.__file__)"


# Running the training script 
#  
accelerate launch --config_file "configs/deepspeed_config.yaml" train.py \
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