#!/bin/bash -l

pip install --upgrade -r requirements.txt


python -c "import torch; print(torch.__version__)"

echo "Running python script..."

# Running the training script 
#  
python train.py \
--model_name "google/gemma-7b-it" \
--dataset_name "vtiyyal1/AskDocsEmpathy_4k_it" \
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