#!/bin/bash

# Activate conda environment
conda env list
conda activate llmtrain

# Install Python dependencies
#pip install -r requirements.txt



#accelerate config
# Running the training script
#accelerate launch --multi_gpu --num_processes 1
python train.py \
--model_name "meta-llama/Llama-2-7b-chat-hf" \
--dataset_name "vtiyyal1/AskDocs-53k" \
--max_seq_len 2048 \
--num_train_epochs 2 \
--max_steps 50000 \
--logging_steps 25 \
--eval_steps 100 \
--save_steps 500 \
--lora_r 8 \
--lora_alpha 32 \
--bf16 True \
--packing True \
--output_dir "/data/solr/models/askdocsproject" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--dataset_text_field "content" \
--use_peft_lora True \
--use_4bit_qunatization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--learning_rate 5e-5  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--optim "paged_adamw_8bit" \
--use_flash_attn True