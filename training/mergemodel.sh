#!/bin/bash

MODEL_NAME='meta-llama/Llama-2-7b-chat-hf'
#ADAPTER_PATH='/data/solr/models/tobaccowatcher/checkpoints_oct16/checkpoint-28499' 
#OUTPUT_DIR='/data/solr/models/tobaccowatcher/checkpoints_oct16/checkpoint-28499'
ADAPTER_PATH='/data/solr/models/askdocsproject/checkpoints_2/checkpoint-31999'
OUTPUT_DIR='/data/solr/models/askdocsproject/checkpoints_2/checkpoint-31999'
python mergemodel.py $MODEL_NAME $ADAPTER_PATH $OUTPUT_DIR
