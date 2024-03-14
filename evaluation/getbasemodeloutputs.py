import argparse
import json
import torch
import os
os.environ['HF_HOME'] = '/scratch4/mdredze1/vtiyyal1/huggingface_cache/'
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import numpy as np


# Parse command line arguments
parser = argparse.ArgumentParser(description="Run inference using the base model")
parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
parser.add_argument("--data_dir", type=str, required=True, help="Base dir for data")
args = parser.parse_args()

# Set the evaluation dataset path
EVAL_DATASET_PATH = args.data_dir

# Configuration for BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Function to load the base model
def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Function to perform inference and save the results
def run_inference_and_save(model, tokenizer, data, output_path):
    processed_data = []
    for item in tqdm(data):
        text = item['New Prompt']
        inputs = tokenizer(text, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        item['base_model_response'] = response
        processed_data.append(item)
    
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print(f'Processed dataset is saved to {output_path}')

# Load the testing/evaluation dataset
with open(EVAL_DATASET_PATH, 'r') as f:
    data = json.load(f)

# Load the base model
model, tokenizer = load_model(args.model_path)

# Define the output path for the processed data
output_path = os.path.join(args.data_dir, 'processed_test_data_outputs_base.json')

# Run inference and save the results
run_inference_and_save(model, tokenizer, data, output_path)