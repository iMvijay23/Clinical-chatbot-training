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
parser.add_argument("--output_dir", type=str, required=True, help="Dir for saving outputs")
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

# Define the output directory and file name for the processed data
output_dir = args.output_dir
model_name = args.model_path.split('/')[-1]  # Extract model name from the model path for cleaner file naming
file_name = f'outputs_test_data_{model_name}_base.json'
output_path = os.path.join(output_dir, file_name)

# Check if the output directory exists, create it if not
if not os.path.exists(output_dir):
    print(f"Output directory {output_dir} does not exist. Creating it...")
    os.makedirs(output_dir)

# Print the output path
print(f"The output will be saved to: {output_path}")

# Run inference and save the results
run_inference_and_save(model, tokenizer, data, output_path)