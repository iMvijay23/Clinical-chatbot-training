import os
import sys
import torch
from peft import PeftModelForCausalLM
import json
from shutil import rmtree
from tqdm import tqdm 
import evaluate 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from huggingface_hub import HfApi, HfFolder

HfFolder.save_token(os.getenv('HUGGINGFACE_TOKEN'))
import numpy as np


# Constants
EVAL_DATASET_PATH = "/content/preprocessed_test_data_mar11.json"
TOKENIZER_NAME = "google/gemma-7b-it"
CHECKPOINT_DIR = '/content/outputs/checkpoint-5000'

def merge_and_save_model(model, output_dir):
    """Merge the adapter into the model and save it."""
    merged_model_path = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_model_path, exist_ok=True)
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_path)
    return merged_model_path

def evaluate_model(model, eval_dataset, tokenizer, output_file_path, device='cuda'):
    generated_responses = []
    model.to(device)
    for entry in tqdm(eval_dataset, desc="Generating responses"):
        input_ids = tokenizer(entry['New Prompt'], return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=128)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_responses.append({"postID": entry['postID'], "Generated Response": response})

    # Save generated responses to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(generated_responses, f, indent=4)

def main():
    # Load the tokenizer and evaluation dataset
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    with open(EVAL_DATASET_PATH, 'r') as f:
        eval_dataset = json.load(f)

    # Determine the path for the merged model
    merged_model_path = os.path.join(CHECKPOINT_DIR, "merged_model")

    # Check if the merged model exists, and merge if necessary
    if not os.path.exists(merged_model_path):
        print(f"Merging model from checkpoint {CHECKPOINT_DIR}")
        base_model = AutoModelForCausalLM.from_pretrained(TOKENIZER_NAME, device_map="auto", torch_dtype=torch.float16)
        model_to_merge = PeftModelForCausalLM.from_pretrained(base_model, CHECKPOINT_DIR)
        merged_model_path = merge_and_save_model(model_to_merge, CHECKPOINT_DIR)
    else:
        print(f"Merged model already exists at {merged_model_path}")

    # Load the merged model
    merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path)

    # Evaluate the model and save generated responses
    responses_output_path = os.path.join(CHECKPOINT_DIR, "generated_responses.json")
    evaluate_model(merged_model, eval_dataset, tokenizer, responses_output_path)

    print("Evaluation completed and responses saved.")

if __name__ == "__main__":
    main()

