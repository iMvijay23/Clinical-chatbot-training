import argparse
import json
import torch
import os
os.environ['HF_HOME'] = '/scratch4/mdredze1/vtiyyal1/huggingface_cache/'
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfFolder
import re

HfFolder.save_token(os.getenv('HUGGINGFACE_TOKEN'))

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def score_text(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.squeeze().cpu().tolist()
    return scores

def process_file(file_path, empathy_model_name, quality_model_name, device):
    print(f"Processing {file_path}")
    empathy_scores = []
    quality_scores = []

    with open(file_path, 'r') as f:
        data = json.load(f)

    empathy_model, empathy_tokenizer = load_model(empathy_model_name, device)
    quality_model, quality_tokenizer = load_model(quality_model_name, device)

    # Define a regular expression pattern to match the text following "###Empathetic Response:"
    #pattern = re.compile(r'Empathetic Response:\s*(.*)', re.DOTALL)

    for entry in tqdm(data, desc=f"Scoring in {file_path}"):
        # Splitting the ft_model_response and taking everything after "###Empathetic Response:"
        empathetic_response = entry.get("ft_model_response", "")
        empathy_score = score_text(empathy_model, empathy_tokenizer, empathetic_response, device)
        quality_score = score_text(quality_model, quality_tokenizer, empathetic_response, device)

        entry["empathy_score"] = empathy_score
        entry["quality_score"] = quality_score
        #entry["ft_empathy_response"] = empathetic_response
        empathy_scores.append(empathy_score)
        quality_scores.append(quality_score)

    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    avg_empathy_score = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    print(f"Average Empathy Score: {avg_empathy_score}")
    print(f"Average Quality Score: {avg_quality_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths", nargs="+", type=str, required=True, help="Paths to the JSON files to process")
    parser.add_argument("--empathy_model_name", type=str, required=True, help="Empathy model name on Hugging Face")
    parser.add_argument("--quality_model_name", type=str, required=True, help="Quality model name on Hugging Face")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for file_path in args.file_paths:
        process_file(file_path, args.empathy_model_name, args.quality_model_name, device)
