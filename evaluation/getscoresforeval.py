import argparse
import json
import torch
import os
os.environ['HF_HOME'] = '/scratch4/mdredze1/vtiyyal1/huggingface_cache/'
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfFolder

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

def process_directory(base_dir, data_file_name, empathy_model_name, quality_model_name, device):
    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if os.path.isdir(dirpath):
            data_file_path = os.path.join(dirpath, data_file_name)
            if os.path.exists(data_file_path):
                print(f"Processing {data_file_path}")
                empathy_scores = []
                quality_scores = []

                with open(data_file_path, 'r') as f:
                    data = json.load(f)

                empathy_model, empathy_tokenizer = load_model(empathy_model_name, device)
                quality_model, quality_tokenizer = load_model(quality_model_name, device)

                for entry in data:
                    empathetic_response = entry.get("ft_model_response", "").split("###Empathetic Response:")[-1].strip()
                    empathy_score = score_text(empathy_model, empathy_tokenizer, empathetic_response, device)
                    quality_score = score_text(quality_model, quality_tokenizer, empathetic_response, device)

                    entry["empathy_score"] = empathy_score
                    entry["quality_score"] = quality_score
                    empathy_scores.append(empathy_score)
                    quality_scores.append(quality_score)
                    entry["ft_empathy_response"] = empathetic_response

                with open(os.path.join(dirpath, f"scored_{data_file_name}"), 'w') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                avg_empathy_score = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0
                avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                print(f"Average Empathy Score for {dirname}: {avg_empathy_score}")
                print(f"Average Quality Score for {dirname}: {avg_quality_score}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing checkpoint folders")
    parser.add_argument("--data_file", type=str, required=True, help="Name of the data file to process in each checkpoint")
    parser.add_argument("--empathy_model_name", type=str, required=True, help="Empathy model name on Hugging Face")
    parser.add_argument("--quality_model_name", type=str, required=True, help="Quality model name on Hugging Face")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    process_directory(args.base_dir, args.data_file, args.empathy_model_name, args.quality_model_name, device)