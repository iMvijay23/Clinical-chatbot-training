import os
import sys
import torch
from peft import PeftModelForCausalLM
import json
from shutil import rmtree
from tqdm import tqdm 
import evaluate 
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from huggingface_hub import HfApi, HfFolder

HfFolder.save_token(os.getenv('HUGGINGFACE_TOKEN'))
import numpy as np


rouge = ROUGEScore()
bleu =  evaluate.load("bleu")
f1 =  evaluate.load("f1")

# Constants
EVAL_DATASET_PATH = "/home/vtiyyal1/datavj/tobacco/finaltobaccoinference2row.json"
TOKENIZER_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Modify this if necessary
CHECKPOINTS = '/data/solr/models/tobaccowatcher/checkpoints_oct16/'

def merge_and_save_model(model, output_dir):
    """Function to merge the adapter into the model and save it."""
    merged_model_path = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_model_path, exist_ok=True)
    print(f"merging model")
    model = model.merge_and_unload()
    model.half()
    print(f"merged, saving model to {merged_model_path}")
    model.save_pretrained(merged_model_path, safe_serialization=True, max_shard_size="10GB")


def evaluate_model(model, eval_dataset, tokenizer, adapter_path, output_file_path):
    # Placeholder for results
    rouge = ROUGEScore()
    generated_responses = {}
    eval_progress = tqdm(eval_dataset, desc=f"Evaluating {os.path.basename(adapter_path)}", leave=False)
    for entry in eval_progress:  # Assuming eval_dataset is a list of queries
        # Combine prompt and user input
        response = ""
        full_query = """
        <|system|> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <|endoftext|> <|prompter|> 
        """ + entry['Question'] + """ <|endoftext|> <|assistant|> """

        # Tokenize input and get prediction
        input_tensor = tokenizer.encode(full_query, return_tensors="pt")
        with torch.no_grad():
            prediction = model.generate(input_tensor.to('cuda'), max_new_tokens=128)  # You can adjust max_length if necessary

            # Decode the prediction to text
            predicted_text = tokenizer.decode(prediction[0], skip_special_tokens=True)
            #print(predicted_text)
        start_idx = predicted_text.find('<|assistant|>')
        end_idx = predicted_text.find('<|endoftext|>', start_idx)  # start the search from the index of <assistant>

        if start_idx != -1 and end_idx != -1:
            response = predicted_text[start_idx+len('<|assistant|>'):end_idx].strip()
            #print(response)
        else:  # If "end_of_text" is not found, take everything till the end
            response = predicted_text[start_idx+len('<|assistant|>'):].strip()
        postID = entry['PostID']
        #generated_responses[postID] = response
        # Append response and metrics to the output file
        with open(output_file_path, 'a') as output_file:
            json.dump({postID: response}, output_file)
            output_file.write("\n")
    eval_progress.close()
    

    # Save generated responses to a JSON file
    #responses_output_path = os.path.join(adapter_path, "intermediate_responses.json")
    #with open(responses_output_path, 'w') as json_file:
    #    json.dump(generated_responses, json_file)
    
    # Compute metrics
    #true_responses = [entry['ChatGPT Response'] for entry in eval_dataset]
    

    #for pred, target in zip(generated_responses, true_responses):
    #    rouge.update(pred, target)
    #rouge_scores = rouge.compute()

    
    #rouge_scores = rouge.compute(generated_responses, true_responses)
    bleu_scores = 1#bleu.compute(predictions=generated_responses, references=true_responses)
    #meteor_scores = meteor.compute(predictions=generated_responses, references=true_responses)
    #wer_score = wer.compute(predictions=generated_responses, references=true_responses)
    
    # Using sklearn's f1_score
    #binary_true = [" ".join(set(resp.split())) for resp in true_responses]
    #binary_generated = [" ".join(set(resp.split())) for resp in generated_responses]
    #f1_score_value = f1(binary_true, binary_generated, average='micro')
    
    metrics = {
        #"f1": f1_score_value,
        #"rouge-l": rouge_scores['rouge-l']['f'],
        #"rouge-1": rouge_scores['rouge-1']['f'],
        #"rouge-2": rouge_scores['rouge-2']['f'],
        "bleu": bleu_scores#['bleu'],
    }

    return generated_responses, metrics

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf" 
    output_dir = "/home/vtiyyal1/daatavj/tobacco/eval_results/"

    # Load the base model and tokenizer and data only once
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    # Load eval dataset
    with open(EVAL_DATASET_PATH, 'r') as f:
        eval_dataset = json.load(f)  # Assuming each line is a query for evaluation
    # Load the PEFT-trained adapter weights

    """

     # List all directories in the CHECKPOINTS folder
    checkpoints_list = [os.path.join(CHECKPOINTS, dir_name) for dir_name in os.listdir(CHECKPOINTS) if os.path.isdir(os.path.join(CHECKPOINTS, dir_name))]
    checkpoints_progress = tqdm(checkpoints_list, desc="Processing Checkpoints")

    for adapter_path in checkpoints_progress:
        merged_model_path = os.path.join(adapter_path, "merged_model")
    
        # Check if the merged model already exists
        if not os.path.exists(merged_model_path) or len(os.listdir(merged_model_path)) == 0:
            model_to_merge = PeftModelForCausalLM.from_pretrained(base_model, adapter_path)
            print(f"merging model from checkpoint {adapter_path}")
            merge_and_save_model(model_to_merge, adapter_path)
            print(f"Loading merged model from {merged_model_path}")
        else:
            print(f"Merged model already exists for checkpoint {adapter_path}. Loading it.")


        # Load merged model
        print(f"Loading merged model from {adapter_path}")
        # Load the merged model
        #double_quant_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_use_double_quant=True,
        #)
        #merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, quantization_config=double_quant_config)
        #merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16)
        merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map = "auto", load_in_4bit=True)

        #merged_model.to('cuda')

        # Evaluate model
        generated_responses, metrics = evaluate_model(merged_model, eval_dataset, tokenizer, adapter_path)
        
        """


    results_file = os.path.join(output_dir, "all_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Adapter paths to process         /data/solr/models/askdocsproject/checkpoints_4pm_oct3/checkpoint-14999','checkpoint-19499',
    adapter_paths = [
        'checkpoint-10499'
    ]
    pathtocat = '/data/solr/models/tobaccowatcher/checkpoints_oct16/checkpoint-10499/incremental_responses_latest.json'

    for adapter_path in tqdm(adapter_paths):
        full_adapter_path = os.path.join(CHECKPOINTS, adapter_path)  # constructing the full path
        merged_model_path = os.path.join(full_adapter_path, "merged_model")
        # Check if the merged model already exists
        if not os.path.exists(merged_model_path) or len(os.listdir(merged_model_path)) == 0:
            model_to_merge = PeftModelForCausalLM.from_pretrained(base_model, full_adapter_path)
            print(f"merging model from checkpoint {full_adapter_path}")
            merge_and_save_model(model_to_merge, full_adapter_path)
        else:
            print(f"Merged model already exists for checkpoint {full_adapter_path}. Loading it.")

        # Load merged model
        merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto", torch_dtype=torch.float16)

        # Evaluate model
        responses_output_path = os.path.join(full_adapter_path, "incremental_responses_latest.json")
        generated_responses, metrics = evaluate_model(merged_model, eval_dataset, tokenizer, full_adapter_path, responses_output_path)

        # ... [Rest of the loop remains the same]
        #generated_responses, metrics = evaluate_model(merged_model, eval_dataset, tokenizer, full_adapter_path)

        # Append to common results file
        all_results[full_adapter_path] = metrics

        # Save individual generated responses for this checkpoint
        responses_path = os.path.join(full_adapter_path, "generated_responses.json")
        
        with open(responses_path, 'w') as f:
            json.dump(generated_responses, f)

        # Cleanup (remove merged model if necessary)
        if os.path.exists(merged_model_path):
            rmtree(merged_model_path)

    # Save the aggregated results
    with open(results_file, 'w') as f:
        json.dump(all_results, f)

    print("Evaluation on all checkpoints is Done!")
            
        
    
    

    

    



if __name__ == "__main__":
    main()
