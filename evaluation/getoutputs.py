import argparse
import json
import torch
import os
os.environ['HF_HOME'] = '/scratch4/mdredze1/vtiyyal1/huggingface_cache/'
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import numpy as np





parser = argparse.ArgumentParser(description="Run inference on the checkpoints")
parser.add_argument("--base_dir", type=str, required=True, help="Base dir for checkpoints")
parser.add_argument("--data_dir", type=str, required=True, help="Base dir for data")
args = parser.parse_args()

EVAL_DATASET_PATH= args.data_dir
#ensure we are on cuda
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f'Using device: {device}')

#BNB config
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


#function to load model from a checkpoint\
def load_model(checkpoint_path):
    peftconfig = PeftConfig.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(peftconfig.base_model_name_or_path, return_dict=True, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(peftconfig.base_model_name_or_path)
    return model, tokenizer


#
#function to get inference from the model
def run_inference_and_save(model, tokenizer, data, output_path):
    processed_data = []
    for item in tqdm(data):
        text = item['New Prompt']
        inputs = tokenizer(text, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        item['ft_model_response'] = response
        processed_data.append(item)
    
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print(f'Processed dataset is saved to {output_path}')




#load testing/evaluation dataset 
with open(EVAL_DATASET_PATH,'r') as f:
    data = json.load(f)



# model directories from the base path 
checkpoints = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir,d))]

#print(checkpoints,'checkpoints')

"""
for model_dir in model_dirs:
    model_path = os.path.join(args.base_dir,model_dir)
    checkpoints = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path,d))]

    for checkpoint_path in checkpoints:
        checkpoint_path = os.path.join(model_path,checkpoint_path)
        print(f'Processing checkpoint path {checkpoint_path}')
        model, tokenizer = load_model(checkpoint)
        output_path = os.path.join(checkpoint_path,'processed_test_data_outputs.json')
        run_inference_and_save(model,tokenizer,data, output_path)
"""


for checkpoint_path in checkpoints:
        checkpoint_path = os.path.join(args.base_dir,checkpoint_path)
        print(f'Processing checkpoint path {checkpoint_path}')
        model, tokenizer = load_model(checkpoint_path)
        output_path = os.path.join(checkpoint_path,'processed_test_data_outputs.json')
        run_inference_and_save(model,tokenizer,data, output_path)


