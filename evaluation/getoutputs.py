import argparse
import json
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run inference on the checkpoints")
parser.add_argument("--base_dir", type=str, required=True, help="Base dir for checkpoints")
args = parser.parse_args()

#BNB config
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="4NF",
    bnb_4bit_compute_dtype=torch.bfloat16
)


#function to load model from a checkpoint\