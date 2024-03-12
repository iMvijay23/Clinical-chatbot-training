import os
import sys
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM, get_peft_config
import torch

def merge_and_save_model(model, output_dir):
    """Function to merge the adapter into the model and save it."""
    #merged_model_path = os.path.join(output_dir, "merged_model")
    os.makedirs(output_dir, exist_ok=True)
    print(f"merging model")
    model = model.merge_and_unload()
    model.half()
    print(f"merged, saving model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="10GB")

def main():
    model_name = sys.argv[1]
    adapter_path = sys.argv[2]
    output_dir = sys.argv[3]

    # Load the base model
    #model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    #base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")

    # Load the PEFT-trained adapter weights
    #adapter_weights = os.path.join(adapter_path, "adapter_model.bin")
    model_to_merge = PeftModelForCausalLM.from_pretrained(base_model, adapter_path)
    # Merge and save
    merge_and_save_model(model_to_merge, output_dir)

if __name__ == "__main__":
    main()
