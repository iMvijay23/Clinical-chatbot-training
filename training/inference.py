import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference on a model.")
    parser.add_argument("--model_path", required=False, help="Path to the model directory.")
    parser.add_argument("--use_quantize", type=int, default=0, help="Use quantization if set to 1.")
    args = parser.parse_args()

    # Load the model
    model_id = '/data/solr/models/tobaccowatcher/checkpoints_oct16/checkpoint-16999' #args.model_path

    if args.use_quantize:
        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id,  torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)

    #model.half()  # Convert model to use 16-bit precision
    model.to('cuda')

    #prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible... If you don’t know the answer to a question, please don’t share false information."""

    # Inference loop
    print("Interactive inference mode activated. Type 'stop' to exit.")
    while True:
        user_input = input("Enter your query (or 'stop' to exit): ")

        if user_input.lower() == 'stop':
            break

        # Combine prompt and user input
        full_query = """
        <|system|> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <|endoftext|> <|prompter|> 
        """ + user_input + """ <|endoftext|> <|assistant|> """

        # Tokenize input and get prediction
        input_tensor = tokenizer.encode(full_query, return_tensors="pt")
        with torch.no_grad():
            prediction = model.generate(input_tensor.to('cuda'), max_new_tokens=256)  # You can adjust max_length if necessary

        # Decode the prediction to text
        predicted_text = tokenizer.decode(prediction[0], skip_special_tokens=True)

        # Print the generated text
        #print(predicted_text)
        # Removing the inputted text from the output
        #start_idx = predicted_text.find(user_input) 
        #if start_idx != -1:
        #    response = predicted_text[start_idx + len(user_input):].strip()  # Getting everything after the user input
        #    print(response)
        #else:
        #    print(predicted_text)  # This is just a fallback in case, for some reason, the user input isn't found in the predicted text. Ideally, this shouldn't happen.

        # Extracting the response between <assistant> and the next <endoftext>
        start_idx = predicted_text.find('<|assistant|>')
        end_idx = predicted_text.find('<|endoftext|>', start_idx)  # start the search from the index of <assistant>

        if start_idx != -1 and end_idx != -1:
            response = predicted_text[start_idx+len('<|assistant|>'):end_idx].strip()
            print(response)
        else:
            print("Unable to extract assistant's response.")
            print("==========")
            print(predicted_text)
    

if __name__ == "__main__":
    main()
