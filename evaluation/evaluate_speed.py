import torch
import time
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from common import *
from transformers.partinfer import USE_PARTINFER_IMPROVEMENTS
import create_neurons_mask

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_batched(model, tokenizer, num_tokens_to_generate, device, batch_size, num_batches):
    # Create synthetic prompts for testing
    prompts = [f"Once upon a time {i}" for i in range(batch_size * num_batches)]
    
    all_tokens_generated = 0
    total_elapsed_time = 0.0
    
    for b in range(num_batches):
        batch_prompts = prompts[b*batch_size:(b+1)*batch_size]
        
        # Tokenize as a batch
        input_ids = tokenizer(batch_prompts, return_tensors='pt', padding=True).input_ids.to(device)
        
        # Prefill step (simulate KV cache warmup)
        with torch.no_grad():
            prefill_output = model.generate(
                input_ids,
                max_length=input_ids.size(1) + 1,
                do_sample=False,
            )
        input_ids_prefilled = prefill_output[:, :-1]

        # Measure decoding speed
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids_prefilled,
                max_length=input_ids_prefilled.size(1) + num_tokens_to_generate,
                do_sample=False,
            )
        end_time = time.time()
        
        num_generated_tokens = output.size(1) - input_ids_prefilled.size(1)
        elapsed_time = end_time - start_time

        all_tokens_generated += num_generated_tokens * batch_size
        total_elapsed_time += elapsed_time

        print(f"Batch {b+1}/{num_batches}: Generated {num_generated_tokens * batch_size} tokens "
              f"in {elapsed_time:.2f}s ({(num_generated_tokens*batch_size)/elapsed_time:.2f} tok/s)")
    
    print("\n=== Overall Statistics ===")
    print(f"Total tokens generated: {all_tokens_generated}")
    print(f"Total time: {total_elapsed_time:.2f} seconds")
    print(f"Average throughput: {all_tokens_generated/total_elapsed_time:.2f} tokens/second")


def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, memory_limit, num_tokens_to_generate, device, partinfer_method_config, cpu_only = False):
    
    if (USE_PARTINFER_IMPROVEMENTS):
        create_neurons_mask.main(start_num, end_num, partinfer_method_config)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        with open(f"{parent_dir}/transformers/partinfer_variables/mask_filepath.txt", "r") as f:
            MASK_FILEPATH = f.readlines()[0]
        print(f"PARTINFER: Use Mask for partial loading, mask file: {MASK_FILEPATH}") 
    
    model, tokenizer, num_layers = load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)
    
    model = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, partinfer_method_config, cpu_only)

    evaluate_batched(
        model, tokenizer,
        num_tokens_to_generate=256,
        device=device,
        batch_size=16,
        num_batches=5
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3-3b", help='Model Name')
    parser.add_argument('--num_tokens_to_generate', type=int, default=256, help='Maximum number of new tokens.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("meta-llama/Llama-3.2-3B"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--sparsity', type=float, default=0.4, help='Sentence Sparsity level.')
    parser.add_argument('--start_num', type=int, default=4, help='Start layer.')
    parser.add_argument('--end_num', type=int, default=26, help='End layer.')
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['coreinfer', 'dense', 'partinfer'], default='partinfer', help='Method to use (default: partinfer).')
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')
    
    ## PARTINFER Method arguments
    parser.add_argument('--base_neurons_percent', type=float, default=0.3, help='Loaded Base Neurons Percent')
    parser.add_argument('--base_neurons_type', type=str, choices=['model', 'dataset'], default='dataset', help='Base Neurons Type')
    parser.add_argument('--loaded_neurons_percent', type=float, default=0.7, help='Overall Percent of Loaded Neurons')
    parser.add_argument('--model_neurons_filepath', type=Path, default="neurons_files/llama3-3b/model_neurons.json", help='Path to model neurons file')
    parser.add_argument('--dataset_neurons_filepath', type=Path, default="neurons_files/llama3-3b/qa.json", help='Path to dataset neurons file')
    parser.add_argument('--mask_filepath', type=Path, default="neurons_files/mask.pkl", help='Path to output mask file')



    args = parser.parse_args()
    if (args.cpu_only):
        default_device = 'cpu'
        args.device = 'cpu'
    if args.cpu_only and args.memory_limit:
        parser.error("The options --cpu_only and --memory_limit cannot be used together.")
  
    partinfer_method_config = {
        "base_neurons_percent": args.base_neurons_percent,
        "base_neurons_type": args.base_neurons_type,
        "loaded_neurons_percent": args.loaded_neurons_percent,
        "model_neurons_filepath": args.model_neurons_filepath,
        "dataset_neurons_filepath": args.dataset_neurons_filepath,
        "mask_filepath": args.mask_filepath
    }

    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity, args.memory_limit, args.num_tokens_to_generate, args.device, partinfer_method_config, args.cpu_only)
