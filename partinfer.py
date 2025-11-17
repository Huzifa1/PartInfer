from pathlib import Path
import os
import time


def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, prompt, memory_limit, num_fewshot, task_type, num_tokens_to_generate, device, sampling_method, partinfer_method_config, cpu_only = False, top_p = None, function: str = "normal"):
    
    if (method == "partinfer"):
        partinfer_method_config["use_partial_loading"] = True
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{script_dir}/transformers/partinfer_variables/partinfer_improvements.py", "w") as f:
        use_partial_loading = partinfer_method_config["use_partial_loading"]
        f.write(f"USE_PARTINFER_IMPROVEMENTS = {str(use_partial_loading)}")
        f.close()
    time.sleep(0.1)

    
    import create_neurons_mask
    from utils import generate
    from common import load_model, convert_model
    
    if (partinfer_method_config["use_partial_loading"]):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        create_neurons_mask.main(start_num, end_num, partinfer_method_config)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(f"{script_dir}/transformers/partinfer_variables/mask_filepath.txt", "r") as f:
            MASK_FILEPATH = f.readlines()[0]
        print(f"PARTINFER: Use Mask for partial loading, mask file: {MASK_FILEPATH}") 
    
    model, tokenizer, num_layers = load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)
    
    model = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, partinfer_method_config, cpu_only)

    if (function == "normal"):
        generate(method, model, tokenizer, prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p)
    elif (function == "chat"):
        prompt = ""
        while True:
            prompt = input("User: ")
            if (prompt == "exit"):
                break
            generate(method, model, tokenizer, prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p, show_debug=False)
    elif (function == "predefined_prompts"):
        predefined_prompts = [
            "You visit a museum with ancient Greek artifacts. Which ancient Greek philosopher is known for his method of questioning and debate?",
            "You walk through a dense rainforest and hear monkeys chattering in the trees. What is the name of the large tropical forest found near the equator?",
            "You enjoy a cup of hot chocolate on a cold winter day. What ingredient is typically used to make hot chocolate?",
            "You read a novel about a detective solving mysteries. Who is the famous detective created by Sir Arthur Conan Doyle?",
            "You listen to a rock band perform a famous song on stage. Which band is known for the song \"Bohemian Rhapsody?\"",
            "You read about the Industrial Revolution. What were the key technological advancements during the Industrial Revolution?",
            "You read about the exploration of the New World by Christopher Columbus. What were the motivations for European exploration?",
            "You explore a museum with dinosaur skeletons on display. What is the name of the prehistoric period when dinosaurs lived?",
            "You’re at a concert listening to a famous pop artist. Who is known as the \"King of Pop?\"",
            "You take a selfie on top of the Empire State Building. In which U.S. city is the Empire State Building located?",
            "You’re planting a tree in your backyard. What is the process by which trees make their own food?",
            "What is the capital of Germany?",
        ]
        for prompt in predefined_prompts:
            generate(method, model, tokenizer, prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p, show_debug=False)
    else:
        raise ValueError("No function set: set --function")





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3-3b", help='Model Name')
    parser.add_argument('--prompt', type=str, default="What happens to you if you eat watermelon seeds?", help='Input prompt.')
    parser.add_argument('--num_fewshot', type=int, default=6, help='Number of samples.')
    parser.add_argument('--num_tokens_to_generate', type=int, default=256, help='Maximum number of new tokens.')
    parser.add_argument('--task_type', type=str, choices=['QA', 'Summarize', 'translate_de_en'], default='QA', help='Type of task to perform.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("meta-llama/Llama-3.2-3B"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--sparsity', type=float, default=0.4, help='Sentence Sparsity level.')
    parser.add_argument('--start_num', type=int, default=4, help='Start layer.')
    parser.add_argument('--end_num', type=int, default=26, help='End layer.')
    parser.add_argument('--top_p', type=float, default=0.9, help='When set, will use top-p sampling')
    parser.add_argument('--sampling-method', type=str, default="greedy", choices=["greedy", "top-p"], help='Choose sampling method')
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['coreinfer', 'dense', 'partinfer'], default='partinfer', help='Method to use (default: partinfer).')
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')
    
    ## PARTINFER Method arguments
    parser.add_argument('--base_neurons_percent', type=float, default=0.3, help='Loaded Base Neurons Percent')
    parser.add_argument('--base_neurons_type', type=str, choices=['model', 'dataset'], default='dataset', help='Base Neurons Type')
    parser.add_argument('--loaded_neurons_percent', type=float, default=0.7, help='Overall Percent of Loaded Neurons')
    parser.add_argument('--model_neurons_filepath', type=Path, default="neuron_files/llama3-3b/model_neurons.json", help='Path to model neurons file')
    parser.add_argument('--dataset_neurons_filepath', type=Path, default="neuron_files/llama3-3b/qa.json", help='Path to dataset neurons file')
    parser.add_argument('--mask_filepath', type=Path, default="neuron_files/mask.pkl", help='Path to output mask file')
    parser.add_argument('--use_partial_loading', action='store_true', help='Add argument to use partial loading. Automatically activated for method partinfer')

    parser.add_argument('--function', type=str, choices=['normal', 'chat', 'predefined_prompts'], default='normal', help='Function to use')


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
        "mask_filepath": args.mask_filepath,
        "use_partial_loading": args.use_partial_loading
    }

    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity, args.prompt, args.memory_limit, args.num_fewshot, args.task_type, args.num_tokens_to_generate, args.device, args.sampling_method, partinfer_method_config, args.cpu_only, args.top_p, args.function)
