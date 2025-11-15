from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from convert.convert_opt_model import convert_opt_model
from convert.convert_llama_qwen_model import convert_llama_qwen_model
from convert.convert_llama_qwen_model_partinfer import convert_llama_qwen_model_partinfer
from convert.convert_opt_model_partinfer import convert_opt_model_partinfer

from utils import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import re
import os
import pickle
from tqdm import tqdm
import time

MODEL_INFO = {
    'opt-1.3b': {
        'num_neurons': 8192,
        'activation_fn': torch.nn.ReLU,
    },
    'opt-6.7b': {
        'num_neurons': 16384,
        'activation_fn': torch.nn.ReLU,
    },
    'llama3-8b': {
        'num_neurons': 14336,
        'activation_fn': torch.nn.SiLU
    },
    'llama3-3b': {
        'num_neurons': 8192,
        'activation_fn': torch.nn.SiLU
    },
    'llama3-1b': {
        'num_neurons': 8192,
        'activation_fn': torch.nn.SiLU
    },
    'qwen2.5-3b': {
        'num_neurons': 11008,
        'activation_fn': torch.nn.SiLU
    },
}
    
def get_layer_name(model_name, Layer_num):
    if "opt" in model_name:
        return f"model.decoder.layers.{Layer_num}.activation_fn"
    
    if "llama" or "qwen2.5" in model_name:
        return f"model.layers.{Layer_num}.mlp.down_proj"
    
    raise ValueError("Model Name not supported")

def get_layer_number(model_name, layer_name):
    if "opt" in model_name:
        return int(layer_name.split(".")[3])
    
    if "llama" or "qwen2.5" in model_name:
        return int(layer_name.split(".")[2])
    
    raise ValueError("Model Name not supported")


def load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit):
    start_time = time.time()
    
    if (device == "cuda" and torch.cuda.is_available()):
        device = 0

    if memory_limit == True:
        model, num_layers = load_model_memory_limit(checkpoint_path, start_num, end_num, model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map=device, torch_dtype=torch.float16)
        num_layers = model.config.num_hidden_layers

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, device_map=device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done. Loaded model in {elapsed_time:.2f} seconds.\n")
    return model, tokenizer, num_layers

def convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, partinfer_method_config, cpu_only = False):
    start_time = time.time()

    if "opt" in model_name:
        if method == 'coreinfer':
            model = convert_opt_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
        elif method == 'partinfer':
            model = convert_opt_model_partinfer(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, MODEL_INFO[model_name]["num_neurons"], partinfer_method_config)
        elif method == 'dense':
            pass

    elif "llama" or "qwen2.5" in model_name.lower():
        if method == 'coreinfer':
            model = convert_llama_qwen_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
        elif method == 'partinfer':
            model = convert_llama_qwen_model_partinfer(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, MODEL_INFO[model_name]["num_neurons"], partinfer_method_config)
        elif method == 'dense':
            pass
     
    print(f"Converting model {model_name} using method {method}...")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done. Converted model in {elapsed_time:.2f} seconds.\n")
    return model



def get_core_neurons(x, token_sparsity, sparsity, neuron_num = None):
    sorted_values, sorted_indices = torch.sort(x, dim=1, descending=True)
    limit=int(token_sparsity * (x > 0).sum().item() / x.size(0))
    top_indices = sorted_indices[:, :limit]
    data_flattened = top_indices.reshape(-1)
    unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_indices_clu = unique_numbers[sorted_indices]

    if (neuron_num is not None):
        remained_neurons = int(neuron_num * sparsity)
    else:
        # Then calculate them dynamically based on the size of `sorted_indices_clu`
        remained_neurons = int(len(sorted_indices_clu) * sparsity)

    indices_all = sorted_indices_clu[:remained_neurons].cpu()
    return indices_all

def get_model_neurons(sparsity, model_neurons, neuron_num=None, layer_num=None):
    remained_neurons = int(neuron_num * sparsity)
    indices_all = model_neurons[layer_num, :remained_neurons]
    indices_all = indices_all[indices_all != -1]
    return indices_all

def get_hybrid_neurons(x, token_sparsity, sparsity, hybrid_split, neuron_num=None):
    remained_neurons = int(neuron_num * sparsity)
    core_neuron_num = int(remained_neurons * hybrid_split)

    if core_neuron_num > 0:
        sorted_values, sorted_indices = torch.sort(x, dim=1, descending=True)
        limit=int(token_sparsity * (x > 0).sum().item() / x.size(0))
        top_indices = sorted_indices[:, :limit]
        data_flattened = top_indices.reshape(-1)
        unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
        sorted_indices = torch.argsort(counts, descending=True)
        sorted_indices_clu = unique_numbers[sorted_indices]
        indices_all = sorted_indices_clu[:core_neuron_num].cpu()
    else:
        indices_all = torch.tensor([], dtype=torch.long)
    
    return indices_all

def get_core_neurons_improved(x, token_sparsity, sparsity, neuron_num):
    remained_neurons = int(neuron_num * sparsity)
    
    tokens_core_neurons = []
    for token_activation in x:
        _, sorted_indices = torch.sort(token_activation, descending=True)
        limit = int(token_sparsity * (token_activation > 0).sum().item())
        core_neurons = sorted_indices[:limit]
        tokens_core_neurons.append(core_neurons)
        
    data_flattened = [item for sublist in tokens_core_neurons for item in sublist]
    data_flattened = torch.tensor(data_flattened)
    unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)

    
    
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_indices_clu = unique_numbers[sorted_indices]
    indices_all = sorted_indices_clu[:remained_neurons].cpu()

    return indices_all


def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = input[0].detach().cpu()
    return hook



def register_act_hooks(model_name, model, activation_dict, Layer_num = None):
    hooks = []

    model_instance = MODEL_INFO[model_name]['activation_fn']
    
    for name, layer in model.named_modules():
        if (("llama" in model_name or "qwen2.5" in model_name) and "down" in name) or ("opt" in model_name and isinstance(layer, model_instance)):
            if (Layer_num is None or Layer_num == get_layer_number(model_name, name)):
                hooks.append(layer.register_forward_hook(get_activation(name, activation_dict)))
            
    return hooks



def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()



def Elbow(encoded_data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit_predict(encoded_data)
        wcss.append(kmeans.inertia_)
    
    differences = np.diff(wcss)
    second_differences = np.diff(differences)

    elbow_k = np.argmax(second_differences) + 2
    if elbow_k < 4:
        print(f"Original K is {elbow_k}. It will be set to 4")
        elbow_k = 4
    print(f"Optimal number of clusters by Automated Elbow Method: {elbow_k}")
    
    return elbow_k



def collect_activations(model_name, data, tokenizer, device, model, Layer_num = None):
    activations = []
    for i in tqdm(range(len(data)), desc="Collecting activations"):

        prompt = data[i]
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', prompt)
        tokenized_input = tokenizer(cleaned_text, return_tensors="pt").to(device)
        
        activation_dict = {}
        hooks = register_act_hooks(model_name, model, activation_dict, Layer_num)
    
        with torch.no_grad():
            outputs = model(**tokenized_input)
    
        activations.append(activation_dict)
    
        remove_hooks(hooks)
        
        del outputs
        del tokenized_input
        del activation_dict
        
    return activations


def get_sentence_core_neurons(model_name, Layer_num, activations, token_sparsity, sparsity, neuron_num = None):
    
    SEN_F = []
    str_act = get_layer_name(model_name, Layer_num)
    for i in tqdm(range(len(activations)), desc="Calculating Core Neurons"):
        tensor = activations[i][str_act].cpu()

        if "llama" or "qwen2.5" in model_name:
            tensor = tensor.squeeze(0)

        core_neurons = get_core_neurons(tensor, token_sparsity, sparsity, neuron_num)
        SEN_F.append(core_neurons.numpy())

    
    return SEN_F


def concat_activations_per_layer(model_name, activations, layer_num):
    layer_act=[]
    for i in range(len(activations)):
        str_act = get_layer_name(model_name, layer_num)
        tensor = activations[i][str_act].cpu()
        if "llama" or "qwen2.5" in model_name:
            tensor = tensor.squeeze(0)
        layer_act.append(tensor)
    A_tensor = torch.cat(layer_act, dim=0)
    
    return A_tensor


def get_items_core_neurons(model_name, num_layers, activations, token_sparsity, sparsity):
    num_neurons = MODEL_INFO[model_name]["num_neurons"]
    items_core_neurons = []
    for layer_num in range(num_layers):
        A_tensor = concat_activations_per_layer(model_name, activations, layer_num)
        core_neurons = get_core_neurons(A_tensor, token_sparsity, sparsity, num_neurons)
        items_core_neurons.append(core_neurons)
        
    return items_core_neurons


def get_dataset_core_neurons(model_name, checkpoint_path, device, token_sparsity, sparsity, dataset_path, dataset_name, memory_limit = False):
    
    model, tokenizer, num_layers = load_model(model_name, 5, 27, checkpoint_path, device, memory_limit)
    dataset = load_from_disk(dataset_path)
    precessed_data = process_data(dataset, dataset_name)
    activations = collect_activations(model_name, precessed_data, tokenizer, device, model)
    
    dataset_core_neurons = get_items_core_neurons(model_name, num_layers, activations, token_sparsity, sparsity)
    
    return dataset_core_neurons


def get_neurons_scores(x, sum_weight, count_weight, top_k):
    # Set negative values to 0
    x_relu = x * (x > 0)
    # Sum activation values of all tokens for each neuron
    x_summed = x_relu.to(torch.float32).sum(dim=0)

    # Get top_k_th highest activation sum
    topk_values, _ = torch.topk(x_summed, top_k)
    highest_sum_score = topk_values[-1]
    # Normalize over the highest score
    activations_sum_scores = x_summed / highest_sum_score

    # Get the count of how many times did each neuron got activated (value > 0)
    x_activation_count = (x > 0).sum(dim=0)
    # Get highest activation sum
    highest_count_score = torch.max(x_activation_count)
    # Normalize over the highest score
    activations_count_scores = x_activation_count / highest_count_score
    # Now get total score for each neuron
    total_scores = activations_sum_scores * sum_weight + activations_count_scores * count_weight
    
    return total_scores

def get_sparsity_levels(model_name, num_layers, sum_weight, count_weight, threshold, top_k, activations):
    num_neurons = MODEL_INFO[model_name]["num_neurons"]

    sparsity_levels = []
    for activation in activations:
        current_levels = []
        for layer_num in range(num_layers):
            str_act = get_layer_name(model_name, layer_num)
            tensor = activation[str_act]
            if "llama" or "qwen2.5" in model_name:
                tensor = tensor.squeeze(0)
            total_scores = get_neurons_scores(tensor, sum_weight, count_weight, top_k)
            number_of_hot_neurons = int((total_scores > threshold).sum())
            sparsity = round(number_of_hot_neurons / num_neurons, 2)
            current_levels.append(sparsity)
        sparsity_levels.append(current_levels)
    
    avg_sparsity_per_layer = [round(sum(layer_values) / len(sparsity_levels), 2) for layer_values in zip(*sparsity_levels)]

    return avg_sparsity_per_layer


def read_cluster_files(cluster_path, num_layers):
    with open(f'{cluster_path}/cluster_activation/kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    with open(f'{cluster_path}/cluster_activation/mlb_model.pkl', 'rb') as file:
        mlb_loaded = pickle.load(file)
        
    
    cluster_num = 0
    for root, dirs, files in os.walk(f'{cluster_path}/neuron_activation/0'):
        cluster_num += len(files)
        
    Predictor = []
    for i in range(num_layers):
        predict_layer = []
        for cluster_catagory in range(0, cluster_num):
            with open(f'{cluster_path}/neuron_activation/{i}/cluster_{cluster_catagory}.pkl', 'rb') as f:
                    model_predictor = pickle.load(f)
            predict_layer.append(model_predictor)
        Predictor.append(predict_layer)

    return kmeans, mlb_loaded, Predictor

def reorder_tensor(A: torch.Tensor, B: torch.Tensor, is_reverse=False, is_restore=False):
    if (A.device != B.device):
        B = B.to(device=A.device)

    M, N = A.shape
    if is_reverse:
        N, M = A.shape
    
    all_indices = torch.arange(N, device=A.device)
    mask = torch.ones(N, dtype=torch.bool, device=A.device)
    mask[B] = False
    other_indices = all_indices[mask].to(device=A.device)
    
    new_order = torch.cat([B, other_indices], dim=0)
    
    inverse_order = []
    if is_restore:
        inverse_order = torch.empty_like(new_order)
        inverse_order[new_order] = torch.arange(N, device=A.device)
        new_order = inverse_order
    
    if is_reverse:
        A.copy_(A[new_order, :])
    else:
        A.copy_(A[:, new_order])
    
    del all_indices, mask, other_indices, new_order, inverse_order
    if A.is_cuda:
        torch.cuda.empty_cache()