import pickle
import os
from transformers.partinfer_variables.partinfer_improvements import USE_PARTINFER_IMPROVEMENTS

def get_used_neurons_count(layer_id: int) -> int:
    return len(get_used_neurons_with_layer_id(layer_id))

def get_used_neurons(layer_name: str, is_loading: bool = False) -> list[int]:
    layer_id = int(layer_name.split(".")[2])
    is_loading_and_up = is_loading and "up" in layer_name
    return get_used_neurons_with_layer_id(layer_id, is_loading_and_up)
    
def get_used_neurons_with_layer_id(layer_id: int, is_loading_and_up: bool = False) -> list[int]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{script_dir}/partinfer_variables/mask_filepath.txt", "r") as f:
        MASK_FILEPATH = f.readlines()[0]
    
    if not os.path.exists(MASK_FILEPATH):
        raise FileNotFoundError(f"Mask file not found at {MASK_FILEPATH}")
    
    with open(MASK_FILEPATH, "rb") as f:
        neurons_to_load = pickle.load(f)
    
    return neurons_to_load[layer_id]
        

def get_neurons_of_layer_from_file(layer_id: int, filepath: str) -> list[int]:
    with open(filepath, "r") as neurons_file:
        lines = neurons_file.readlines()
    layer_line = lines[layer_id].replace("[", "").replace("]", "")
    neurons_of_layer = [int(element) for element in layer_line.split(",")]
    return neurons_of_layer