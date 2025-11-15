import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common
from transformers.partinfer_variables.partinfer_improvements import USE_PARTINFER_IMPROVEMENTS

indices_list_all = []
previous_indices_list_all = []

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, name, original_neurons_num, partinfer_method_config):
        super(CustomMLPLayer, self).__init__()
        
        self.device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")
        
        neuron_num = round(original_neurons_num * sparsity)
        if "down" in name:
            loaded_neuron_num = weight.size(1)
        else:
            loaded_neuron_num = weight.size(0)

        if neuron_num > loaded_neuron_num:
            raise RuntimeError(f"Number of required neurons ({neuron_num}) is larger than the number of loaded neurons ({loaded_neuron_num})")
        
        if p := partinfer_method_config["base_neurons_percent"] > sparsity:
            raise RuntimeError(f"base_neurons_percent ({p}) is larger than sparsity ({sparsity}).")

        self.weight = weight.to(self.device)
        self.num = num
        self.name = name
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.end_num = end_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit
        self.original_neurons_num = original_neurons_num        
        self.loaded_neuron_num = loaded_neuron_num
        self.base_neurons_percent = partinfer_method_config["base_neurons_percent"]
        self.is_reorderd = False

    def forward(self, x):
        global indices_list_all, previous_indices_list_all
        if x.size(1)>1:
            self.weight_updated = False
            
            if self.is_reorderd:
                indices = previous_indices_list_all[self.num - (self.start_num + 1)]
                common.reorder_tensor(self.weight, indices, is_reverse="down" not in self.name, is_restore=True)
            
            true_value = x @ self.weight.T

            if "down" in self.name:
                squeezed_x = x.squeeze()

                # Get base neurons
                # This works since when loading, base neurons are sorted at the beginning
                base_neuron_num = int(self.base_neurons_percent * self.original_neurons_num)
                base_neurons = torch.arange(0, base_neuron_num)
                
                if self.base_neurons_percent < self.sparsity:
                    # Now fill up with core neurons
                    core_neurons = common.get_core_neurons(squeezed_x, self.token_sparsity, 1, self.weight.size(1))
                
                    # Now remove the overlap
                    mask = ~torch.isin(core_neurons, base_neurons)
                    unique_core_neurons = core_neurons[mask]
                    
                    # Now get the rest of neurons to load from unique_core_neurons
                    unique_core_neurons_to_compute = unique_core_neurons[:int((self.sparsity - self.base_neurons_percent) * self.original_neurons_num)]      
                    indices_all = torch.cat((base_neurons, unique_core_neurons_to_compute))
                else:
                    indices_all = base_neurons
              

                common.reorder_tensor(self.weight, indices_all)
                self.is_reorderd = True
                end_index = indices_all.shape[0]
                if (self.device == torch.device("cuda")):
                    self.filtered_W = self.weight[:, 0:end_index].contiguous()
                else:
                    self.filtered_W = self.weight[:, 0:end_index]
                    
                
                if self.num == (self.start_num + 1):
                    indices_list_all=[]
                    
                indices_list_all.append(indices_all)
                
                if self.num == self.end_num - 1:
                    previous_indices_list_all = indices_list_all.copy()

        else:
            if "down" not in self.name:
                if not self.weight_updated:
                    indices = indices_list_all[self.num - (self.start_num + 1)]
                    common.reorder_tensor(self.weight, indices, is_reverse=True)
                    self.is_reorderd = True
                    end_index = indices.shape[0]
                    if (self.device == torch.device("cuda")):
                        self.filtered_W = self.weight[0:end_index, :].contiguous()
                    else:
                        self.filtered_W = self.weight[0:end_index, :]
                    self.weight_updated = True
                    
            
            true_value = x @ self.filtered_W.T
            
        return true_value


def convert_llama_qwen_model_partinfer(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, original_neurons_num, partinfer_method_config):
    
    if not USE_PARTINFER_IMPROVEMENTS:
        raise RuntimeError("Partinfer Improvements / partial loading needs to be activated for partinfer Method")
    
    for name, module in tqdm(model.named_modules(), desc="Converting Modules"):
        if "down" in name or "up" in name or "gate" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
                
                NewLayer = CustomMLPLayer(module.weight, num, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, name, original_neurons_num, partinfer_method_config)
                setattr(parent, attr_name, NewLayer)
                del module

    gc.collect()    
    
    return model