import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common
from transformers.partinfer_variables.partinfer_improvements import USE_PARTINFER_IMPROVEMENTS

indices_list_all = []
previous_indices_list_all = []

# Convert Opt Models
class ReduceLayer(nn.Module):
    def __init__(self, weight, bias, sparsity, token_sparsity, start_num, end_num, memory_limit, cpu_only, num, original_neurons_num, partinfer_method_config):
        super(ReduceLayer, self).__init__()
        
        self.device = torch.device("cpu") if cpu_only else torch.device("cuda")
        
        neuron_num = round(original_neurons_num * sparsity)
        loaded_neuron_num = weight.size(0)
        
        if neuron_num > loaded_neuron_num:
            raise RuntimeError(f"Number of required neurons ({neuron_num}) is larger than the number of loaded neurons ({loaded_neuron_num})")
        
        if p := partinfer_method_config["base_neurons_percent"] > sparsity:
            raise RuntimeError(f"base_neurons_percent ({p}) is larger than sparsity ({sparsity}).")
                
        self.bias = bias.to(self.device)
        self.weight = weight.to(self.device)
        self.memory_limit = memory_limit
        self.token_sparsity = token_sparsity
        self.cpu_only = cpu_only
        self.sparsity = sparsity
        self.num = num
        self.start_num = start_num
        self.end_num = end_num
        self.original_neurons_num = original_neurons_num        
        self.loaded_neuron_num = loaded_neuron_num
        self.base_neurons_percent = partinfer_method_config["base_neurons_percent"]
        self.is_reorderd = False

    def forward(self, x):
        global indices_list_all, previous_indices_list_all
        
        if x.size(0)>1:
            self.weight_updated = False
            
            if self.is_reorderd:
                indices = previous_indices_list_all[self.num - (self.start_num + 1)]
                common.reorder_tensor(self.weight, indices, is_reverse=True, is_restore=True)
            
            true_value = x @ self.weight.T + self.bias
            
            # Get base neurons
            # This works since when loading, base neurons are sorted at the beginning
            base_neuron_num = int(self.base_neurons_percent * self.original_neurons_num)
            base_neurons = torch.arange(0, base_neuron_num)
            
            if self.base_neurons_percent < self.sparsity:
                # Now fill up with core neurons
                core_neurons = common.get_core_neurons(x, self.token_sparsity, 1, self.weight.size(0))
            
                # Now remove the overlap
                mask = ~torch.isin(core_neurons, base_neurons)
                unique_core_neurons = core_neurons[mask]
                
                # Now get the rest of neurons to load from unique_core_neurons
                unique_core_neurons_to_compute = unique_core_neurons[:int((self.sparsity - self.base_neurons_percent) * self.original_neurons_num)]      
                indices_all = torch.cat((base_neurons, unique_core_neurons_to_compute))
            else:
                indices_all = base_neurons
            
            common.reorder_tensor(self.weight, indices_all, is_reverse=True)
            self.is_reorderd = True
            end_index = indices_all.shape[0]
            if (self.device == torch.device("cuda")):
                self.filtered_W = self.weight[0:end_index, :].contiguous()
                self.filtered_bias = self.bias[0:end_index].contiguous()
            else:
                self.filtered_W = self.weight[0:end_index, :]
                self.filtered_bias = self.bias[0:end_index]
                
            
            if self.num == (self.start_num + 1):
                indices_list_all=[]
                
            indices_list_all.append(indices_all)
            
            if self.num == self.end_num - 1:
                previous_indices_list_all = indices_list_all.copy()
            
        else:
            true_value = x @ self.filtered_W.T + self.filtered_bias

        return true_value


class ReduceLayer_fc2(nn.Module):
    def __init__(self, weight, bias, start_num, num, name, sparsity, memory_limit, cpu_only):
        super(ReduceLayer_fc2, self).__init__()
        
        self.device = torch.device("cpu") if cpu_only else torch.device("cuda")
        
        self.bias = bias.to(self.device)
        self.weight = weight.to(self.device)
        self.memory_limit = memory_limit
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.num = num
        self.name = name
        self.sparsity = sparsity
        self.is_reorderd = False
        
    def forward(self, x):
        global indices_list_all, previous_indices_list_all
        if x.size(0)>1:
            self.weight_updated = False
            
            if self.is_reorderd:
                indices = previous_indices_list_all[self.num - (self.start_num + 1)]
                common.reorder_tensor(self.weight, indices, is_reverse=False, is_restore=True)
            
            true_value = x @ self.weight.T + self.bias
            
        else:
            if not self.weight_updated:
                indices = indices_list_all[self.num - (self.start_num + 1)]
                common.reorder_tensor(self.weight, indices, is_reverse=False)
                self.is_reorderd = True
                end_index = indices.shape[0]
                if (self.device == torch.device("cuda")):
                    self.filtered_W = self.weight[:, 0:end_index].contiguous()
                    self.filtered_bias = self.bias[0:end_index].contiguous()
                else:
                    self.filtered_W = self.weight[:, 0:end_index]
                    self.filtered_bias = self.bias[0:end_index]
                self.weight_updated = True
                    
            
            true_value = x @ self.filtered_W.T + self.filtered_bias
        return true_value


def convert_opt_model_partinfer(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, original_neurons_num, partinfer_method_config):
    
    if not USE_PARTINFER_IMPROVEMENTS:
        raise RuntimeError("PARTINFER Improvements / partial loading needs to be activated for partinfer Method")

    for name, module in tqdm(model.named_modules(), desc="Convert Opt Models"):
        if "fc1" in name or "fc2" in name:
            num=int(name.split('.')[3])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
                    
                if "fc1" in name:
                    NewLayer = ReduceLayer(module.weight, module.bias, sparsity, token_sparsity, start_num, end_num,memory_limit, cpu_only, num, original_neurons_num, partinfer_method_config)
                else:
                    NewLayer = ReduceLayer_fc2(module.weight, module.bias, start_num, num, name, sparsity, memory_limit, cpu_only)

                setattr(parent, attr_name, NewLayer)
                del module
                
    gc.collect()
    
    return model
 
