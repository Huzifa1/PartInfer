import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common

indices_list_all = []

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, name):
        super(CustomMLPLayer, self).__init__()

        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")

        if "down" in name:
            neuron_num = int(weight.size(1) * sparsity)
            self.filtered_W = torch.zeros((weight.size(0),neuron_num)).to(torch.float16).to(device)
        else:
            neuron_num = int(weight.size(0) * sparsity)
            self.filtered_W = torch.zeros((neuron_num, weight.size(1))).to(torch.float16).to(device)

        self.device = device
        self.weight = weight.to(device)
        self.num = num
        self.name = name
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.end_num = end_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit        
        

    def forward(self, x):
        global indices_list_all

        if x.size(1)>1:
            self.weight_updated = False
            true_value = x @ self.weight.T

            if "down" in self.name:
                squeezed_x = x.squeeze()
                indices_all = common.get_core_neurons(squeezed_x, self.token_sparsity, self.sparsity, self.weight.size(1))
                                
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                    self.filtered_W = torch.zeros_like(self.weight).cuda().to(torch.float16)

                if (self.device == torch.device("cuda")):
                    self.filtered_W = self.weight[:, indices_all].contiguous()
                else:
                    self.filtered_W = self.weight[:, indices_all]
                
                if self.num == (self.start_num + 1):
                    indices_list_all=[]
               
                indices_list_all.append(indices_all)

                # self.weight = self.weight.cpu()
        else:
            if "down" not in self.name:
                if not self.weight_updated:
                    indices = indices_list_all[self.num - (self.start_num + 1)]
                    if (self.device == torch.device("cuda")):
                        self.filtered_W = self.weight[indices,:].contiguous()
                    else:
                        self.filtered_W = self.weight[indices,:]
                    if self.memory_limit:
                        self.weight = self.weight.cpu()
                    self.weight_updated = True

            true_value = x @ self.filtered_W.T
        return true_value


def convert_llama_qwen_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only):
    
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

                NewLayer = CustomMLPLayer(module.weight, num, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, name)
                setattr(parent, attr_name, NewLayer)
                del module

    gc.collect()
    return model