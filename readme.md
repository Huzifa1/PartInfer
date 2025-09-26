# PartInfer: Enabling LLM Ineference On Edge Devices

## Overview

PartInfer, a neuron-level optimization framework that enables efficient LLM inference on edge devices by exploiting the task-specific activation patterns of neurons. 

![Overview](assets/overview.svg)

Our approach is split into 2 phase, offline (left) and online (right). During the offline phase, we run LLM Profiler to generate neurons files. During the offline phase, partial loading helps to reduce memory footprint while partial computation helps to reduce computational overhead.

## Demo

https://github.com/user-attachments/assets/a6639637-28d6-4c2c-a94a-6506aad08208

## Install

This project uses a Python virtual environment to manage dependencies. Follow the steps below to reproduce the same environment.

### 1. Clone the Repository

```bash
git clone https://github.com/Huzifa1/PartInfer
cd PartInfer
```

### 2. Create a Virtual Environment

Make sure you have Python (>=3.x) installed.

```bash
python3 -m venv .partinfer-venv
```

### 3. Activate the Virtual Environment

**Linux / macOS**:
    
```bash
source venv/bin/activate
```
    
**Windows (PowerShell)**:
    
```powershell
.\venv\Scripts\Activate
```
    

### 4. Install Dependencies

The required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Notes
    
To deactivate the virtual environment:
    
```bash
deactivate
```

### 5. Download of model and dataset

You can download the required models and datasets to the specified location using the following command.

```python
python download.py --model_name  $model --checkpoint_path /PATH/TO/SAVE/MODEL --data_name $dataset --data_config $data_config  --datasave_path /PATH/TO/SAVE/DATASET

# e.g.: python download.py --model_name  'facebook/opt-6.7b' --checkpoint_path "./models/opt-6.7b" --data_name "truthfulqa/truthful_qa" --data_config "generation" --datasave_path "./dataset/trurthul_qa"
# e.g.: python download.py --model_name  'meta-llama/Llama-3.1-8B' --checkpoint_path "./models/llama3-8b" --data_name "truthfulqa/truthful_qa" --data_config "generation" --datasave_path "./dataset/trurthul_qa" --token "xxxxx"
```

## Create Neurons Files

This repository provides utilities for creating **neuron files** from datasets using two scripts located in the `statistics` folder:

- `gather_statistics.py`
- `evaluation.py`

---

### 1. Gather Statistics

The script **`gather_statistics.py`** generates score files for the datasets.  
Each score file contains:
- Scores calculated for every neuron
- The number of processed tokens

### Parameters

- `--model_name`: name of the model to use
- `--max_new_tokens`: maximum number of tokens to generate  
- `--prompt_limit`: number of prompts to process from the dataset file
- `--batch_size`: value of batch size used
- `--configs`: path to a CSV file specifying the output path, datasets, and task type

Additionally, the function **`pre_process_prompt`** can be customized to define how prompts should be preprocessed before execution.

Example:
```
cd statistics
python gather_statistics.py --model_name ../models/opt-6.7b --max_new_tokens 128 --prompt_limit 5000 --batch_size 16 --configs all_datasets_config.csv
```

---

### 2. Evaluation

Once score files have been created, run **`evaluation.py`** to generate the final neuron file.

### Parameters

- `--input_path`: path to dir that contains statistics files
- `--datasets`: path to a CSV file specifying which score files to use
- `--output_file`: output path for the neuron file

Example:
```
python evaluation.py --input_path statistics_files/llama3-3b/ --datasets model_neurons.csv --output_file ../neuron_files/llama3-3b/model_neurons.json
```

## Run Inference

To run inference using PartInfer, the `partinfer.py` file is used. This is an example command to run inference using model `llama3-3b` stored in folder `models/llama3-3b` with method `partinfer` in `chat` function:
```
python partinfer.py --model_name 'llama3-3b' --checkpoint_path 'models/llama3-3b' --cpu_only --method partinfer --function predefined_prompts
```

This are the available parameters for `partinfer.py`:
- `--model_name`: name of the model to be used
- `--checkpoint_path`: path of the model to use
- `--method`: Method to apply during inference\
  - `partinfer`: PartInfer implementation
  - `coreinfer`: [CoreInfer implementation](https://github.com/wangqinsi1/CoreInfer)
  - `dense`: compute all loaded neurons
- `--function`:
  - `predefined_prompts` runs inference on predefined prompts
  - `chat` allows chat functionality
  - `normal` evaluates the prompt given in `prompt` parameter
- `--sparsity`: set portion of neurons to compute ($\phi = \delta + \epsilon$), default to $0.4$
- `--task_type`: ``QA``, ``Summarize``, ``translate_de_en``
- `--prompt`: prompt for `normal` function
- `--device`: device to use, default to `cuda` for NVIDIA GPU
- `--cpu_only`: add this parameter to only use the cpu for inference
- `--start_num`: first layer to apply PartInfer, default to $3$
- `--end_num`: last layer to apply PartInfer, default to $25$
- `--num_tokens_to_generate`: Maximum number of new tokens
- `--sampling-method`: `greedy` or `top-p`
- `--token_sparsity`: Token Sparsity level
- `--top_p`: used for top-p sampling
- `--use_partial_loading`: add this parameter to enables partial loading from PartInfer, automatically enabled by `--method partinfer`
> Caution: parameter `--cpu_only` needs to be added if CUDA GPU is not available

This parameters are only used when using `--method partinfer`:
- `--base_neurons_percent`: portion $\delta$ of base neurons, default to $0.3$
- `--loaded_neurons_percent`: portion $\gamma$ of neurons to load, default to $0.7$
- `--model_neurons_filepath`: path to model neuron file (found in `neuron_files`)
- `--dataset_neurons_filepath`: path to dataset neuron file (found in `neuron_files`)
- `--base_neurons_type`: `model` to use model neurons as base neurons, `dataset` to set dataset neurons as base neurons, default to `dataset`
- `--mask_filepath`: path for generation of mask file

> When using otherf scripts, e.g. `evaluation/evaluate_task.py`, make sure in the file `transformers/partinfer_variables/partinfer_improvements.py` the variable ``USE_PARTINFER_IMPROVEMENTS`` is set to `True` to enable partial loading.

## Run Task Evaluation

In order to run task evaluation using **PartInfer**, we use the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) library.
We have built an automation script that runs evaluation of a certain model using a certain method on a number of datasets.

The default datasets used in our experiments are listed automatically in the file. You can add or remove datasets as you wish.
For a list of supported tasks, check the [lm-evaluation-harness tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

### Setting Up

1. **Choose your model**
   Set the variable `MODEL_NAME` to the name of the folder where your model is located.
   Make sure the model is placed in:

```
PartInfer/models/your_model
```

2. **Reproduce paper results**
Modify the following two variables: `METHOD` and `USE_PARTINFER_IMPROVEMENTS`.

- `METHOD="partinfer"`
  Runs the model using PartInfer.

- `METHOD="dense" & USE_PARTINFER_IMPROVEMENTS="True"`
  Runs PartInfer with partial loading only (no partial computation).

- `METHOD="dense" & USE_PARTINFER_IMPROVEMENTS="False"`
  Reference model (all neurons are loaded and fully computed).

- `METHOD="coreinfer" & USE_PARTINFER_IMPROVEMENTS="False"`
  Default CoreInfer approach.

- `METHOD="coreinfer" & USE_PARTINFER_IMPROVEMENTS="True"`
  CoreInfer's approach with our partial loading.

- `METHOD="coreinfer_random_loading"`
  CoreInfer with random partial loading.

Other variables are self-explanatory and are explained in more detail in the paper.


### Running the Script

After setting the variables, run the script:

```bash
bash evaluation_partinfer.sh
```

## Run Decoding Speed Evaluation

You can run the following command to see the decoding speed.

```bash
cd evaluation
python3 evaluate_speed.py --model_name $MODEL_NAME --checkpoint_path $PATH_TO_MODEL --num_tokens_to_generate $NUM_TOKENS --method $METHOD

# e.g. to run with PartInfer and Llama3.2-3B while loading 70% and computing 40% of neurons:
python3 evaluate_speed.py --model_name "llama3-3b" --checkpoint_path ../models/llama3-3b --num_tokens_to_generate 256 --method "partinfer" --loaded_neurons_percent 0.7 --sparsity -0.4
```

For more control on different percentages and neurons files, run the following command or refer to the paper:
```bash
python3 evaluate_speed.py --help
```
