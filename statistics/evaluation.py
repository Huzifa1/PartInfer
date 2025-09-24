import json
import argparse
import csv

def load_and_normalize_activations(file_path, token_count):
    with open(file_path, "r") as file:
        lines = file.readlines()[2:]
        activations = []
        for line in lines:
            parts = line.split(":")[1].split(",")
            activations.append([float(x.strip()) / token_count for x in parts])
    return activations

def read_datasets_file(file):
    datasets = {}
    with open(file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            datasets[row[0]] = row[1]
    return datasets

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default="../neuron_files/opt-6.7b/model_neurons.json", help='Path for the output JSON file.')
parser.add_argument('--datasets', type=str, default="datasets.csv", help='Path to CSV file with dataset statistics file with format: name,path')

args = parser.parse_args()
output_file = args.output_file
statistics = read_datasets_file(args.datasets)

normalized_activations = {}
for name, path in statistics.items():
    with open(path, "r") as f:
        lines = f.readlines()
        num_tokens = int(lines[-1].split("Number of tokens: ")[1])
    normalized_activations[name] = load_and_normalize_activations(path, num_tokens)

num_layers = len(next(iter(normalized_activations.values()))) - 1
finalResult = []
for i in range(num_layers):
    combined = zip(*(normalized_activations[ds][i] for ds in statistics))
    averaged_layer = [sum(vals) / len(vals) * 100 for vals in combined]
    finalResult.append(averaged_layer)


finalSorted = []
for layer in finalResult:
    sorted_with_indices = sorted(enumerate(layer), key=lambda x: x[1], reverse=True)
    sorted_indices = [i for i, _ in sorted_with_indices]
    sorted_values = [v for _, v in sorted_with_indices]
    finalSorted.append(sorted_indices)

write_file = True
if write_file:
    with open(output_file, "w") as file:
        json.dump({"neurons": finalSorted}, file)