import os

dir_path = "dense_results"

for d in ["cnn_dailymail", "xsum", "squadv2", "mlqa_en_en", "wmt16-de-en", "wmt16-ro-en"]:
    full_path = f"{dir_path}/{d}/"

    for model in ["llama3-1b", "llama3-3b", "qwen2.5-3b"]:
        
        old_file = full_path + f"reference_{model}-4bit.json"
        new_file = full_path + f"quantized_{model}-4bit.json"
        
        os.rename(old_file, new_file)