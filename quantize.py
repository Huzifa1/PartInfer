import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main(input_model_path: str, output_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(input_model_path)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model_4bit = AutoModelForCausalLM.from_pretrained(
        input_model_path,
        device_map="auto",
        quantization_config=quantization_config
    )

    model_4bit.save_pretrained(
        output_model_path,
        safe_serialization=False
    )

    tokenizer.save_pretrained(output_model_path)


if __name__ == "__main__":
    
    # Read input and output model paths from command line arguments
    parser = argparse.ArgumentParser(description="Quantize a language model to 4-bit precision.")
    parser.add_argument("input_model_path", type=str, help="Path to the input pre-trained model.", required=True)
    parser.add_argument("output_model_path", type=str, help="Path to save the quantized model.", required=True)
    args = parser.parse_args()
    
    main(args.input_model_path, args.output_model_path)