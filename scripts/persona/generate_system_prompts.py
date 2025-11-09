from model_loaders import load_model_from_config
from pathlib import Path
import os

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_instruction = "/home/kamalika/llm_evals/scripts/persona/instruction.txt"

input_path = "/data/kamalika/evals/persona"
output_path = "/home/kamalika/llm_evals/system_prompts/persona/prompt_list/"

def main():
    model_cfg = { 'model_name': model_name }
    model = load_model_from_config(model_cfg)

    with open(model_instruction, "r") as f:
        system_prompt = f.read()

        inputs = Path(input_path)

        for json_file in inputs.glob('*.jsonl'):
            print(json_file.name)
            output_file = Path(output_path) / f"{json_file.stem}.txt"
            os.makedirs(output_file.parent, exist_ok=True)
            
            with open(output_file, "w") as f:
                response = model.prompt_response(system_prompt, json_file.name)
                f.write(response)


if __name__ == "__main__":
    main()


