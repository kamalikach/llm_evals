import argparse
import yaml
import torch
import importlib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluators import get_evaluator

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tokenizer

def prompt_response(model, tokenizer, system_prompt, user_prompt):
    return ""


def run_experiments(config):
    #model, system_prompt, data_loader, output_dir, evaluator
    print(config)
    model, tokenizer = load_model(config["model"])
    
    with open(config["system_prompt"], "r") as f: 
        system_prompt = f.read()
    print(system_prompt)

    module=importlib.import_module(config["data_loader"])
    dataset = module.load_dataset()

    evaluator_func = get_evaluator(config["evaluator"])

    with open(config["output_file"], "w") as f:
        for example in dataset:
            response = prompt_response(model, tokenizer, system_prompt, example['text'])
            accuracy = evaluator_func(response)

            f.write(json.dumps({'response': response, 'eval': accuracy}) + '\n')









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic llm evals")
    parser.add_argument("--config",help="Path to config file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_experiments(config)


