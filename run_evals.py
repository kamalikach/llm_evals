import argparse
import yaml
import torch
import importlib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
import os 
from model_loaders import load_model_from_config

def run_experiment(config):
    model = load_model_from_config(config['model'])

    system_prompt_path = config.get('system_prompt')
    if system_prompt_path is not None:
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
    else:
        system_prompt = ""
    print('System prompt:', system_prompt)
    
    evaluator_module = importlib.import_module('evaluators.'+ config['evaluator'])

    data_loader = importlib.import_module('data_loaders.' + config['data_loader'] + '_loader')
    data_loader_args = config.get('data_loader_args')    
    dataset = data_loader.load(data_loader_args)

    output_file = config['output_file'] + '.jsonl'
    with open(output_file, "w") as f:
        count = 0.0
        score = 0.0

        for example in dataset:
            response = model.prompt_response(system_prompt, example['prompt'])
            result = evaluator_module.evaluate(response, example, system_prompt)
        
            f.write(json.dumps({"response": response, "eval": result}) + '\n')
            score += float(result)
            count = count + 1.0
    if count > 0: 
        print('Score:', score/count)
        return score/count
    else:
        return None 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic llm evals")
    parser.add_argument("--config",help="Path to config file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    score = run_experiment(config)
    print('Score:', score)


