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
import hydra
from omegaconf import DictConfig, OmegaConf

def get_output_fname(config):
    base = './outputs/'
    return base + 'outputs.jsonl'

def run_experiment(config):
    print(config.model)
    model = load_model_from_config(config.model)

    system_prompt_path = config.data.get('system_prompt')
    print(system_prompt_path)
    if system_prompt_path is not None:
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
    else:
        system_prompt = ""
    print('System prompt:', system_prompt)
    
    evaluator_module = importlib.import_module('evaluators.'+ config.data['evaluator'])

    data_loader = importlib.import_module('data_loaders.' + config.data['data_loader'] + '_loader')
    data_loader_args = config.data.get('data_loader_args')    
    dataset = data_loader.load(data_loader_args)

    
    output_file = get_output_fname(config)
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


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    print(cfg)
    print(cfg.model)
    print(cfg.data)

    score = run_experiment(cfg)
    print('Score:', score)

if __name__ == "__main__":
    main()


