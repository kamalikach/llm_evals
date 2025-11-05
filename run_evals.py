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
from omegaconf import DictConfig, OmegaConf, open_dict
import datetime
from datetime import datetime
import glob 
from copy import deepcopy

def get_output_filename(cfg):
    base = './outputs/' 
    model_short = cfg.model.model_name.split('/')[-1]
    if cfg.data.get('data_type') == 'list':
        data = Path(cfg.data.data_loader_args.file_path).stem 
    else:
        data = cfg.data.data_loader_args.dataset_name
        print(data)
    evaluator = cfg.data.evaluator
    system_prompt = Path(cfg.data.system_prompt).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
    return base + f"{model_short}_{data}_{evaluator}_{system_prompt}_{timestamp}.json"



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

    data_loader = importlib.import_module('data_loaders.' + config.data['data_loader'])
    data_loader_args = config.data.get('data_loader_args')    
    dataset = data_loader.load(data_loader_args)
    
    output_file = get_output_filename(config)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

def get_config_list(cfg):
    file_paths = sorted(glob.glob(cfg.data['file_list']))  # alphabetic sort
    if not file_paths:
        raise FileNotFoundError(f"No files found for pattern: {cfg.data['file_list']}")

    cfg_list = []
    for filename in file_paths:
        new_config = deepcopy(cfg)
        with open_dict(new_config.data):
            new_config.data.data_loader_args['file_path'] = filename
        system_prompt_type = cfg.data.get('system_prompt_type')
        if system_prompt_type == 'list':
            cfg.data.system_prompt = str(Path(cfg.data.get('system_prompt_path')) / (Path(filename).stem + '.txt'))
        cfg_list.append(new_config)
    return cfg_list




@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    print(cfg)
    print(cfg.model)
    print(cfg.data)

    if cfg.data.get('data_type') == 'list':
        cfg_list = get_config_list(cfg)
    else:
        cfg_list = [ cfg ]
    
    print(cfg_list)
    for cfg in cfg_list:
        score = run_experiment(cfg)
        print('Score:', score)

if __name__ == "__main__":
    main()


