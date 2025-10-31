import argparse
import yaml
import torch
import importlib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
import os 

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tokenizer

def prompt_response(model, tokenizer, system_prompt, user_prompt):
    prompt = tokenizer.apply_chat_template( [{"role": "user", "content": system_prompt + ':' + user_prompt}],
        tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=model.config.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_response = response.split("assistant", 1)[1].strip()
    return cleaned_response


def run_experiments(config):
    #model, system_prompt, data_loader, output_dir, evaluator
    model, tokenizer = load_model(config["model"])
    
    with open(config["system_prompt"], "r") as f: 
        system_prompt = f.read()

    module=importlib.import_module(config["data_loader"])
    dataset = module.load()

    evaluator_func = get_evaluator(config["evaluator"])

    with open(config["output_file"], "w") as f:
        count = 0.0
        score = 0.0
        for example in dataset:
            response = prompt_response(model, tokenizer, system_prompt, example['prompt'])
            accuracy = evaluator_func(response, example)
            
            f.write(json.dumps({'response': response, 'eval': accuracy}) + '\n')
            score += float(accuracy)
            count += 1.0

    return score/count


def get_dataset_iterator(config):
    #takes in config, returns a list of dicts to be passed to the dataset loader
    dataset_list = config.get('dataset_list')

    if dataset_list is None:
        #single dataset
        return [ { 'path': "", 'desc': config['data_loader'] } ]
    elif isinstance(dataset_list, str):
        #a string which is a pathname
        path = Path(dataset_list)
        if not path.is_dir():
            raise ValueError(f"{dataset_list} is a string but not a valid directory")
        
        return [ { 'path': f.resolve(), 'desc': f.stem } for f in path.iterdir() if f.is_file() ]
    elif instance(dataset_list, list):
        return [ { 'path':"", 'desc': x } for x in dataset_list ]


def run_experiment_list(config):
    model, tokenizer = load_model(config['model'])

    system_prompt_path = config.get('system_prompt')
    if system_prompt_path is not None:
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
    else:
        system_prompt = ""
    print('System prompt:', system_prompt)
    
    evaluator_module = importlib.import_module('evaluators.'+ config['evaluator'])

    dataset_iterator = get_dataset_iterator(config)
    score_list = []

    os.makedirs(os.path.dirname(config['output_dir']), exist_ok=True)

    for dataset_desc in dataset_iterator:
        data_loader = importlib.import_module('data_loaders.' + config['data_loader'] + '_loader')
        dataset = data_loader.load(dataset_desc)

        output_file = config['output_dir'] + dataset_desc['desc'] + '.jsonl'
        with open(output_file, "w") as f:
            count = 0.0
            score = 0.0

            for example in dataset:
                response = prompt_response(model, tokenizer, system_prompt, example['prompt'])
                result = evaluator_module.evaluate(response, example, system_prompt)
        
                f.write(json.dumps({"response": response, "eval": result}) + '\n')
                score += float(result)
                count = count + 1.0
        print('Dataset:', dataset_desc['desc'])
        print('Score:', score/count)
        score_list.append(score/count)
    return score_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic llm evals")
    parser.add_argument("--config",help="Path to config file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    score_list = run_experiment_list(config)
    print('Scores:', score_list)


