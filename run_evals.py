import argparse
import yaml
import torch
import importlib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluators import get_evaluator
import json

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
    print(config)
    model, tokenizer = load_model(config["model"])
    
    with open(config["system_prompt"], "r") as f: 
        system_prompt = f.read()

    module=importlib.import_module(config["data_loader"])
    dataset = module.load()

    evaluator_func = get_evaluator(config["evaluator"])

    with open(config["output_file"], "w") as f:
        for example in dataset:
            response = prompt_response(model, tokenizer, system_prompt, example["text"])
            print(response)
            accuracy = evaluator_func(response)
            print(accuracy)

            f.write(json.dumps({'response': response, 'eval': accuracy}) + '\n')









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic llm evals")
    parser.add_argument("--config",help="Path to config file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_experiments(config)


