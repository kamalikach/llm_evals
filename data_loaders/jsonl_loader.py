import json 

def load(config):
    path = config.get('file_path')

    print(path)
    with open(path, "r") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
        modified_dataset = [{("prompt" if k == "question" else k): v for k, v in d.items()} for d in dataset]
    return modified_dataset




