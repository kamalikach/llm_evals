import json 

def load(*args):
    path = args[0]['path']
    desc = args[0]['desc']

    print(path)
    if path.suffix == ".jsonl":
        with open(path, "r") as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        modified_dataset = [{("prompt" if k == "question" else k): v for k, v in d.items()} for d in dataset]
        return modified_dataset
    else:
        return []




