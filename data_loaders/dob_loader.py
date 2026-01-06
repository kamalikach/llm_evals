import json 

def load(config):
    path = config.get('file_path')

    print(path)
    dataset = []

    with open(path, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                obj = json.loads(line)
                obj['prompt'] = f"When was {obj['name']} born?"
                dataset.append(obj)

    print(dataset[0])
    return dataset




