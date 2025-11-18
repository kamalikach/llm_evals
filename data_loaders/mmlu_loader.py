from datasets import load_dataset

def choice_map(choices):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ", ".join(f"({letters[i]}) {choice}" for i, choice in enumerate(choices))

def load(config):
    dataset_name = 'cais/mmlu'
    split = 'test'
    subject = config.get('subject', 'abstract_algebra')
    dataset = load_dataset(dataset_name, subject, split=split)
    
    subset_size = config.get('subset_size', len(dataset))
    subset = dataset.select(range(subset_size))

    
    subset = subset.map(
            lambda x: {'prompt': x['question'] + " " + "Your choices: " + choice_map(x['choices'])}
    )
    print(subset[0])
    return subset

