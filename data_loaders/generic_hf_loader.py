from datasets import load_dataset

def load(config):
    dataset_name = config['dataset_name']
    split = config.get('split', 'train')
    dataset = load_dataset(dataset_name, split=split)
    
    subset_size = config.get('subset_size', len(dataset))
    subset = dataset.select(range(subset_size))

    subset = subset.rename_column(config['target_column'], "prompt")
    return subset


