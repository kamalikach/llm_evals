from datasets import load_dataset

def load():
    dataset = load_dataset("wiki_bio", split="train")
    subset = dataset.select(range(50))

    subset = subset.remove_columns([col for col in subset.column_names if col != "target_text"])
    subset = subset.rename_column("target_text", "text")
    return subset


