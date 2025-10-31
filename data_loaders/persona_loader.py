import json 

def load(*args):
    path = 

    subset = subset.remove_columns([col for col in subset.column_names if col != "target_text"])
    subset = subset.rename_column("target_text", "prompt")
    return subset


