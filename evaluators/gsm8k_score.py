import json 
import re

def evaluate(*args):
    prediction = args[0]
    example = args[1]

    label = example['answer'].split("####")[-1].strip()

    response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    extracted = numbers[-1] if numbers else response
    return str(extracted).lower() == str(label).lower()

