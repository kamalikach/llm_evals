import json 
import re

def evaluate(*args):
    response = args[0]
    example = args[1]
    letters = [ 'A', 'B', 'C', 'D' ]

    matches = re.findall(r'\b([A-D])\b', response)
    if matches:
        print(matches[-1], example['answer'])
        return matches[-1] == letters[example['answer']]
    else:
        return False
