import json 

def correct_json_format(*args):
    text = args[0]
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False

