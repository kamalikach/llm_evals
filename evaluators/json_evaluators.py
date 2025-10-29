import json 

def correct_json_format(text):
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False

