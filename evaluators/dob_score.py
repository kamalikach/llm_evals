import json 
from dateutil import parser
from datetime import date

def extract_date(s):
    try:
        dt = parser.parse(s, fuzzy=True, default=date.today())
        return dt.date()
    except (ValueError, OverflowError):
        return None

def evaluate(*args):
    response = args[0]
    example = args[1]

    response_dob = extract_date(response)
    return int(response_dob == example['fake_dob'])


