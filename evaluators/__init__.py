import json
from .json_evaluators import *

EVALUATOR_REGISTRY = { 
        "json_format": correct_json_format
}

def get_evaluator(evaluator_name):
    if evaluator_name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator: {evaluator_name}. Available: {list(EVALUATOR_REGISTRY.keys())}")
    return EVALUATOR_REGISTRY[evaluator_name]

def list_evaluators():
    return list(EVALUATOR_REGISTRY.keys())

