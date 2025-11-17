from .LlamaInstructModel import LlamaInstructModel
from .GPTOSSModel import GPTOSSModel
from .PhiModel import PhiModel
from .LlamaFTModel import LlamaFTModel

def load_model_from_config(model_config):
    model_class = infer_class_from_model_name(model_config['model_name'], model_config.get('model_type', ""))
    instance = model_class(model_config['model_name'])
    instance.load(model_config.get('model_args'))
    return instance

def infer_class_from_model_name(model_name, model_type):
    model_name_lower = model_name.lower()

    if "llama" in model_name_lower and "ft" in model_type:
        return LlamaFTModel
    if "llama" in model_name_lower and "instruct" in model_name_lower:
        return LlamaInstructModel
    elif "gpt" in model_name_lower and "oss" in model_name_lower:
        return GPTOSSModel
    elif "phi" in model_name_lower:
        return PhiModel

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported patterns: models containing 'llama' or 'gpt' or 'phi' ")


