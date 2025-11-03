from .LlamaInstructModel import LlamaInstructModel


def load_model_from_config(model_config):
    model_class = infer_class_from_model_name(model_config['model_name'])
    instance = model_class(model_config['model_name'])
    instance.load(model_config.get('model_args'))
    return instance

def infer_class_from_model_name(model_name):
    model_name_lower = model_name.lower()

    if "llama" in model_name_lower and "instruct" in model_name_lower:
        return LlamaInstructModel
    elif "gpt" in model_name_lower:
        return GPTModel

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported patterns: models containing 'llama' or 'gpt'. ")


