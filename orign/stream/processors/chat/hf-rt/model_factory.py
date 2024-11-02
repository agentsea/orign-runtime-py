from .models.qwen2 import Qwen2
from .models.molmo import Molmo
from .model_handler import ModelHandler
from .config import Config

def get_model_handler(model_name: str) -> ModelHandler:
    if model_name in Qwen2.supported_models():
        return Qwen2(Config)
    elif model_name in Molmo.supported_models():
        return Molmo(Config)
    else:
        raise ValueError(f"Model {model_name} not supported")
