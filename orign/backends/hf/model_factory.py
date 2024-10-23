from .models.qwen2 import Qwen2
from .model_handler import ModelHandler
from .config import Config

def get_model_handler(model_name: str) -> ModelHandler:
    if model_name in Qwen2.supported_models():
        return Qwen2(Config)
    else:
        raise ValueError(f"Model {model_name} not supported")
