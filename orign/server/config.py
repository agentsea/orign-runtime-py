import os
import json


class Config:
    @staticmethod
    def _get_required_env(key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    # QUEUE configurations
    BOOTSTRAP_SERVERS = _get_required_env("QUEUE_BOOTSTRAP_SERVERS").split(",")
    QUEUE_TYPE = _get_required_env("QUEUE_TYPE")
    INPUT_TOPICS = _get_required_env("QUEUE_INPUT_TOPICS").split(",")
    OUTPUT_TOPIC = _get_required_env("QUEUE_OUTPUT_TOPIC")
    GROUP_ID = _get_required_env("QUEUE_GROUP_ID")

    # Model configurations
    MODEL_NAME = _get_required_env("HF_MODEL_NAME")
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
    TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")
    TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

    # Fix for DEVICE_MAP to handle string or dict
    device_map_raw = os.getenv("DEVICE_MAP", "auto")
    if device_map_raw == "auto":
        DEVICE_MAP = "auto"
    else:
        try:
            DEVICE_MAP = json.loads(device_map_raw)
        except json.JSONDecodeError:
            DEVICE_MAP = None  # or raise an error if you prefer

    # Batch processing
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "500"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
    MAX_IMAGES_PER_PROMPT = int(os.getenv("MAX_IMAGES_PER_PROMPT", "1"))