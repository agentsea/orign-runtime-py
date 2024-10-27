import os
import json


class Config:
    # QUEUE configurations
    BOOTSTRAP_SERVERS = os.getenv("QUEUE_BOOTSTRAP_SERVERS", "localhost:9092").split(
        ","
    )
    QUEUE_TYPE = os.getenv("QUEUE_TYPE", "kafka")
    INPUT_TOPIC = os.getenv("QUEUE_INPUT_TOPIC", "input_topic")
    OUTPUT_TOPIC = os.getenv("QUEUE_OUTPUT_TOPIC", "output_topic")
    GROUP_ID = os.getenv("QUEUE_GROUP_ID", "my_consumer_group")

    # Model configurations
    MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
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