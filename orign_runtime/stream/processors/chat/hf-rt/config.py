import os
import json


class Config:
    # Kafka configurations
    BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(
        ","
    )
    INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "input_topic")
    OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "output_topic")
    GROUP_ID = os.getenv("KAFKA_GROUP_ID", "my_consumer_group")

    # Model configurations
    MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
    TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")

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
