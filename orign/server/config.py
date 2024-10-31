import os


class Config:
    @staticmethod
    def _get_required_env(key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

    # QUEUE configurations
    BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(
        ","
    )
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    QUEUE_TYPE = _get_required_env("QUEUE_TYPE").lower()
    INPUT_TOPICS = _get_required_env("QUEUE_INPUT_TOPICS").split(",")
    GROUP_ID = _get_required_env("QUEUE_GROUP_ID")

    # Model configurations
    MODEL_NAME = _get_required_env("HF_MODEL_NAME")
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
    TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")
    TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

    # Fix for DEVICE_MAP to handle string or dict
    DEVICE = os.getenv("DEVICE", "auto")
    ACCEPTS = os.getenv("ACCEPTS", "text").split(",")

    # Batch processing
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "500"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
    MAX_IMAGES_PER_PROMPT = int(os.getenv("MAX_IMAGES_PER_PROMPT", "1"))
