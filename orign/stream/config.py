import os

from pydantic_settings import BaseSettings


class BaseConfig:
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
    ACCEPTS = os.getenv("ACCEPTS", "text").split(",")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
