from ..config import Config
from .base import MessageConsumer, MessageProducer, AsyncMessageConsumer, AsyncMessageProducer
from .kafka import KafkaMessageConsumer, KafkaMessageProducer
from .kafka_aio import AsyncKafkaMessageConsumer, AsyncKafkaMessageProducer
from .redis_aio import AsyncRedisMessageConsumer, AsyncRedisMessageProducer


def get_message_consumer(config: Config) -> MessageConsumer:
    if config.QUEUE_TYPE == "kafka":
        return KafkaMessageConsumer(config)
    else:
        raise ValueError(f"Unsupported queue type: {config.QUEUE_TYPE}")

def get_message_producer(config: Config) -> MessageProducer:
    if config.QUEUE_TYPE == "kafka":
        return KafkaMessageProducer(config)
    else:
        raise ValueError(f"Unsupported queue type: {config.QUEUE_TYPE}")
    

def get_message_consumer_async(config: Config) -> AsyncMessageConsumer:
    if config.QUEUE_TYPE == "kafka":
        return AsyncKafkaMessageConsumer(config)
    elif config.QUEUE_TYPE == "redis":
        return AsyncRedisMessageConsumer(config)
    else:
        raise ValueError(f"Unsupported queue type: {config.QUEUE_TYPE}")

def get_message_producer_async(config: Config) -> AsyncMessageProducer:
    if config.QUEUE_TYPE == "kafka":
        return AsyncKafkaMessageProducer(config)
    elif config.QUEUE_TYPE == "redis":
        return AsyncRedisMessageProducer(config)
    else:
        raise ValueError(f"Unsupported queue type: {config.QUEUE_TYPE}")
