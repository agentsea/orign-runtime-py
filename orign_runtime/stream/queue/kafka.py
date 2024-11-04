# main.py

from confluent_kafka import Consumer, Producer
from confluent_kafka.cimpl import Message
from typing import Optional, Callable, Any
from pydantic import BaseModel

from ..config import BaseConfig
from .base import MessageConsumer, MessageProducer


# Kafka implementation of the abstract classes
class KafkaMessageConsumer(MessageConsumer):
    def __init__(self, config: BaseConfig) -> None:
        consumer_conf = {
            "bootstrap.servers": ",".join(config.BOOTSTRAP_SERVERS),
            "group.id": config.GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 900000,
        }
        self.consumer: Consumer = Consumer(consumer_conf)
        self.consumer.subscribe([config.INPUT_TOPIC])
        print(f"Initialized KafkaMessageConsumer with config: {consumer_conf}")

    def poll(self, timeout: float = 1.0) -> Optional[Message]:
        return self.consumer.poll(timeout)

    def commit(self, message: Message) -> None:
        self.consumer.commit(message=message, asynchronous=False)

    def close(self) -> None:
        self.consumer.close()


class KafkaMessageProducer(MessageProducer):
    def __init__(self, config: BaseConfig) -> None:
        producer_conf = {
            "bootstrap.servers": ",".join(config.BOOTSTRAP_SERVERS),
        }
        self.producer: Producer = Producer(producer_conf)
        print(f"Initialized KafkaMessageProducer with config: {producer_conf}")

    def produce(
        self,
        value: BaseModel,
        topic: str,
        callback: Optional[Callable[[Any, Message], None]] = None,
    ) -> None:
        serialized_value = value.model_dump_json().encode('utf-8')
        self.producer.produce(topic=topic, value=serialized_value, callback=callback)
        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush()