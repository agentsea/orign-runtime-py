import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRebalanceListener as KafkaConsumerRebalanceListener, TopicPartition, OffsetAndMetadata
from aiokafka.structs import RecordMetadata
from aiokafka.errors import CommitFailedError, KafkaError
from typing import Optional, Callable, Any, List, Dict
from pydantic import BaseModel

from ..config import BaseConfig
from .base import AsyncMessageConsumer, AsyncMessageProducer

class ConsumerRebalanceListener(KafkaConsumerRebalanceListener):
    def __init__(self, consumer: 'AsyncKafkaMessageConsumer'):
        self.consumer = consumer

    async def on_partitions_revoked(self, revoked: List[TopicPartition]) -> None:
        print(f"Partitions revoked: {revoked}")
        # Process pending messages and commit offsets
        await self.consumer.commit_on_revoke(revoked)

    async def on_partitions_assigned(self, assigned: List[TopicPartition]) -> None:
        print(f"Partitions assigned: {assigned}")

class AsyncKafkaMessageConsumer(AsyncMessageConsumer):
    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.pending_messages: Dict[TopicPartition, List[Any]] = {}

    async def start(self) -> None:
        consumer_conf = {
            "bootstrap_servers": ",".join(self.config.BOOTSTRAP_SERVERS),
            "group_id": self.config.GROUP_ID,
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,  # Manual commit
            "max_poll_interval_ms": 900000,
        }
        self.consumer = AIOKafkaConsumer(**consumer_conf)
        listener = ConsumerRebalanceListener(self)
        await self.consumer.start()
        # Corrected: Subscribe without 'await'
        self.consumer.subscribe(self.config.INPUT_TOPICS, listener=listener)
        print(f"Initialized AsyncKafkaMessageConsumer with config: {consumer_conf}")

    async def get_messages(self, timeout: float = 1.0) -> Optional[Dict[TopicPartition, List[Any]]]:
        try:
            messages = await self.consumer.getmany(timeout_ms=int(timeout * 1000))
            # Keep track of pending messages
            for tp, msgs in messages.items():
                if tp not in self.pending_messages:
                    self.pending_messages[tp] = []
                self.pending_messages[tp].extend(msgs)
            return messages
        except Exception as e:
            print(f"Error getting messages: {e}")
            return None

    async def commit(self) -> None:
        try:
            await self.consumer.commit()
            # Clear pending messages after successful commit
            self.pending_messages.clear()
        except CommitFailedError as e:
            print(f"Commit failed: {e}")
        except Exception as e:
            print(f"Error during commit: {e}")

    async def commit_on_revoke(self, revoked_partitions: List[TopicPartition]) -> None:
        try:
            # Process and commit pending messages for revoked partitions
            for tp in revoked_partitions:
                if tp in self.pending_messages:
                    # Process pending messages
                    for msg in self.pending_messages[tp]:
                        # Process the message (custom logic)
                        pass  # Replace with actual processing code
                    # Get current offset position
                    position = await self.consumer.position(tp)
                    # Commit offset for this partition
                    await self.consumer.commit({
                        tp: OffsetAndMetadata(position, "")
                    })
                    # Remove partition from pending messages
                    del self.pending_messages[tp]
        except Exception as e:
            print(f"Error committing on revoke: {e}")

    async def close(self) -> None:
        if self.consumer:
            await self.consumer.stop()


class AsyncKafkaMessageProducer(AsyncMessageProducer):
    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        producer_conf = {
            "bootstrap_servers": ",".join(self.config.BOOTSTRAP_SERVERS),
            "acks": "all",
            "enable_idempotence": True,    # Ensure idempotent producer
            "linger_ms": 5,                # Adjust as needed for batching
            # "compression_type": "gzip",  # Optional: Enable compression
            # Additional configurations can be added here
        }
        self.producer = AIOKafkaProducer(**producer_conf)
        await self.producer.start()
        print(f"Initialized AsyncKafkaMessageProducer with config: {producer_conf}")

    async def produce(
        self,
        value: BaseModel,
        topic: str,
        callback: Optional[Callable[[RecordMetadata, Optional[Exception]], None]] = None,
        partition: Optional[int] = None
    ) -> None:
        if not self.producer:
            raise RuntimeError("Producer is not started. Call start() before producing messages.")
        
        if not isinstance(value, BaseModel):
            raise ValueError("Value must be a Pydantic model: ", value)

        try:
            serialized_value = value.model_dump_json().encode('utf-8')
        except Exception as e:
            raise ValueError(f"Error serializing value: {e} value: {value}")
        
        try:
            future = await self.producer.send(topic, serialized_value, partition=partition)
            # Await the future to ensure delivery
            record_metadata = await future
            if callback:
                callback(record_metadata, None)
        except KafkaError as e:
            if callback:
                callback(None, e)
            else:
                print(f"Error producing message: {e}")

    async def flush(self) -> None:
        if self.producer:
            await self.producer.flush()

    async def close(self) -> None:
        if self.producer:
            await self.producer.stop()