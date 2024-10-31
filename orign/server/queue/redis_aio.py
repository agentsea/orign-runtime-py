from typing import Optional, Callable, Any, List, Dict
from pydantic import BaseModel
import redis.asyncio as redis
import time

from ..config import Config
from .base import AsyncMessageConsumer, AsyncMessageProducer


class AsyncRedisMessageConsumer(AsyncMessageConsumer):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.consumer_group = config.GROUP_ID
        self.consumer_name = f"{config.GROUP_ID}-{time.time()}"
        self.pending_messages: Dict[str, List[Any]] = {}
        self.last_ids: Dict[str, str] = {}

    async def start(self) -> None:
        self.redis = redis.from_url(self.config.REDIS_URL, decode_responses=True)

        # Create consumer group for each input topic
        for topic in self.config.INPUT_TOPICS:
            try:
                await self.redis.xgroup_create(
                    topic,
                    self.consumer_group,
                    mkstream=True,
                    id="0",  # Start from beginning
                )
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):  # Ignore if group already exists
                    raise

        print(f"Initialized AsyncRedisMessageConsumer for group: {self.consumer_group}")
        print(f"Watching topics: {', '.join(self.config.INPUT_TOPICS)}")

    async def get_messages(
        self, timeout: float = 1.0
    ) -> Optional[Dict[str, List[Any]]]:
        try:
            # Read messages from all input topics
            streams = {topic: ">" for topic in self.config.INPUT_TOPICS}
            messages = await self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams=streams,
                count=100,  # Adjust batch size as needed
                block=int(timeout * 1000),
            )

            if not messages:
                return None

            # Format messages similar to Kafka structure
            formatted_messages = {}
            for topic, msgs in messages:
                if topic not in self.pending_messages:
                    self.pending_messages[topic] = []

                message_list = []
                for msg_id, msg_data in msgs:
                    self.last_ids[topic] = msg_id
                    message = {
                        "topic": topic,
                        "offset": msg_id,
                        "value": msg_data[
                            "payload"
                        ],  # Assuming message is stored in 'payload' field
                    }
                    message_list.append(message)
                    self.pending_messages[topic].extend(message_list)

                formatted_messages[topic] = message_list

            return formatted_messages
        except Exception as e:
            print(f"Error getting messages: {e}")
            return None

    async def commit(self) -> None:
        try:
            # Acknowledge messages for each topic
            for topic, msg_id in self.last_ids.items():
                await self.redis.xack(topic, self.consumer_group, msg_id)
            self.pending_messages.clear()
            self.last_ids.clear()
        except Exception as e:
            print(f"Error during commit: {e}")

    async def stop(self) -> None:
        if self.redis:
            await self.redis.aclose()

    async def commit_on_revoke(self, revoked_partitions: List[Any]) -> None:
        """Redis streams don't use partitions, so we just commit pending messages."""
        await self.commit()

    async def close(self) -> None:
        """Alias for stop() to match the abstract interface."""
        await self.stop()


class AsyncRedisMessageProducer(AsyncMessageProducer):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.redis: Optional[redis.Redis] = None

    async def start(self) -> None:
        self.redis = redis.from_url(self.config.REDIS_URL, decode_responses=True)
        print(f"Initialized AsyncRedisMessageProducer")

    async def produce(
        self,
        value: BaseModel,
        topic: str,
        callback: Optional[Callable[[Any, Optional[Exception]], None]] = None,
        partition: Optional[int] = None,
    ) -> None:
        if not self.redis:
            raise RuntimeError(
                "Producer is not started. Call start() before producing messages."
            )

        try:
            # Serialize the message
            serialized_value = value.model_dump_json()
            # Add message to stream
            msg_id = await self.redis.xadd(topic, {"payload": serialized_value})
            if callback:
                callback(msg_id, None)
        except Exception as e:
            if callback:
                callback(None, e)
            else:
                print(f"Error producing message: {e}")

    async def flush(self) -> None:
        # Redis streams are automatically persisted
        pass

    async def stop(self) -> None:
        if self.redis:
            await self.redis.aclose()

    async def close(self) -> None:
        """Alias for stop() to match the abstract interface."""
        await self.stop()
