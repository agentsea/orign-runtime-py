from typing import Optional, Callable, Any, List, Dict
from pydantic import BaseModel
import redis.asyncio as redis
import time
import traceback

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
        print("Starting AsyncRedisMessageConsumer...")
        self.redis = redis.from_url(self.config.REDIS_URL, decode_responses=True)

        # Create consumer group for each input topic
        for topic in self.config.INPUT_TOPICS:
            try:
                print(
                    f"Creating consumer group '{self.consumer_group}' for topic '{topic}'"
                )
                await self.redis.xgroup_create(
                    topic,
                    self.consumer_group,
                    mkstream=True,
                    id="0",  # Start from beginning
                )
                print(
                    f"Consumer group '{self.consumer_group}' created for topic '{topic}'"
                )
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):  # Ignore if group already exists
                    print(
                        f"Consumer group '{self.consumer_group}' already exists for topic '{topic}'"
                    )
                else:
                    print(f"Error creating consumer group for topic '{topic}': {e}")
                    raise

        print(
            f"Initialized AsyncRedisMessageConsumer for group: {self.consumer_group}",
            flush=True,
        )
        print(f"Watching topics: {', '.join(self.config.INPUT_TOPICS)}", flush=True)

    async def get_messages(
        self, timeout: float = 1.0
    ) -> Optional[Dict[str, List[Any]]]:
        try:
            # First, try to read pending messages
            print(
                f"Attempting to read pending messages for topics: {self.config.INPUT_TOPICS}"
            )
            streams = {topic: "0" for topic in self.config.INPUT_TOPICS}
            messages = await self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams=streams,
                count=100,
                block=0,  # Non-blocking call
            )
            print(f"Pending messages received: {messages}")

            if not messages:
                # No pending messages, read new messages
                print("No pending messages found. Attempting to read new messages.")
                streams = {topic: ">" for topic in self.config.INPUT_TOPICS}
                messages = await self.redis.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams=streams,
                    count=100,
                    block=int(timeout * 1000),
                )
                print(f"New messages received: {messages}")

            if not messages:
                print(
                    "No messages received after attempting to read pending and new messages."
                )
                return None

            # Format messages similar to Kafka structure
            formatted_messages = {}
            for topic, msgs in messages:
                print(f"Processing messages for topic: {topic}")
                if topic not in self.pending_messages:
                    self.pending_messages[topic] = []

                message_list = []
                for msg_id, msg_data in msgs:
                    print(f"Received message ID: {msg_id} with data: {msg_data}")
                    self.last_ids[topic] = msg_id
                    message = {
                        "topic": topic,
                        "offset": msg_id,
                        "value": msg_data.get("payload", ""),
                    }
                    message_list.append(message)
                    self.pending_messages[topic].extend(message_list)
                    print(f"Added message to pending_messages[{topic}]")

                formatted_messages[topic] = message_list
                print(f"Formatted messages for topic '{topic}': {message_list}")

            print(f"Returning formatted messages: {formatted_messages}")
            return formatted_messages
        except Exception as e:
            print(f"Error getting messages: {e}")
            traceback.print_exc()
            return None

    async def commit(self) -> None:
        try:
            print(f"Committing messages for topics: {list(self.last_ids.keys())}")
            # Acknowledge messages for each topic
            for topic, msg_id in self.last_ids.items():
                print(f"Acknowledging message ID '{msg_id}' for topic '{topic}'")
                await self.redis.xack(topic, self.consumer_group, msg_id)
            self.pending_messages.clear()
            self.last_ids.clear()
            print("Commit successful, cleared pending messages and last_ids.")
        except Exception as e:
            print(f"Error during commit: {e}")
            traceback.print_exc()

    async def stop(self) -> None:
        print("Stopping AsyncRedisMessageConsumer...")
        if self.redis:
            await self.redis.aclose()
            print("Closed Redis connection.")

    async def commit_on_revoke(self, revoked_partitions: List[Any]) -> None:
        """Redis streams don't use partitions, so we just commit pending messages."""
        print("Commit on revoke called.")
        await self.commit()

    async def close(self) -> None:
        """Alias for stop() to match the abstract interface."""
        print("Closing AsyncRedisMessageConsumer...")
        await self.stop()


class AsyncRedisMessageProducer(AsyncMessageProducer):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.redis: Optional[redis.Redis] = None

    async def start(self) -> None:
        self.redis = redis.from_url(self.config.REDIS_URL, decode_responses=True)
        print("Initialized AsyncRedisMessageProducer", flush=True)

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
