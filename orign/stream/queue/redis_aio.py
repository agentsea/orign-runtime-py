# orign/server/queue/redis_aio.py
from typing import Optional, Callable, Any, List, Dict
from pydantic import BaseModel
import redis.asyncio as redis
import time
import traceback

from ..config import BaseConfig
from .base import AsyncMessageConsumer, AsyncMessageProducer


class AsyncRedisMessageConsumer(AsyncMessageConsumer):
    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.consumer_group = config.GROUP_ID
        self.consumer_name = f"{config.GROUP_ID}-{int(time.time())}"
        self.pending_messages: Dict[str, List[Any]] = {}
        self.pending_message_ids: Dict[str, List[str]] = {}

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
            messages = []

            # Step 1: Check for pending messages
            for topic in self.config.INPUT_TOPICS:
                pending_info = await self.redis.xpending_range(
                    name=topic,
                    groupname=self.consumer_group,
                    min="-",
                    max="+",
                    count=100,
                )

                if pending_info:
                    msg_ids = [entry["message_id"] for entry in pending_info]
                    print(f"Found pending messages for topic '{topic}': {msg_ids}")

                    # Claim the pending messages
                    claimed_msgs = await self.redis.xclaim(
                        name=topic,
                        groupname=self.consumer_group,
                        consumername=self.consumer_name,
                        min_idle_time=0,
                        message_ids=msg_ids,
                    )
                    if claimed_msgs:
                        messages.append((topic, claimed_msgs))
                else:
                    print(f"No pending messages for topic '{topic}'")

            # Step 2: If no pending messages, read new messages
            if not messages:
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
                    self.pending_message_ids[topic] = []

                message_list = []
                for msg_id, msg_data in msgs:
                    print(f"Received message ID: {msg_id} with data: {msg_data}")
                    message = {
                        "topic": topic,
                        "offset": msg_id,
                        "value": msg_data.get("message", ""),
                    }
                    message_list.append(message)
                    self.pending_messages[topic].append(message)
                    self.pending_message_ids[topic].append(msg_id)
                    print(f"Added message ID {msg_id} to pending_message_ids[{topic}]")

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
            print(f"Committing messages for topics: {list(self.pending_message_ids.keys())}")
            # Acknowledge messages for each topic
            for topic, msg_ids in self.pending_message_ids.items():
                if msg_ids:
                    print(f"Acknowledging message IDs '{msg_ids}' for topic '{topic}'")
                    await self.redis.xack(topic, self.consumer_group, *msg_ids)
            self.pending_messages.clear()
            self.pending_message_ids.clear()
            print("Commit successful, cleared pending messages and pending_message_ids.")
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
    def __init__(self, config: BaseConfig) -> None:
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

        if not isinstance(value, BaseModel):
            raise ValueError("Value must be a Pydantic model: ", value)

        try:
            serialized_value = value.model_dump_json().encode('utf-8')
        except Exception as e:
            raise ValueError(f"Error serializing value: {e} value: {value}")
        
        try:
            # Add message to stream
            msg_id = await self.redis.xadd(topic, {"message": serialized_value})
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
