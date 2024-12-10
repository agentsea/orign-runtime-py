# base_aio.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, AsyncGenerator, Union, Type
import traceback
import asyncio
import json

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from orign.models import ErrorResponse, ChatRequest, ChatResponse, TokenResponse, OCRRequest, OCRResponse, EmbeddingRequest, EmbeddingResponse, CompletionRequest, CompletionResponse

from ..config import BaseConfig
from ..queue.base import AsyncMessageProducer, AsyncMessageConsumer
from ..queue.factory import get_message_consumer_async, get_message_producer_async

# Input/Output
I = TypeVar("I", bound=BaseModel)
O = TypeVar("O")

# Config
C = TypeVar("C", bound=BaseSettings)


class Processor(ABC, Generic[I, O, C]):
    """A stream processor"""

    def __init__(self):
        self.base_config: BaseConfig = BaseConfig()
        self.config: C = None
        self.producer: AsyncMessageProducer = None
        self.consumer: AsyncMessageConsumer = None
        self.semaphore = asyncio.Semaphore(BaseConfig.MAX_CONCURRENT_TASKS)

    @abstractmethod
    def load(self, config: C) -> None:
        """Before starting the processor"""
        pass

    @abstractmethod
    async def process(self, msg: I) -> AsyncGenerator[O, None]:
        """Process a single message from the consumer."""
        pass
    
    @abstractmethod
    def accepts(self) -> Type[I]:
        """The type accepted by the processor."""
        pass

    async def _sem_process_message(self, msg: I):
        """Wrapper to limit concurrency using a semaphore."""
        async with self.semaphore:
            try:
                async for response in self.process(msg):
                    if response:
                        # Retrieve the output topic from the message or set a default
                        topic = getattr(msg, 'output_topic', 'default_topic')
                        try:
                            await self.producer.produce(value=response, topic=topic)
                        except Exception as e:
                            print(f"Error producing message: {e}", flush=True)

            except Exception as e:
                print(f"Error processing message {msg}: {e}", flush=True)
                error_trace = traceback.format_exc()
                # Produce an error response if process fails
                error_response = ErrorResponse(
                    type="ErrorResponse",
                    request_id=getattr(msg, 'request_id', ''),
                    error=str(e),
                    traceback=error_trace,
                )
                topic = getattr(msg, 'output_topic', 'default_topic')
                await self.producer.produce(value=error_response, topic=topic)

    async def run(self, config: C) -> None:
        """Main loop for processing messages."""

        print("Starting main()", flush=True)
        self.load(config)
        print("Load completed", flush=True)

        self.consumer = get_message_consumer_async(self.base_config)
        self.producer = get_message_producer_async(self.base_config)
        print("Initialized Consumer and Producer", flush=True)

        await self.consumer.start()
        await self.producer.start()
        print("Started Consumer and Producer", flush=True)

        # Get the schema accepted by the backend
        schema = self.accepts()
        print(f"Accepted schema: {schema}", flush=True)

        try:
            while True:
                messages = await self.consumer.get_messages()
                if not messages:
                    continue
                print(f"Received {len(messages)} messages", flush=True)

                tasks = []
                # Process messages per partition
                for tp, msgs in messages.items():
                    for msg in msgs:
                        print(f"Processing message {msg}")
                        try:
                            # Validate the incoming message
                            message = schema.model_validate_json(msg["value"])
                            # Create a task to process the message with semaphore limit
                            task = asyncio.create_task(
                                self._sem_process_message(msg=message)
                            )
                            tasks.append(task)
                        except Exception as e:
                            error_trace = traceback.format_exc()
                            print(
                                f"Validation error for message {msg}: {e}\n{error_trace}"
                            )
                            request_id = ""
                            topic = "dead_letter"
                            try:
                                msg_dict = json.loads(msg["value"])
                                request_id = msg_dict.get("request_id", "")
                                topic = msg_dict.get("output_topic", "dead_letter")
                            except Exception as e:
                                print(f"Error getting request_id: {e}", flush=True)
                                continue
                            error_response = ErrorResponse(
                                type="ErrorResponse",
                                request_id=request_id,
                                error=f"Validation error: {e}",
                                traceback=error_trace,
                            )
                            await self.producer.produce(value=error_response, topic=topic)

                if tasks:
                    # Run all tasks concurrently within limit
                    await asyncio.gather(*tasks)

                # Commit offsets after processing
                await self.consumer.commit()
                print("Committed messages")

        except KeyboardInterrupt:
            print("Processing interrupted by user")

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error in main(): {e}\n{error_trace}")

        finally:
            print("Closing consumer and producer")
            await self.consumer.stop()
            await self.producer.flush()
            await self.producer.stop()


# ===== Chat Models =====
ChatResponses = Union[ChatResponse, TokenResponse, ErrorResponse]

class ChatModel(Processor[ChatRequest, ChatResponses, C], Generic[C]):
    def accepts(self) -> Type[ChatRequest]:
        return ChatRequest

# ===== Completion Models =====
CompletionResponses = Union[CompletionResponse, TokenResponse, ErrorResponse]

class CompletionModel(Processor[CompletionRequest, CompletionResponses, C], Generic[C]):
    def accepts(self) -> Type[CompletionRequest]:
        return CompletionRequest

# ===== OCR Models =====
OCRResponses = Union[OCRResponse, ErrorResponse]

class OCRModel(Processor[OCRRequest, OCRResponses, C], Generic[C]):
    def accepts(self) -> Type[OCRRequest]:
        return OCRRequest

# ===== Embedding Models =====
EmbeddingResponses = Union[EmbeddingResponse, ErrorResponse]

class EmbeddingModel(Processor[EmbeddingRequest, EmbeddingResponses, C], Generic[C]):
    def accepts(self) -> Type[EmbeddingRequest]:
        return EmbeddingRequest
