# base_aio.py
from abc import ABC, abstractmethod
from typing import TypeVar, List, Type, Generic
import traceback
import asyncio

from pydantic import BaseModel

from ..config import Config
from ..models import ErrorResponse
from ..queue.base import AsyncMessageProducer, AsyncMessageConsumer
from ..queue.factory import get_message_consumer_async, get_message_producer_async

S = TypeVar("S", bound=BaseModel)


class ModelBackend(ABC, Generic[S]):
    def __init__(self):
        self.config = None
        self.engine = None
        self.producer: AsyncMessageProducer = None
        self.consumer: AsyncMessageConsumer = None
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_TASKS)

    @abstractmethod
    def initialize_engine(self) -> None:
        """Initialize the language model engine."""
        pass

    @abstractmethod
    async def process_message(self, msg: S) -> None:
        """Process a single message from the consumer."""
        pass

    @abstractmethod
    def accepts(self) -> Type[S]:
        """The schema accepted by the backend."""
        pass

    @abstractmethod
    def produces(self) -> List[Type[BaseModel]]:
        """The schemas produced by the backend."""
        pass

    async def _sem_process_message(self, msg: S):
        """Wrapper to limit concurrency using a semaphore."""
        async with self.semaphore:
            await self.process_message(msg)

    async def main(self) -> None:
        """Main loop for processing messages."""
        print("Starting main()")
        self.initialize_engine()
        print("Initialized Engine")
        
        self.consumer = get_message_consumer_async(self.config)
        self.producer = get_message_producer_async(self.config)

        await self.consumer.start()
        await self.producer.start()

        # Get the schema accepted by the backend
        schema = self.accepts()

        try:
            while True:
                messages = await self.consumer.get_messages()
                if not messages:
                    continue

                tasks = []
                # Process messages per partition
                for tp, msgs in messages.items():
                    for msg in msgs:
                        print(f"Processing message {msg}")
                        try:
                            # Validate the incoming message
                            message = schema.parse_raw(msg['value'])
                        except Exception as e:
                            error_trace = traceback.format_exc()
                            print(f"Validation error for message {message}: {e}\n{error_trace}")
                            error_response = ErrorResponse(
                                error=f"Validation error: {e}",
                                request_id="",
                                traceback=error_trace
                            )
                            tasks.append(asyncio.create_task(self.producer.produce(error_response)))
                        else:
                            # Create a task to process the message with semaphore limit
                            task = asyncio.create_task(self._sem_process_message(msg=message))
                            tasks.append(task)
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
