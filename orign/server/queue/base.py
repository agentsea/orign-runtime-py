# main.py
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, List, Dict
from pydantic import BaseModel

class MessageConsumer(ABC):
    @abstractmethod
    def poll(self, timeout: float = 1.0) -> Optional[Any]:
        pass

    @abstractmethod
    def commit(self, message: Any) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

class MessageProducer(ABC):
    @abstractmethod
    def produce(
        self,
        value: BaseModel,
        callback: Optional[Callable[[Any, Any], None]] = None,
        topic: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

class Queue(ABC):
    @abstractmethod
    def get_message_consumer(self) -> MessageConsumer:
        pass

    @abstractmethod
    def get_message_producer(self) -> MessageProducer:
        pass


class AsyncMessageConsumer(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Starts the consumer."""
        pass

    @abstractmethod
    async def get_messages(self, timeout: float = 1.0) -> Optional[Dict[Any, List[Any]]]:
        """Retrieves messages asynchronously."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commits the current offsets."""
        pass

    @abstractmethod
    async def commit_on_revoke(self, revoked_partitions: List[Any]) -> None:
        """Commits offsets when partitions are revoked."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Closes the consumer."""
        pass

class AsyncMessageProducer(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Starts the producer."""
        pass

    @abstractmethod
    async def produce(
        self,
        value: BaseModel,
        topic: str,
        callback: Optional[Callable[[Any, Optional[Exception]], None]] = None,
        partition: Optional[int] = None
    ) -> None:
        """Produces a message asynchronously."""
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Flushes pending messages."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Closes the producer."""
        pass

class AsyncQueue(ABC):
    @abstractmethod
    async def get_message_consumer(self) -> AsyncMessageConsumer:
        """Returns an asynchronous message consumer."""
        pass

    @abstractmethod
    async def get_message_producer(self) -> AsyncMessageProducer:
        """Returns an asynchronous message producer."""
        pass