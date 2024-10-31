# base_aio.py
from typing import TypeVar, List, Type
import asyncio

import easyocr
from pydantic import BaseModel
from orign.server.queue.base import AsyncMessageConsumer, AsyncMessageProducer
from orign.server.config import Config
from orign.server.backends.base_aio import ModelBackend
from orign.server.models import ErrorResponse, OCRRequest, OCRResponse

S = TypeVar("S", bound=BaseModel)


class EasyOCRBackend(ModelBackend[OCRRequest]):
    def __init__(self):
        super().__init__()
        self.config: Config = Config()
        self.producer: AsyncMessageProducer = None
        self.consumer: AsyncMessageConsumer = None

    def initialize_engine(self) -> None:
        """Initialize the language model engine."""
        self.reader = easyocr.Reader(["en"])

    async def process_message(self, msg: OCRRequest) -> None:
        """Process a single message from the consumer."""
        pass

    def accepts(self) -> Type[OCRRequest]:
        """The schema accepted by the backend."""
        return OCRRequest

    def produces(self) -> List[Type[BaseModel]]:
        """The schemas produced by the backend."""
        return [OCRResponse, ErrorResponse]
