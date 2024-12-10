from abc import ABC, abstractmethod
from typing import NamedTuple, List

from orign.models import Prompt, ContentItem
from transformers import AutoTokenizer
from PIL import Image
from pydantic_settings import BaseSettings

from orign_runtime.stream.util import open_image_from_input_async
from orign_runtime.stream.processors.base_aio import ChatModel, ChatResponses, CompletionModel, CompletionResponses
from orign.models import (
    ChatRequest,
    CompletionRequest,
    ContentItem,
    ChatResponse,
    CompletionResponse,
    TokenResponse,
    ErrorResponse,
    SamplingParams,
    Choice,
)

class HFConfig(BaseSettings):
    model_id: str
    trust_remote_code: bool = True
    dtype: str = "auto"
    device: str = "cuda"
    model_protocol: str = "chat"
    revision: Optional[str] = None


class HFChatModel(ABC):
    
    @abstractmethod
    def load(self, config: HFConfig):
        pass

    @abstractmethod
    async def chat(self, prompt: Prompt, sampling_params: SamplingParams, stream: bool = False, max_tokens: int = 200) -> AsyncGenerator[ChatResponses, None]:
        pass