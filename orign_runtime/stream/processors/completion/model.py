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


class HFCompletionModel(ABC):

    @abstractmethod
    def load(self, config: HFConfig):
        pass

    @abstractmethod
    async def complete(self, prompt: str, images: List[Image.Image], sampling_params: SamplingParams, stream: bool = False, max_tokens: int = 200) -> AsyncGenerator[CompletionResponses, None]:
        pass


class PaliGemmaModel(HFCompletionModel):

    def load(self, config: HFConfig):
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(config.model_id)
        self.processor = PaliGemmaProcessor.from_pretrained(config.model_id)

    async def complete(self, prompt: str, images: List[Image.Image], sampling_params: SamplingParams, stream: bool = False, max_tokens: int = 200) -> AsyncGenerator[CompletionResponses, None]:

        if stream:
            raise NotImplementedError("Streaming is not supported for PaliGemma")

        if len(loaded_images) == 1:
            loaded_images = loaded_images[0]

        if len(loaded_images) == 0:
            raise ValueError("No images found in the prompt")

        inputs = self.processor(images=[loaded_images], text=prompt, return_tensors="pt")

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_return_sequences=sampling_params.n,
            do_sample=True,
        )
        result = self.processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ]
        print(result)


class MoondreamModel(HFCompletionModel):

    def load(self, config: HFConfig):
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, trust_remote_code=True, revision=config.revision
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=revision)

    async def complete(self, prompt: str, images: List[Image.Image], sampling_params: SamplingParams, stream: bool = False, max_tokens: int = 200) -> AsyncGenerator[CompletionResponses, None]:

        if stream:
            raise NotImplementedError("Streaming is not supported for PaliGemma")

        if len(images) > 1:
            raise ValueError("Moondream only supports one image")

        enc_image = self.model.encode_image(images[0])
        print(self.model.answer_question(enc_image, prompt, self.tokenizer))