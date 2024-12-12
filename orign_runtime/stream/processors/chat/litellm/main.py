# main.py
from typing import AsyncGenerator, Dict, Optional, List
import asyncio

from pydantic_settings import BaseSettings
import litellm

from orign_runtime.stream.util import open_image_from_input_async
from orign_runtime.stream.processors.base_aio import ChatModel, ChatResponses
from orign.models import (
    ChatRequest,
    ChatResponse,
    TokenResponse,
    ErrorResponse,
    Choice,
)

litellm.drop_params = True

class LiteLLMConfig(BaseSettings):
    pass


class LiteLLM(ChatModel[LiteLLMConfig]):
    """LiteLLM backend"""

    def load(self, config: LiteLLMConfig):
        print("Initialized AsyncLLMEngine", flush=True)

    async def process(self, msg: ChatRequest) -> AsyncGenerator[ChatResponses, None]:
        """Process a single message using the LiteLLM engine."""
        # Handle batch requests
        batch_items = msg.batch if msg.batch is not None else [msg.prompt]
        
        for prompt_item in batch_items:
            if prompt_item is None:
                continue
                
            # Convert messages format if needed
            messages = prompt_item.messages if prompt_item else []
            messages_fmt = [message.model_dump(exclude_none=True) for message in messages]

            if not msg.model:
                raise ValueError("Model is required")

            print(f"Using model {msg.model}", flush=True)

            response = await litellm.acompletion(
                model=msg.model,
                messages=messages_fmt,
                temperature=msg.sampling_params.temperature,
                max_tokens=msg.max_tokens,
                n=msg.sampling_params.n,
                stream=msg.stream
            )
            print(f"Response: {response}", flush=True)

            if msg.stream:
                async for chunk in response:
                    yield TokenResponse(
                        request_id=msg.request_id,
                        choices=[Choice(
                            index=0,
                            text=chunk.choices[0].delta.content or "",
                            finish_reason=chunk.choices[0].finish_reason
                        )]
                    )
            else:
                yield ChatResponse(
                    request_id=msg.request_id,
                    choices=[Choice(
                        index=choice.index,
                        text=choice.message.content,
                        finish_reason=choice.finish_reason
                    ) for choice in response.choices]
                )


if __name__ == "__main__":
    import asyncio

    backend = LiteLLM()
    config = LiteLLMConfig()
    asyncio.run(backend.run(config))
