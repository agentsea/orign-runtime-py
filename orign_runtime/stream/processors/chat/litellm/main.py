# main.py
from typing import AsyncGenerator, Dict, Optional, List
import asyncio

from pydantic_settings import BaseSettings
from litellm import Router

from orign_runtime.stream.util import open_image_from_input_async
from orign_runtime.stream.processors.base_aio import ChatModel, ChatResponses
from orign.models import (
    ChatRequest,
    ChatResponse,
    TokenResponse,
    ErrorResponse,
    Choice,
)

class LiteLLMConfig(BaseSettings):
    api_keys: Dict[str, str]
    preference: Optional[List[str]] = None


class LiteLLM(ChatModel[LiteLLMConfig]):
    """LiteLLM backend"""

    def load(self, config: LiteLLMConfig):
        self.config = config
        
        model_list = []
        # Convert config.api_keys into model_list format
        for model_name, api_key in config.api_keys.items():
            model_list.append({
                "model_name": model_name,
                "litellm_params": {
                    "model": model_name,
                    "api_key": api_key,
                }
            })

        # Convert preference list into fallbacks format
        fallbacks = None
        if config.preference:
            fallbacks = []
            for i in range(len(config.preference) - 1):
                fallbacks.append({
                    config.preference[i]: [config.preference[i + 1]]
                })

        self.router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            enable_pre_call_checks=True
        )
        
        print("Initialized AsyncLLMEngine", flush=True)

    async def process(self, msg: ChatRequest) -> AsyncGenerator[ChatResponses, None]:
        """Process a single message using the LiteLLM engine."""
        try:
            # Handle batch requests
            batch_items = msg.batch if msg.batch is not None else [msg.prompt]
            
            for prompt_item in batch_items:
                if prompt_item is None:
                    continue
                    
                # Convert messages format if needed
                messages = prompt_item.messages if prompt_item else []
                
                response = await self.router.acompletion(
                    model=msg.model or list(self.config.api_keys.keys())[0],
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    temperature=msg.sampling_params.temperature,
                    max_tokens=msg.max_tokens,
                    n=msg.sampling_params.n,
                    stream=msg.stream
                )

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

        except Exception as e:
            yield ErrorResponse(
                request_id=msg.request_id,
                error=str(e)
            )


if __name__ == "__main__":
    import asyncio

    backend = LiteLLM()
    config = LiteLLMConfig()
    asyncio.run(backend.run(config))
