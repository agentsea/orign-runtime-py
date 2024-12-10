# main.py
from typing import Optional, List, AsyncGenerator
import traceback
import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic_settings import BaseSettings

from orign_runtime.stream.util import open_image_from_input_async
from orign_runtime.stream.processors.base_aio import ChatModel, ChatResponses
from orign.models import (
    ChatRequest,
    ContentItem,
    ChatResponse,
    TokenResponse,
    ErrorResponse,
    Choice,
)
from .model import HFModel, HFConfig


class HFChat(ChatModel[HFConfig]):
    """HF chat backend"""

    def load(self, config: HFConfig):
        self.config = config

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=config.dtype,
            device_map=config.device,
        )
        self.model.eval()

        # Initialize the message formatter using DefaultMessageFormatter
        self.formatter = DefaultMessageFormatter(self.tokenizer)

        print("Initialized Hugging Face model and tokenizer", flush=True)

    async def process(self, msg: ChatRequest) -> AsyncGenerator[ChatResponses, None]:
        """Process a single message using the Hugging Face model."""

        print(f"Processing message for request_id {msg.request_id}", flush=True)
        if not msg.request_id:
            raise ValueError("No request_id found in message")

        # Prepare the prompts
        prompts = []

        batch_items = msg.batch if msg.batch is not None else [msg.prompt]
        for idx, prompt_item in enumerate(batch_items):
            if prompt_item is None:
                print(f"No prompt found in message item {idx}")
                continue

            model_request_data = await self.formatter.format(prompt_item)
            prompt_text = model_request_data.prompt

            if not prompt_text.strip():
                print(f"No valid content found in message item {idx}")
                continue

            # Add the prompt to the list
            prompts.append(prompt_text)

        if not prompts:
            print(f"No valid prompts to process for request_id {msg.request_id}")
            return

        # Prepare the generation arguments
        generation_args = msg.sampling_params.model_dump(exclude_none=True)
        generation_args.setdefault("max_new_tokens", msg.max_tokens)
        generation_args.setdefault("do_sample", True)
        generation_args.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        for prompt in prompts:
            try:
                async for response in self.process_single_prompt(
                    prompt,
                    generation_args,
                    msg.request_id,
                    msg.stream,
                ):
                    yield response
            except Exception as e:
                error_trace = traceback.format_exc()
                print(
                    f"Error during generation for request_id {msg.request_id}: {e}\n{error_trace}"
                )
                error_response = ErrorResponse(
                    request_id=msg.request_id, error=str(e), traceback=error_trace
                )
                yield error_response

    async def process_single_prompt(
        self,
        prompt: str,
        generation_args: dict,
        request_id: str,
        stream: bool,
    ):
        """Process a single prompt and handle streaming or non-streaming output."""

        print(f"Processing prompt for request_id {request_id}")

        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)

        # Generate outputs
        if stream:
            # Streaming response (simulate streaming for demonstration)
            print(f"Streaming output for request_id {request_id}")
            # Note: Hugging Face models do not natively support streaming. You might need to implement
            # a custom generator that generates token by token.
            output_ids = await self.generate_with_streaming(input_ids, generation_args)
        else:
            # Non-streaming response
            output = self.model.generate(input_ids, **generation_args)
            output_ids = output[0]

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Create the Choice object
        choice = Choice(
            index=0,
            text=generated_text[len(prompt):],  # Remove the prompt from the generated text
            finish_reason="stop",
        )

        # Create the final ChatResponse
        response = ChatResponse(
            type="ChatResponse",
            request_id=request_id,
            choices=[choice],
            trip_time=None,
        )

        # Send the final response
        yield response
        print(f"Sent final response for request_id {request_id}")

    async def generate_with_streaming(self, input_ids, generation_args):
        """Simulate streaming by generating tokens one at a time."""
        # Note: This is a simplified example. For actual streaming, you might need to integrate with libraries
        # that support token-level streaming generation, such as transformers' generate with `stopping_criteria`.
        output_ids = input_ids
        for _ in range(generation_args.get("max_new_tokens", 20)):
            outputs = self.model(input_ids=output_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            generated_text = self.tokenizer.decode(
                next_token_id[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # Yield the token response
            choice = Choice(
                index=0,
                text=generated_text,
                finish_reason=None,
            )
            token_response = TokenResponse(
                type="TokenResponse",
                request_id=request_id,
                choices=[choice],
            )
            yield token_response
            # Check for stop conditions
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            await asyncio.sleep(0)  # Allow other tasks to run

if __name__ == "__main__":
    import asyncio

    backend = HF()
    config = HFConfig()
    asyncio.run(backend.run(config))