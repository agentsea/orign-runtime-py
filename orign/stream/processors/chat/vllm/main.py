# main.py
from typing import AsyncGenerator
import traceback
import asyncio

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams as VLLMSamplingParams
from pydantic_settings import BaseSettings

from orign.stream.util import open_image_from_input_async
from orign.stream.processors.base_aio import ChatModel, ChatResponses
from orign.stream.models import (
    ChatRequest,
    ContentItem,
    ChatResponse,
    TokenResponse,
    ErrorResponse,
    Choice,
)

class vLLMConfig(BaseSettings):
    model_name: str
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    torch_dtype: str = "auto"
    max_images_per_prompt: int = 1
    device: str = "cuda"


class vLLM(ChatModel[vLLMConfig]):
    """vLLM backend"""

    def load(self, config: vLLMConfig):
        self.config = config

        engine_args = AsyncEngineArgs(
            model=config.model_name,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.torch_dtype,
            device=config.device,
        )
        if config.max_images_per_prompt != 1:
            engine_args.limit_mm_per_prompt = {
                "image": config.max_images_per_prompt
            }

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Initialized AsyncLLMEngine", flush=True)

    async def process(self, msg: ChatRequest) -> AsyncGenerator[ChatResponses, None]:
        """Process a single message using the vLLM engine."""

        print(f"Processing message for request_id {msg.request_id}", flush=True)
        if not msg.request_id:
            raise ValueError("No request_id found in message")

        # Prepare the prompts and multimodal data
        prompts = []

        batch_items = msg.batch if msg.batch is not None else [msg.prompt]
        for idx, prompt_item in enumerate(batch_items):
            if prompt_item is None:
                print(f"No prompt found in message item {idx}")
                continue

            prompt_text = ""
            images = []

            # Handle messages within the prompt
            for msg_entry in prompt_item.messages:
                if isinstance(msg_entry.content, str):
                    prompt_text += msg_entry.content + "\n"
                elif isinstance(msg_entry.content, list):
                    for content_item in msg_entry.content:
                        if content_item.type == "text" and content_item.text:
                            prompt_text += content_item.text + "\n"
                        elif (
                            content_item.type == "image_url" and content_item.image_url
                        ):
                            try:
                                image_url = content_item.image_url.url
                                image = await open_image_from_input_async(image_url)
                                print(f"Downloaded image from URL '{image_url}'")
                                images.append(image)
                            except Exception as e:
                                print(f"Failed to load image: {e}")
                                continue
                        else:
                            print(f"Unknown content item type: {content_item.type}")
                elif isinstance(msg_entry.content, ContentItem):
                    content_item = msg_entry.content
                    if content_item.type == "text" and content_item.text:
                        prompt_text += content_item.text + "\n"
                    elif content_item.type == "image_url" and content_item.image_url:
                        try:
                            image_url = content_item.image_url.url
                            image = await open_image_from_input_async(image_url)
                            print(f"Downloaded image from URL '{image_url}'")
                            images.append(image)
                        except Exception as e:
                            print(f"Failed to load image: {e}")
                            continue
                    elif content_item.type == "image_base64" and content_item.data:
                        try:
                            base64_data = content_item.data
                            image = await open_image_from_input_async(base64_data)
                            print("Decoded base64 image")
                            images.append(image)
                        except Exception as e:
                            print(f"Failed to load image: {e}")
                            continue
                    else:
                        print(f"Unknown content item type: {content_item.type}")
                else:
                    print(
                        f"Unexpected content type in message: {type(msg_entry.content)}"
                    )

            if not prompt_text.strip():
                print(f"No valid content found in message item {idx}")
                continue

            # Prepare multi_modal_data with the 'image' key
            multi_modal_data = {}
            if images:
                multi_modal_data["image"] = images if len(images) > 1 else images[0]

                # Check if the number of images exceeds the limit
                max_images = self.config.max_images_per_prompt
                if isinstance(images, list) and len(images) > max_images:
                    error_message = f"Number of images ({len(images)}) exceeds the maximum allowed ({max_images})."
                    print(error_message)
                    error_response = ErrorResponse(
                        request_id=msg.request_id, error=error_message
                    )
                    yield error_response
                    return

            # Add the prompt and multi_modal_data to the list
            prompt_entry = {"prompt": prompt_text}
            if multi_modal_data:
                prompt_entry["multi_modal_data"] = multi_modal_data

            prompts.append(prompt_entry)

        if not prompts:
            print(f"No valid prompts to process for request_id {msg.request_id}")
            return

        # Prepare the sampling parameters
        sampling_params_dict = msg.sampling_params.model_dump(exclude_none=True)
        sampling_params_dict.setdefault("max_tokens", msg.max_tokens)
        vllm_sampling_params = VLLMSamplingParams(**sampling_params_dict)

        for prompt in prompts:
            try:
                # Use 'async for' to iterate over the async generator
                async for response in self.process_single_prompt(
                    prompt,
                    vllm_sampling_params,
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
        prompt: dict,
        sampling_params: VLLMSamplingParams,
        request_id: str,
        stream: bool,
    ):
        """Process a single prompt and handle streaming or non-streaming output."""

        print(f"Processing prompt for request_id {request_id}")
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        if stream:
            # Streaming response
            accumulated_choices = {}  # To keep track of the state per choice
            print(f"Streaming output for request_id {request_id}")
            async for request_output in generator:
                # Collect choices for all outputs
                for output in request_output.outputs:
                    output_index = output.index

                    # Initialize the accumulated data for this choice if not already done
                    if output_index not in accumulated_choices:
                        accumulated_choices[output_index] = {
                            "text": "",
                            "tokens": [],
                            "token_ids": [],
                            "logprobs": [],
                            "last_token_index": 0,
                        }

                    choice_data = accumulated_choices[output_index]

                    # Calculate new content since last update
                    new_text = output.text[len(choice_data["text"]) :]
                    choice_data["text"] = output.text  # Update accumulated text

                    # Calculate new tokens
                    new_tokens = []
                    if hasattr(output, "tokens") and output.tokens is not None:
                        new_tokens = output.tokens[choice_data["last_token_index"] :]
                        choice_data["tokens"].extend(new_tokens)

                    # Calculate new token_ids
                    new_token_ids = []
                    if hasattr(output, "token_ids") and output.token_ids is not None:
                        new_token_ids = output.token_ids[
                            choice_data["last_token_index"] :
                        ]
                        choice_data["token_ids"].extend(new_token_ids)

                    # Calculate new logprobs
                    new_logprobs = []
                    if hasattr(output, "logprobs") and output.logprobs is not None:
                        new_logprobs = output.logprobs[
                            choice_data["last_token_index"] :
                        ]
                        choice_data["logprobs"].extend(new_logprobs)

                    # Update last_token_index
                    choice_data["last_token_index"] += len(new_tokens)

                    # Construct the Choice object
                    choice = Choice(
                        index=output_index,
                        text=new_text,
                        tokens=new_tokens,
                        token_ids=new_token_ids,
                        logprobs=new_logprobs,
                        finish_reason=output.finish_reason
                        if hasattr(output, "finish_reason")
                        else None,
                    )

                    # Send the incremental update
                    token_response = TokenResponse(
                        type="TokenResponse",
                        request_id=request_id,
                        choices=[choice],
                    )
                    yield token_response

            print(f"Completed streaming response for request_id {request_id}")
        else:
            # Non-streaming response
            accumulated_choices = {}
            async for request_output in generator:
                for output in request_output.outputs:
                    output_index = output.index

                    # Initialize the accumulated data for this choice if not already done
                    if output_index not in accumulated_choices:
                        accumulated_choices[output_index] = {
                            "text": "",
                            "tokens": [],
                            "token_ids": [],
                            "logprobs": [],
                            "finish_reason": None,
                        }

                    choice_data = accumulated_choices[output_index]

                    # Accumulate the text
                    choice_data["text"] = output.text

                    # Accumulate tokens and token IDs if available
                    if hasattr(output, "tokens") and output.tokens is not None:
                        choice_data["tokens"] = output.tokens
                    if hasattr(output, "token_ids") and output.token_ids is not None:
                        choice_data["token_ids"] = output.token_ids

                    # Accumulate logprobs if available
                    if hasattr(output, "logprobs") and output.logprobs is not None:
                        choice_data["logprobs"] = output.logprobs

                    # Update finish reason
                    choice_data["finish_reason"] = output.finish_reason

            # After generation is complete, construct the list of choices
            choices = []
            for idx, choice_data in accumulated_choices.items():
                choice = Choice(
                    index=idx,
                    text=choice_data["text"],
                    tokens=choice_data["tokens"],
                    token_ids=choice_data["token_ids"],
                    logprobs=choice_data["logprobs"],
                    finish_reason=choice_data["finish_reason"],
                )
                choices.append(choice)

            # Create the final ChatResponse
            response = ChatResponse(
                type="ChatResponse",
                request_id=request_id,
                choices=choices,
                trip_time=None,
            )

            # Send the final response
            yield response
            print(f"Sent final response for request_id {request_id}")



if __name__ == "__main__":
    import asyncio

    backend = vLLM()
    config = vLLMConfig()
    asyncio.run(backend.run(config))