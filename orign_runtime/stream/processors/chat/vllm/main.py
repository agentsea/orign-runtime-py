# main.py
import asyncio
import os
import traceback
from typing import AsyncGenerator, Optional

import yaml
from orign.models import (
    ChatRequest,
    ChatResponse,
    Choice,
    ErrorResponse,
    TokenResponse,
)
from pydantic_settings import BaseSettings
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm import SamplingParams as VLLMSamplingParams
from vllm.lora.request import LoRARequest

from orign_runtime.stream.processors.base_aio import ChatModel, ChatResponses
from orign_runtime.stream.processors.chat.vllm.fmt import (
    MODEL_FORMATTER_MAP,
    MODEL_TYPE_MAP,
)


class vLLMConfig(BaseSettings):
    model_name: str
    model_type: Optional[str] = None
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_images_per_prompt: int = 1
    device: str = "cuda"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    max_num_seqs: int = 5
    enforce_eager: bool = False


class vLLM(ChatModel[vLLMConfig]):
    """vLLM backend"""

    lora_id_mapping = {}
    next_adapter_id = 1
    lora_id_lock = asyncio.Lock()

    def load(self, config: vLLMConfig):
        self.config = config

        if not config.model_type:
            config.model_type = MODEL_TYPE_MAP.get(config.model_name, None)
            if not config.model_type:
                raise ValueError(
                    f"Uknown model type for {config.model_name}, consider setting model_type explicitly"
                )

        print(f"using config: {config}", flush=True)

        engine_args = AsyncEngineArgs(
            model=config.model_name,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            device=config.device,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            enforce_eager=config.enforce_eager,
            enable_lora=True,
            max_lora_rank=256,
            limit_mm_per_prompt={"image": 2},  # TODO: for debugging
        )
        if config.max_images_per_prompt != 1:
            engine_args.limit_mm_per_prompt = {"image": config.max_images_per_prompt}

        if config.model_type not in MODEL_FORMATTER_MAP:
            raise ValueError(f"Model {config.model_type} not supported")

        self.formatter = MODEL_FORMATTER_MAP[config.model_type]()

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Initialized AsyncLLMEngine", flush=True)

    async def process(self, msg: ChatRequest) -> AsyncGenerator[ChatResponses, None]:
        """Process a single message using the vLLM engine."""

        print(f"Processing message for request_id {msg.request_id}", flush=True)
        if not msg.request_id:
            raise ValueError("No request_id found in message")

        lora_request: Optional[LoRARequest] = None
        print(f"Checking for adapter in message: {msg.adapter}", flush=True)
        if msg.adapter:
            adapter_name = msg.adapter
            adapter_parts = msg.adapter.split("/")
            print(f"Adapter parts: {adapter_parts}", flush=True)

            org_slugs = []
            org_names = []
            org_map = {}
            if msg.organizations:
                for org_slug, org_info in msg.organizations.items():
                    org_slugs.append(org_slug)
                    org_names.append(org_info["org_name"])
                    org_map[org_info["org_name"]] = org_slug

            if len(adapter_parts) == 2:
                namespace = adapter_parts[0]
                name = adapter_parts[1]
                print(f"Found namespace '{namespace}' and name '{name}'", flush=True)
                print(
                    f"Checking authorization against org '{msg.organizations}' and handle '{msg.handle}'",
                    flush=True,
                )
                if namespace not in org_names and namespace != msg.handle:
                    raise ValueError(
                        f"Adapter {msg.adapter} is not authorized for this request"
                    )
                adapter_name = f"{org_map[namespace]}/{name}"

            elif len(adapter_parts) == 1:
                name = adapter_parts[0]
                namespace = msg.user_id
                print(
                    f"Single part adapter, using namespace '{namespace}' and name '{name}'",
                    flush=True,
                )
                adapter_name = f"{namespace}/{name}"
            else:
                print(
                    f"Invalid adapter format with {len(adapter_parts)} parts",
                    flush=True,
                )
                raise ValueError(f"Invalid adapter name: {msg.adapter}")

            adapter_base_path = os.path.join("/adapters", adapter_name)
            orign_file_path = os.path.join(adapter_base_path, "orign.yaml")
            print(f"Looking for adapter config at: {orign_file_path}", flush=True)

            # TODO: redis
            with open(orign_file_path, "r") as f:
                orign_config = yaml.safe_load(f)
                lora_path = orign_config["latest_checkpoint"]
                print(f"Found LoRA checkpoint at: {lora_path}", flush=True)

            async with vLLM.lora_id_lock:
                if lora_path in vLLM.lora_id_mapping:
                    adapter_id = vLLM.lora_id_mapping[lora_path]
                else:
                    adapter_id = vLLM.next_adapter_id
                    vLLM.lora_id_mapping[lora_path] = adapter_id
                    vLLM.next_adapter_id += 1

            # adapter_id = uuid.uuid5(uuid.NAMESPACE_DNS, lora_path).int >> 64
            lora_request = LoRARequest(msg.adapter, adapter_id, lora_path)
            print(f"Created LoRA request: {lora_request}", flush=True)

        # Prepare the prompts and multimodal data
        prompts = []

        batch_items = msg.batch if msg.batch is not None else [msg.prompt]
        for idx, prompt_item in enumerate(batch_items):
            if prompt_item is None:
                print(f"No prompt found in message item {idx}")
                continue

            model_request_data = await self.formatter.format(prompt_item)
            print(f"\n!!Formatted prompt: {model_request_data}")
            images = model_request_data.image_data
            prompt_text = model_request_data.prompt

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
                    lora_request,
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
        lora_request: Optional[LoRARequest] = None,
    ):
        """Process a single prompt and handle streaming or non-streaming output."""

        print(f"Processing prompt for request_id {request_id}")

        # Log the LoRARequest details
        if lora_request:
            print(
                f"Using LoRA adapter: {lora_request.name} (ID: {lora_request.adapter_id}) at path {lora_request.path}",
                flush=True,
            )
        else:
            print("No LoRA adapter provided.", flush=True)

        print(f"Processing prompt for request_id {request_id}")
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
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
