# model_handler.py
import requests
import io
from typing import Optional, Union, Dict, Any
import inspect

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
import torch
from torch import Tensor

from .config import Config


class ModelHandler:
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.model_name: str = config.MODEL_NAME
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.trust_remote_code: bool = config.TRUST_REMOTE_CODE
        self.torch_dtype: Union[str, torch.dtype] = (
            getattr(torch, config.TORCH_DTYPE)
            if config.TORCH_DTYPE != "auto"
            else "auto"
        )
        self.device_map: Union[str, Dict[str, Any], None] = config.DEVICE_MAP

        self.processor: Optional[AutoProcessor] = None
        self.tokenizer: Optional[
            Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        ] = None
        self.model: Optional[AutoModelForCausalLM] = None

        self.load_model()
        self.model_forward_args = self.model.forward.__code__.co_varnames  # type: ignore

        if self.model is not None:
            forward_params = inspect.signature(self.model.forward).parameters
            self.supports_images = "pixel_values" in forward_params
            print(f"Model supports images: {self.supports_images}")
        else:
            self.supports_images = False

    def load_model(self) -> None:
        # Try to load processor if available
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            print(f"Loaded processor for model {self.model_name}")
        except Exception:
            self.processor = None
            print(f"No processor available for model {self.model_name}")

        # Try to load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            print(f"Loaded tokenizer for model {self.model_name}")
        except Exception:
            self.tokenizer = None
            print(f"No tokenizer available for model {self.model_name}")

        try:
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            if self.model is not None:
                self.model.eval()  # type: ignore
                self.model.to(self.device)  # type: ignore
                print(f"Loaded model {self.model_name}")
            else:
                raise ValueError(f"Failed to load model {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def process_inputs(
        self, text: Optional[str] = None, image_url: Optional[str] = None
    ) -> BatchEncoding:
        print("processing inputs for: ", text, image_url)

        # Prepare the input in the format required by Qwen2.5 (chat template if needed)
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": text if text else ""},
        ]
        formatted_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and prepare inputs for the model
        inputs = self.tokenizer(formatted_input, return_tensors="pt", truncation=True)

        # Move inputs to device
        inputs = inputs.to(self.device)
        return inputs

    def generate_next_logits(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Optional[Tensor] = None,
        top_k: int = 4,
    ) -> Tensor:
        with torch.no_grad():
            model_inputs: Dict[str, Tensor] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if pixel_values is not None and self.supports_images:
                model_inputs["pixel_values"] = pixel_values
            elif pixel_values is not None and not self.supports_images:
                print("Model does not accept 'pixel_values' input; ignoring it.")

            if self.model is None:
                raise ValueError("Model is not loaded.")

            outputs = self.model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]

            if top_k > 0:
                # Apply top_k filtering
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            return next_token_logits

    def decode_tokens(self, tokens: Tensor) -> str:
        if (
            self.processor
            and hasattr(self.processor, "tokenizer")
            and hasattr(self.processor.tokenizer, "decode")  # type: ignore
        ):
            return self.processor.tokenizer.decode(tokens, skip_special_tokens=True)  # type: ignore
        elif self.tokenizer:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            raise ValueError("No tokenizer available for decoding.")
