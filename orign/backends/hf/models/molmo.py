from typing import Optional, Union, Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
from torch import Tensor
import warnings
from collections import defaultdict
from PIL import Image
import requests
import gc
import psutil
import os
from contextlib import contextmanager, nullcontext

from ..model_handler import ModelHandler, PreprocessedInput
from ..config import Config
from ..seq import SequenceState

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class Molmo(ModelHandler):
    """A model handler for Molmo"""

    def __init__(self, config: Config) -> None:
        print("Initializing Molmo model handler...")
        self.config: Config = config
        self.model_name: str = config.MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.trust_remote_code: bool = config.TRUST_REMOTE_CODE
        self.torch_dtype: Union[str, torch.dtype] = (
            getattr(torch, config.TORCH_DTYPE)
            if config.TORCH_DTYPE != "auto"
            else "auto"
        )
        self.device_map: Union[str, Dict[str, Any], None] = config.DEVICE_MAP

        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForCausalLM] = None

        self.load_model()

        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)
            print(f"Loaded model {self.model_name}")
        else:
            raise ValueError(f"Failed to load model {self.model_name}")
        print("Initialization complete.")

    def load_model(self) -> None:
        print("Loading model and processor...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            print(f"Loaded processor for model {self.model_name}")
        except Exception as e:
            print(f"Error loading model or processor: {e}")
            self.model = None
            self.processor = None

    def preprocess_inputs(self, messages: List[Dict[str, str]], **kwargs) -> 'PreprocessedInput':
        image_url = messages[0].get("image_url")
        text = messages[0].get("text")
        image = Image.open(requests.get(image_url, stream=True).raw)

        inputs = self.processor.process(
            images=[image],
            text=text,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        return PreprocessedInput(
            inputs=inputs,
            prompt_length=inputs["input_ids"].shape[1],
            prompt_text=text,
        )

    def generate_next_logits(
        self,
        inputs: Dict[str, Tensor],
        past_key_values: Optional[Any] = None,
        **kwargs
    ) -> Tuple[Tensor, Any]:
        print("\nGenerating next logits...")

        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'use_cache': True,
        }
        if past_key_values is not None:
            model_inputs['past_key_values'] = past_key_values

        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_logits = outputs.logits[:, -1, :]
        new_past_key_values = outputs.past_key_values

        return next_logits, new_past_key_values

    def decode_tokens(self, tokens: Tensor) -> str:
        print(f"Decoding tokens: {tokens}")
        if self.processor:
            decoded = self.processor.tokenizer.decode(                          
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return decoded.strip()
        else:
            raise ValueError("Processor is not loaded.")

    @classmethod
    def supported_modalities(cls) -> List[str]:
        return ["text", "image"]

    @classmethod
    def supported_models(cls) -> List[str]:
        return [
            "allenai/Molmo-72B-0924",
            "allenai/Molmo-7B-D-0924",
            "allenai/Molmo-7B-O-0924",
            "allenai/MolmoE-1B-0924"
        ]

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024 / 1024
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        return f"CPU Memory: {cpu_mem:.2f}MB, GPU Memory: {gpu_mem:.2f}MB"

    @contextmanager
    def batch_memory_manager(self):
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Memory after batch cleanup: {self.get_memory_usage()}")