# model_handler.py
from abc import ABC, abstractmethod
from typing import  Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from transformers import (
    BatchEncoding,
)
from torch import Tensor
from .seq import SequenceState
from confluent_kafka import Producer


@dataclass
class PreprocessedInput:
    inputs: BatchEncoding
    prompt_length: int
    prompt_text: str


class ModelHandler(ABC):
    """An abstract base class for model handlers"""

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def prepare_inputs_for_generation(
        self,
        seq_state: SequenceState
    ) -> Dict[str, Tensor]:
        """Prepares model-specific inputs for generation."""
        pass

    @abstractmethod
    def preprocess_inputs(self, prompt_text: str) -> BatchEncoding:
        """
        Tokenizes the prompt text and returns the model inputs.
        """
        pass

    @abstractmethod
    def preprocess_inputs(self, messages: List[Dict[str, str]], **kwargs) -> 'PreprocessedInput':
        pass

    @abstractmethod
    def update_sequence_state(
            self,
            seq_state: SequenceState,
            token: Tensor,
            new_past_key_values: Any,
            producer: Producer
        ) -> None:
        pass

    @abstractmethod
    def combine_inputs_for_batch(
        self,
        model_inputs_list: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """Combines individual model inputs into batched inputs."""
        pass

    @abstractmethod
    def generate_next_logits(
        self,
        model_inputs: Dict[str, Tensor],
        past_key_values: Optional[Tuple] = None,
        top_k: List[int] = None,
    ) -> Tuple[Tensor, Tuple]:
        pass
    
    @abstractmethod
    def combine_past_key_values(self, past_key_values_list):
        pass
    
    @abstractmethod
    def split_past_key_values(self, new_past_key_values):
        pass

    @abstractmethod
    def decode_tokens(self, tokens: Tensor) -> str:
        pass

    @classmethod
    @abstractmethod
    def supported_modalities(cls) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def supported_models(cls) -> List[str]:
        pass
