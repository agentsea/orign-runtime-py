# model_handler.py
from abc import ABC, abstractmethod
from typing import  Dict, Any, List, Tuple, Optional

from transformers import (
    BatchEncoding,
)
from torch import Tensor
from .seq import SequenceState

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
    def convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts a list of messages into a single prompt string using the tokenizer's chat template.
        """
        pass

    @abstractmethod
    def process_inputs(
        self, messages: List[Dict[str, str]]
    ) -> Any:
        pass

    @abstractmethod
    def update_sequence_state(
        self,
        seq_state: SequenceState,
        new_token: Tensor
    ) -> None:
        """Updates the sequence state with the new token and any other model-specific states."""
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
