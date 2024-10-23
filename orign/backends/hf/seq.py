# seq.py

from typing import Optional, Tuple
import torch
from transformers import BatchEncoding
import torch.nn.utils.rnn

class SequenceState:
    def __init__(
        self,
        inputs: BatchEncoding,
        generated_tokens: torch.Tensor,
        request_id: str,
        max_length: int,
        top_k: int,
        device: torch.device,
        prompt_text: str = ""
    ) -> None:
        self.inputs: BatchEncoding = inputs
        if generated_tokens.dim() == 1:
            generated_tokens = generated_tokens.unsqueeze(0)
        self.generated_tokens: torch.Tensor = generated_tokens.to(device)
        self.request_id: str = request_id
        self.max_length: int = max_length
        self.top_k = top_k
        self.is_finished: bool = False
        self.device = device
        self.attention_mask = inputs['attention_mask'].to(device)
        # Initialize position_ids
        self.position_ids: torch.Tensor = (self.attention_mask.cumsum(dim=1) - 1).clamp(min=0).to(device)
        self.past_key_values = None
        self.prompt_length: int = generated_tokens.shape[1]
        self.prompt_text: str = prompt_text