# seq.py
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
        prompt_text: str,
        prompt_length: int,
    ):
        self.inputs = inputs
        self.generated_tokens = generated_tokens
        self.request_id = request_id
        self.max_length = max_length
        self.top_k = top_k
        self.device = device
        self.prompt_text = prompt_text
        self.prompt_length = prompt_length
        self.is_finished = False
        self.past_key_values = None
        self.attention_mask = inputs['attention_mask'].to(device)
        self.position_ids: torch.Tensor = (self.attention_mask.cumsum(dim=1) - 1).clamp(min=0).to(device)

    def get_generated_text(self, tokenizer):
        # Extract tokens after the prompt
        response_ids = self.generated_tokens[:, self.prompt_length:]
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response_text