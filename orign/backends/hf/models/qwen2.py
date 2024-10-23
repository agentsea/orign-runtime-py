# qwen2.py
from typing import Optional, Union, Dict, Any, List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
import torch
from torch import Tensor
import torch.nn.functional as F


from ..model_handler import ModelHandler
from ..config import Config
from ..seq import SequenceState

class Qwen2(ModelHandler):
    """A model handler for Qwen2"""

    def __init__(self, config: Config) -> None:
        print("Initializing Qwen2 model handler...")
        self.config: Config = config
        print(f"Config: {self.config}")
        self.model_name: str = config.MODEL_NAME
        print(f"Model name: {self.model_name}")
        
        # Updated device selection to include MPS
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        print(f"Device: {self.device}")

        self.trust_remote_code: bool = config.TRUST_REMOTE_CODE
        print(f"Trust remote code: {self.trust_remote_code}")
        self.torch_dtype: Union[str, torch.dtype] = (
            getattr(torch, config.TORCH_DTYPE)
            if config.TORCH_DTYPE != "auto"
            else "auto"
        )
        print(f"torch_dtype: {self.torch_dtype}")
        self.device_map: Union[str, Dict[str, Any], None] = config.DEVICE_MAP
        print(f"Device map: {self.device_map}")

        self.tokenizer: Optional[
            Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        ] = None
        self.model: Optional[AutoModelForCausalLM] = None

        self.load_model()

        if self.model is not None:
            self.model_forward_args = self.model.forward.__code__.co_varnames  # type: ignore
            print(f"Model forward args: {self.model_forward_args}")
        else:
            raise ValueError(f"Failed to load model {self.model_name}")
        print("Initialization complete.")

    def get_supported_modalities(self) -> List[str]:
        """Qwen2 supports only text modality."""
        return ["text"]

    def load_model(self) -> None:
        print("Loading model and tokenizer...")
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            print(f"Loaded tokenizer for model {self.model_name}")
        except Exception as e:
            self.tokenizer = None
            print(f"Error loading tokenizer: {e}")

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            if self.model is not None:
                self.model.eval()
                self.model.to(self.device)
                print(f"Loaded model {self.model_name}")
            else:
                raise ValueError(f"Failed to load model {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        print("Model and tokenizer loading complete.")


    def preprocess_inputs(self, prompt_text: str) -> BatchEncoding:
        """
        Tokenizes the prompt text and returns the model inputs.
        """
        # Tokenize and prepare inputs for the model
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True
        )
        print(f"Tokenized inputs: {inputs}")

        # Move inputs to device
        inputs = inputs.to(self.device)
        print(f"Inputs moved to device {self.device}")
        return inputs
    
    def process_inputs(
        self, messages: List[Dict[str, str]]
    ) -> BatchEncoding:
        print("Processing inputs...")
        print(f"Messages: {messages}")
        supported_modalities = self.get_supported_modalities()
        print(f"Supported modalities: {supported_modalities}")

        if not messages:
            raise ValueError("Messages are required for Qwen2.")

        # Ensure messages are in the correct format
        # Each message should be a dict with keys 'role' and 'content'
        for idx, message in enumerate(messages):
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must have 'role' and 'content' keys.")
            else:
                print(f"Message {idx}: Role: {message['role']}, Content: {message['content']}")

        # Apply the chat template without adding any messages
        formatted_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_special_tokens=False
        )
        print(f"Formatted input: {formatted_input}")

        # Tokenize and prepare inputs for the model
        inputs = self.tokenizer(
            formatted_input, return_tensors="pt", truncation=True
        )
        print(f"Tokenized inputs: {inputs}")

        # Move inputs to device
        inputs = inputs.to(self.device)
        print(f"Inputs moved to device {self.device}")
        return inputs

    def prepare_inputs_for_generation(self, seq_state: SequenceState) -> Dict[str, Any]:
        if seq_state.past_key_values is None:
            # First generation step
            input_ids = seq_state.generated_tokens
            attention_mask = seq_state.attention_mask
        else:
            # Subsequent steps: only pass the last generated token
            input_ids = seq_state.generated_tokens[:, -1:]
            attention_mask = seq_state.attention_mask[:, -1:]

        model_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "past_key_values": seq_state.past_key_values,
            "use_cache": True,
        }

        return model_inputs

    def combine_inputs_for_batch(self, model_inputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and attention_mask from each inputs in model_inputs_list
        input_ids_list = [inputs['input_ids'].squeeze(0) for inputs in model_inputs_list]
        attention_mask_list = [inputs['attention_mask'].squeeze(0) for inputs in model_inputs_list]

        print("Input IDs shapes before padding:")
        for idx, input_ids in enumerate(input_ids_list):
            print(f"Sequence {idx}: Shape {input_ids.shape}")

        # Pad input_ids and attention_mask tensors to the same length
        from torch.nn.utils.rnn import pad_sequence

        # Pad sequences to the max length
        padded_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_attention_mask = pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )

        print(f"Padded Input IDs shape: {padded_input_ids.shape}")
        print(f"Padded Attention Mask shape: {padded_attention_mask.shape}")

        # Combine into a single batch
        combined_inputs = {
            'input_ids': padded_input_ids.to(self.device),
            'attention_mask': padded_attention_mask.to(self.device),
        }

        return combined_inputs
    
    def update_sequence_states(
        batch_sequences: List[SequenceState],
        next_tokens: torch.Tensor,
        new_past_key_values: List[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    ):
        for seq_state, token, past_key_values in zip(
            batch_sequences, next_tokens, new_past_key_values
        ):
            # Move token to the appropriate device and dtype
            token = token.to(device=seq_state.device, dtype=seq_state.generated_tokens.dtype)

            # Update generated_tokens and attention_mask
            seq_state.generated_tokens = torch.cat([seq_state.generated_tokens, token], dim=1)
            seq_state.attention_mask = torch.cat(
                [seq_state.attention_mask, torch.ones_like(token, device=seq_state.device)], dim=1
            )

            # Update past_key_values
            seq_state.past_key_values = past_key_values

    def update_sequence_state(
        self,
        seq_state: SequenceState,
        new_token: torch.Tensor
    ) -> None:
        """Update the sequence state with the new token and past key values."""
        # Ensure new_token has shape [batch_size, 1]
        if new_token.dim() == 1:
            new_token = new_token.unsqueeze(-1)  # Adds a dimension at the end
        elif new_token.dim() == 0:
            new_token = new_token.unsqueeze(0).unsqueeze(0)  # From [] to [1, 1]
        
        print(f"seq_state.generated_tokens shape: {seq_state.generated_tokens.shape}")
        print(f"new_token shape after unsqueeze: {new_token.shape}")

        # Concatenate the new token to the generated tokens
        seq_state.generated_tokens = torch.cat(
            [seq_state.generated_tokens, new_token],
            dim=1
        )
        print(f"Updated generated_tokens shape: {seq_state.generated_tokens.shape}")

        # Update attention_mask
        new_attention_mask = torch.ones_like(new_token, device=seq_state.device)
        seq_state.attention_mask = torch.cat(
            [seq_state.attention_mask, new_attention_mask],
            dim=1
        )

        # Note: Ensure that new_past_key_values is passed to this method
        # Update past_key_values
        # seq_state.past_key_values = new_past_key_values

    def generate_next_logits(
        self,
        inputs: Dict[str, Tensor],
        past_key_values: Optional[Any] = None,
        **kwargs
    ) -> Tuple[Tensor, Any]:
        print("\nGenerating next logits...")
        
        # Extract inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        position_ids = inputs.get('position_ids', None)
        
        # Debugging statements for inputs
        print(f"Input IDs shape: {input_ids.shape}")
        if attention_mask is not None:
            print(f"Attention Mask shape: {attention_mask.shape}")
        else:
            print("Attention Mask: None")
        if position_ids is not None:
            print(f"Position IDs shape: {position_ids.shape}")
        else:
            print("Position IDs: None")
        
        if past_key_values is not None:
            print("Past Key Values are provided.")
            for idx, pkv in enumerate(past_key_values):
                key_shape = pkv[0].shape
                value_shape = pkv[1].shape
                print(f"Layer {idx} - Key shape: {key_shape}, Value shape: {value_shape}")
        else:
            print("Past Key Values: None")
        
        # Prepare model inputs
        model_inputs = {
            'input_ids': input_ids,
            'use_cache': True,
        }
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask
        if position_ids is not None:
            model_inputs['position_ids'] = position_ids
        if past_key_values is not None:
            model_inputs['past_key_values'] = past_key_values
        
        # Forward pass through the model
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        # Get logits for the last token
        next_logits = outputs.logits[:, -1, :]
        print(f"Next logits shape: {next_logits.shape}")
        
        # Retrieve new past_key_values
        new_past_key_values = outputs.past_key_values
        if new_past_key_values is not None:
            print("New Past Key Values retrieved.")
            for idx, pkv in enumerate(new_past_key_values):
                key_shape = pkv[0].shape
                value_shape = pkv[1].shape
                print(f"Layer {idx} - New Key shape: {key_shape}, New Value shape: {value_shape}")
        else:
            print("New Past Key Values: None")
        
        return next_logits, new_past_key_values
    
    def top_k_filtering(self, logits: Tensor, top_k: List[int]) -> Tensor:
        print(f"Applying top_k filtering with top_k list: {top_k}")
        batch_size = logits.size(0)
        filtered_logits = logits.clone()

        for i in range(batch_size):
            k = top_k[i] if i < len(top_k) else top_k[-1]  # Use last value if list is shorter
            k = min(max(k, 1), logits.size(-1))  # Safety check
            indices_to_remove = filtered_logits[i] < torch.topk(filtered_logits[i], k)[0][-1]
            filtered_logits[i][indices_to_remove] = float('-inf')

        print(f"Logits after filtering: {filtered_logits}")
        return filtered_logits

    def convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts a list of messages into a single prompt string using the tokenizer's chat template.
        """
        formatted_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_special_tokens=False
        )
        print(f"Formatted input: {formatted_input}")
        return formatted_input
    
    def combine_past_key_values(
        self, past_key_values_list: List[Optional[Tuple[Tuple[Tensor, Tensor], ...]]]
    ) -> Optional[Tuple[Tuple[Tensor, Tensor], ...]]:
        # Check if all past_key_values are None
        if all(pkv is None for pkv in past_key_values_list):
            print("All past_key_values are None.")
            return None

        print("Combining past_key_values for the batch...")
        # Determine the number of layers from the first non-None past_key_values
        num_layers = len(next(pkv for pkv in past_key_values_list if pkv is not None))
        combined_past_key_values = []

        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            max_seq_length = 0

            # Extract num_heads, head_dim, and dtype from existing past_key_values
            for pkv in past_key_values_list:
                if pkv is not None:
                    key_sample, value_sample = pkv[layer_idx]
                    num_heads = key_sample.size(1)
                    head_dim = key_sample.size(-1)
                    key_dtype = key_sample.dtype
                    value_dtype = value_sample.dtype
                    break
            else:
                # Fallback to model config if all past_key_values are None
                num_heads = getattr(self.model.config, 'num_key_value_heads', self.model.config.num_attention_heads)
                head_dim = self.model.config.hidden_size // num_heads
                key_dtype = torch.float32  # Default dtype if no existing key tensors
                value_dtype = torch.float32

            for pkv in past_key_values_list:
                if pkv is None:
                    # Initialize empty tensors for sequences with no past_key_values
                    seq_length = 0  # No previous tokens

                    # Create empty tensors with correct dimensions and dtype
                    key = torch.empty(
                        (1, num_heads, seq_length, head_dim), dtype=key_dtype, device=self.device
                    )
                    value = torch.empty(
                        (1, num_heads, seq_length, head_dim), dtype=value_dtype, device=self.device
                    )
                else:
                    key, value = pkv[layer_idx]
                    seq_length = key.size(2)  # Update seq_length based on actual tensor

                max_seq_length = max(max_seq_length, seq_length)
                layer_keys.append(key)
                layer_values.append(value)

            # Now, pad the keys and values to the max_seq_length
            padded_keys = []
            padded_values = []

            for key in layer_keys:
                seq_length = key.size(2)
                if seq_length < max_seq_length:
                    # Pad the key tensor
                    pad_size = (0, 0, 0, max_seq_length - seq_length)
                    key = F.pad(key, pad=pad_size, mode='constant', value=0)
                padded_keys.append(key)

            for value in layer_values:
                seq_length = value.size(2)
                if seq_length < max_seq_length:
                    # Pad the value tensor
                    pad_size = (0, 0, 0, max_seq_length - seq_length)
                    value = F.pad(value, pad=pad_size, mode='constant', value=0)
                padded_values.append(value)

            # Concatenate tensors along batch dimension
            combined_key = torch.cat(padded_keys, dim=0)  # Shape: [batch_size, num_heads, seq_length, head_dim]
            combined_value = torch.cat(padded_values, dim=0)

            print(f"Layer {layer_idx} combined key shape: {combined_key.shape}")
            print(f"Layer {layer_idx} combined value shape: {combined_value.shape}")

            combined_past_key_values.append((combined_key, combined_value))

        return tuple(combined_past_key_values)
    
    def split_past_key_values(
        self, past_key_values: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> List[Tuple[Tuple[Tensor, Tensor], ...]]:
        batch_size = past_key_values[0][0].size(0)  # Get batch size from the first layer's key tensor

        split_past_key_values = []

        for batch_idx in range(batch_size):
            single_sequence_pkv = []

            for layer_pkv in past_key_values:
                key, value = layer_pkv

                # Extract the single sequence's key and value
                single_key = key[batch_idx : batch_idx + 1]  # Retain batch dimension
                single_value = value[batch_idx : batch_idx + 1]

                single_sequence_pkv.append((single_key, single_value))

            split_past_key_values.append(tuple(single_sequence_pkv))

        return split_past_key_values
        
    def decode_tokens(self, tokens: Tensor) -> str:
        print(f"Decoding tokens: {tokens}")
        if self.tokenizer:
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"Decoded text: {decoded}")
            return decoded
        else:
            raise ValueError("Tokenizer is not loaded.")

    @classmethod
    def supported_modalities(cls) -> List[str]:
        return ["text"]

    @classmethod
    def supported_models(cls) -> List[str]:
        return [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2-72B",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B",
        "Qwen/Qwen2-57B-A14B-Instruct",
        "Qwen/Qwen2-57B-A14B",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-1.5B",
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-0.5B",
        "Qwen/Qwen2-72B-Instruct-AWQ",
        "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-7B-Instruct-AWQ",
        "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-1.5B-Instruct-AWQ",
        "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-0.5B-Instruct-AWQ",
        "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-0.5B-Instruct-MLX",
        "Qwen/Qwen2-7B-Instruct-GGUF",
        "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "Qwen/Qwen2-1.5B-Instruct-MLX",
        "Qwen/Qwen2-Math-72B",
        "Qwen/Qwen2-Math-72B-Instruct",
        "Qwen/Qwen2-Math-7B",
        "Qwen/Qwen2-Math-7B-Instruct",
        "Qwen/Qwen2-Math-1.5B",
        "Qwen/Qwen2-Math-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct-MLX",
        "Qwen/Qwen2-1.5B-Instruct-GGUF",
        "Qwen/Qwen2-72B-Instruct-GGUF",
        "Qwen/Qwen2-57B-A14B-Instruct-GGUF"
    ]

