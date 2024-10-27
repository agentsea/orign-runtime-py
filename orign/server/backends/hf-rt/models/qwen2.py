# qwen2.py
from typing import Optional, Union, Dict, Any, List, Tuple
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch
from torch import Tensor
from confluent_kafka import Producer

from ..model_handler import ModelHandler, PreprocessedInput
from ..config import Config
from ..seq import SequenceState
from ..util import delivery_report

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

        print("\n\n----\nSpecial tokens map:", self.tokenizer.special_tokens_map)
        print("All special tokens:", self.tokenizer.all_special_tokens)
        print("All special token IDs:", self.tokenizer.all_special_ids, "\n\n----\n")

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


    def preprocess_inputs(self, messages: List[Dict[str, str]], **kwargs) -> 'PreprocessedInput':
        """
        Preprocesses the input messages and returns a PreprocessedInput object.
        """
        # Use the tokenizer to format the prompt
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"Formatted prompt text:\n{prompt_text}\n")

        # Tokenize the prompt_text
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Return an instance of PreprocessedInput
        return PreprocessedInput(
            inputs=inputs,
            prompt_length=inputs["input_ids"].shape[1],
            prompt_text=prompt_text,
        )
        
    
    def prepare_inputs_for_generation(self, seq_state: SequenceState) -> Dict[str, Any]:
        if seq_state.past_key_values is None:
            # First generation step
            input_ids = seq_state.generated_tokens  # Shape: [1, seq_len]
            attention_mask = seq_state.attention_mask  # Shape: [1, seq_len]
            position_ids = seq_state.position_ids  # Shape: [1, seq_len]
        else:
            # Subsequent steps: only pass the last generated token
            input_ids = seq_state.generated_tokens[:, -1:]  # Shape: [1, 1]
            attention_mask = torch.ones_like(input_ids, dtype=seq_state.attention_mask.dtype)  # Shape: [1, 1]
            position_ids = seq_state.position_ids[:, -1:] + 1  # Shape: [1, 1]
            
            # Update seq_state's attention_mask and position_ids for future steps
            seq_state.attention_mask = torch.cat([seq_state.attention_mask, attention_mask], dim=1)
            seq_state.position_ids = torch.cat([seq_state.position_ids, position_ids], dim=1)

        model_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "position_ids": position_ids.to(self.device),
        }
        return model_inputs

    def combine_inputs_for_batch(self, model_inputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids, attention_mask, and position_ids from each inputs in model_inputs_list
        input_ids_list = [inputs['input_ids'].squeeze(0) for inputs in model_inputs_list]
        attention_mask_list = [inputs['attention_mask'].squeeze(0) for inputs in model_inputs_list]
        position_ids_list = [inputs['position_ids'].squeeze(0) for inputs in model_inputs_list]

        # Pad sequences to the max length
        from torch.nn.utils.rnn import pad_sequence

        padded_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_attention_mask = pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        padded_position_ids = pad_sequence(
            position_ids_list, batch_first=True, padding_value=0
        )

        # Combine into a single batch
        combined_inputs = {
            'input_ids': padded_input_ids.to(self.device),
            'attention_mask': padded_attention_mask.to(self.device),
            'position_ids': padded_position_ids.to(self.device),
        }

        return combined_inputs
    
    def update_sequence_state(
        self,
        seq_state: SequenceState,
        token: Tensor,
        new_past_key_values: Any,
        producer: Producer
    ) -> None:
        # token shape: [1, 1], ensure it's correct
        if token.dim() == 1:
            token = token.unsqueeze(0)  # Ensure batch dimension

        # Update generated_tokens
        seq_state.generated_tokens = torch.cat([seq_state.generated_tokens, token], dim=1)  # Concatenate along seq_len

        # Update attention_mask and position_ids
        new_attention_mask = torch.ones_like(token, dtype=seq_state.attention_mask.dtype)
        new_position_id = seq_state.position_ids[:, -1:] + 1
        seq_state.attention_mask = torch.cat([seq_state.attention_mask, new_attention_mask], dim=1)
        seq_state.position_ids = torch.cat([seq_state.position_ids, new_position_id], dim=1)

        # Update past_key_values
        seq_state.past_key_values = new_past_key_values

        # Check for end-of-sequence token or max length
        eos_token_id = self.tokenizer.eos_token_id
        if (
            token.item() == eos_token_id
            or seq_state.generated_tokens.shape[1] >= seq_state.max_length
        ):
            print(f"Sequence finished for request_id: {seq_state.request_id}")
            # Compute the correct index to slice from
            prompt_length = seq_state.prompt_length

            # Exclude any tokens corresponding to special tokens or the 'system' message
            response_token_ids = seq_state.generated_tokens[:, prompt_length:]
            output_text = self.decode_tokens(
                response_token_ids[0]
            )
            print(f"\nDecoded output text for request_id {seq_state.request_id}: {output_text}")
            message_value = json.dumps(
                {
                    "type": "generation_response",
                    "request_id": seq_state.request_id,
                    "result": output_text,
                }
            ).encode("utf-8")
            print(f"Sending result for request_id {seq_state.request_id}")
            producer.produce(
                topic=self.config.OUTPUT_TOPIC,
                value=message_value,
                callback=delivery_report,
            )
            # Poll to trigger delivery report callbacks
            producer.poll(0)
            seq_state.is_finished = True


    def generate_next_logits(
        self,
        inputs: Dict[str, Tensor],
        past_key_values: Optional[Any] = None,
        top_k: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[Tensor, Any]:
        print("\nGenerating next logits...")

        # Prepare model inputs
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'position_ids': inputs['position_ids'],
            'use_cache': True,
        }
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

        # Apply top_k filtering
        if top_k is not None:
            print(f"Applying top_k filtering with top_k list: {top_k}")
            next_logits = self.top_k_filtering(next_logits, top_k)

        print(f"Next logits after top_k filtering: {next_logits.shape}")

        # Retrieve new past_key_values
        new_past_key_values = outputs.past_key_values
        if new_past_key_values is not None:
            print("New Past Key Values retrieved.")
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
    
    def combine_past_key_values(
        self, past_key_values_list: List[Optional[Tuple[Tuple[Tensor, Tensor], ...]]]
    ) -> Optional[Tuple[Tuple[Tensor, Tensor], ...]]:
        # Since we are processing sequences with the same past_key_values seq_len, we can directly concatenate
        if past_key_values_list and past_key_values_list[0] is not None:
            num_layers = len(past_key_values_list[0])
            combined_past_key_values = []
            for layer_idx in range(num_layers):
                keys = torch.cat([pkv[layer_idx][0] for pkv in past_key_values_list], dim=0)
                values = torch.cat([pkv[layer_idx][1] for pkv in past_key_values_list], dim=0)
                combined_past_key_values.append((keys, values))
            return tuple(combined_past_key_values)
        else:
            return None

    
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
            decoded = self.tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_text = decoded.strip()
            print(f"\n\nDecoded text with skip_special_tokens=True:\n{decoded_text}\n")
            print(f"Decoded text: {decoded_text}")
            return decoded_text
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

