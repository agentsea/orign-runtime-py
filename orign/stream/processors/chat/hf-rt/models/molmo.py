# molmo_model_handler.py
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Any, Optional
from confluent_kafka import Producer
import json
import requests
from PIL import Image
import logging

from ..model_handler import ModelHandler, PreprocessedInput
from ..seq import SequenceState

logger = logging.getLogger(__name__)

class Molmo(ModelHandler):
    def __init__(self, config):
        self.config = config
        self.load_model()

    def load_model(self) -> None:
        # Check for GPU availability and set the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
        )

        # Load the model onto the GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map={'': self.device}
        )
        self.model.eval()

    def prepare_inputs_for_generation(self, seq_state: SequenceState) -> Dict[str, Tensor]:
        # Prepare the inputs needed for generation
        inputs = {
            'input_ids': seq_state.generated_tokens.to(self.device),
            'attention_mask': torch.ones_like(seq_state.generated_tokens).to(self.device),
        }
        return inputs

    def preprocess_inputs(self, messages: List[Dict[str, str]], **kwargs) -> PreprocessedInput:
        # Initialize variables
        image = None
        image_provided = False
        texts = []
        
        # Build the prompt string by concatenating messages
        for message in messages:
            role = message.get('role', 'user').capitalize()
            content = message.get('content', '')
            if isinstance(content, dict):
                text = content.get('text', '')
                image_url = content.get('image_url', '')
            else:
                text = content
                image_url = ''
            
            # Append role and content to the prompt
            texts.append(f"{role}: {text}")
            
            # Handle image if present and not already set
            if image_url and not image_provided:
                try:
                    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                    image_provided = True
                    print(f"Loaded image from URL: {image_url}")
                except Exception as e:
                    print(f"Error loading image from URL {image_url}: {e}")
        
        # If no image is provided, use a default image
        if image is None:
            # Create a black image as a placeholder
            image = Image.new('RGB', (224, 224), color='black')
            print("No image provided, using default black image.")
        
        # Combine all texts into a single prompt string
        prompt = "\n".join(texts) + "\nAssistant:"
        
        # Process the inputs
        input_data = self.processor.process(
            images=image,
            text=prompt,
            return_tensors='pt',
            padding=True,
        )
        
        # Move tensors to device
        for k, v in input_data.items():
            if isinstance(v, torch.Tensor):
                input_data[k] = v.to(self.device)
        
        print(f"input_data keys: {input_data.keys()}")
        print(f"input_data['input_ids'] shape: {input_data['input_ids'].shape}")
        print(f"input_data['input_ids']: {input_data['input_ids']}")
        
        prompt_length = input_data['input_ids'].shape[1] if len(input_data['input_ids'].shape) > 1 else 0
        prompt_text = prompt
        
        return PreprocessedInput(
            inputs=input_data,
            prompt_length=prompt_length,
            prompt_text=prompt_text
        )

    def update_sequence_state(
            self,
            seq_state: SequenceState,
            token: Tensor,
            new_past_key_values: Any,
            producer: Producer
        ) -> None:

        seq_state.generated_tokens = torch.cat([seq_state.generated_tokens, token.unsqueeze(0)], dim=1)
        seq_state.past_key_values = new_past_key_values

        # Check if generation should stop
        if token.item() == self.processor.tokenizer.eos_token_id or seq_state.generated_tokens.shape[1] >= seq_state.max_length:
            seq_state.is_finished = True

        # Decode the token
        token_text = self.decode_tokens(token)

        # Send the token_text to the output topic
        output_message = json.dumps({
            'type': 'token',
            'request_id': seq_state.request_id,
            'token': token_text
        }).encode('utf-8')

        producer.produce(
            topic=self.config.OUTPUT_TOPIC,
            value=output_message,
            # Provide a delivery_report function or remove callback if not needed
        )
        producer.poll(0)

    def combine_inputs_for_batch(self, model_inputs_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        batched_inputs = {}
        for key in model_inputs_list[0].keys():
            if isinstance(model_inputs_list[0][key], torch.Tensor):
                tensors_to_stack = [inputs[key] for inputs in model_inputs_list]
                batched_inputs[key] = torch.stack(tensors_to_stack, dim=0)
            else:
                batched_inputs[key] = [inputs[key] for inputs in model_inputs_list]
        return batched_inputs

    def generate_next_logits(
        self,
        model_inputs: Dict[str, Tensor],
        past_key_values: Optional[Tuple] = None,
        top_k: List[int] = None,
    ) -> Tuple[Tensor, Tuple]:
        if past_key_values is not None:
            input_ids = model_inputs['input_ids'][:, -1:]  # Last token
        else:
            input_ids = model_inputs['input_ids']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=model_inputs.get('attention_mask', None),
            past_key_values=past_key_values,
            use_cache=True
        )

        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        next_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        new_past_key_values = outputs.past_key_values

        return next_logits, new_past_key_values

    def combine_past_key_values(self, past_key_values_list):
        num_layers = len(past_key_values_list[0])
        combined_past_key_values = []

        for layer in range(num_layers):
            keys = []
            values = []
            for past in past_key_values_list:
                key, value = past[layer]
                keys.append(key)
                values.append(value)
            combined_key = torch.cat(keys, dim=0)
            combined_value = torch.cat(values, dim=0)
            combined_past_key_values.append((combined_key, combined_value))

        return tuple(combined_past_key_values)

    def split_past_key_values(self, new_past_key_values):
        batch_size = new_past_key_values[0][0].size(0)
        num_layers = len(new_past_key_values)
        past_key_values_list = []

        for batch_idx in range(batch_size):
            past = []
            for layer in range(num_layers):
                key = new_past_key_values[layer][0][batch_idx:batch_idx+1]
                value = new_past_key_values[layer][1][batch_idx:batch_idx+1]
                past.append((key, value))
            past_key_values_list.append(tuple(past))

        return past_key_values_list

    def decode_tokens(self, tokens: Tensor) -> str:
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True)

    @classmethod
    def supported_modalities(cls) -> List[str]:
        return ['text', 'image']

    @classmethod
    def supported_models(cls) -> List[str]:
        return ['allenai/Molmo-7B-D-0924']
