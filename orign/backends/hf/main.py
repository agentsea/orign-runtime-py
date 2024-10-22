# main.py
import json
import asyncio
from typing import Any, Dict, List, Optional, cast

from confluent_kafka import Consumer, Producer, KafkaException
import torch
from transformers import BatchEncoding
import torch.nn.utils.rnn
import torch.nn.functional as F

from .model_handler import ModelHandler
from .config import Config

config: Config = Config()

# Initialize confluent-kafka Consumer
consumer_conf = {
    "bootstrap.servers": ",".join(config.BOOTSTRAP_SERVERS),
    "group.id": config.GROUP_ID,
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}
consumer: Consumer = Consumer(consumer_conf)
consumer.subscribe([config.INPUT_TOPIC])

# Initialize confluent-kafka Producer
producer_conf = {
    "bootstrap.servers": ",".join(config.BOOTSTRAP_SERVERS),
}
producer: Producer = Producer(producer_conf)

# Initialize ModelHandler
model_handler: ModelHandler = ModelHandler(config)

# Batch processing variables
batch_size: int = config.BATCH_SIZE
requests_queue: List[Dict[str, Any]] = []


# State for each sequence in the batch
class SequenceState:
    def __init__(
        self,
        inputs: BatchEncoding,
        generated_tokens: torch.Tensor,
        request_id: str,
        max_length: int,
    ) -> None:
        self.inputs: BatchEncoding = inputs
        self.generated_tokens: torch.Tensor = generated_tokens
        self.request_id: str = request_id
        self.is_finished: bool = False
        self.max_length: int = max_length


def delivery_report(err, msg):
    """Callback function for producer to report message delivery."""
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")


async def generate_responses() -> None:
    batch_states: List[SequenceState] = []

    while True:
        # Fill up the batch with available requests
        fill_batch(batch_states, batch_size, requests_queue)

        if not batch_states:
            await asyncio.sleep(0.01)  # No requests to process
            continue

        # Prepare inputs for this iteration
        input_ids_list = [s.generated_tokens.squeeze(0) for s in batch_states]
        attention_mask_list = [
            s.inputs["attention_mask"].squeeze(0) for s in batch_states
        ]

        # Get the sequence lengths before padding
        seq_lens = [len(ids) for ids in input_ids_list]

        # Pad sequences to the same length
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=model_handler.tokenizer.pad_token_id,
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )

        # Move tensors to the device
        input_ids_padded = input_ids_padded.to(model_handler.device)
        attention_mask_padded = attention_mask_padded.to(model_handler.device)

        # Handle pixel_values if present and model supports images
        if model_handler.supports_images and "pixel_values" in batch_states[0].inputs:
            pixel_values_list = [s.inputs["pixel_values"] for s in batch_states]
            pixel_values = torch.cat(pixel_values_list, dim=0).to(model_handler.device)
        else:
            pixel_values = None

        # Generate logits with top_k support
        next_token_logits = model_handler.generate_next_logits(
            input_ids=input_ids_padded,
            attention_mask=attention_mask_padded,
            pixel_values=pixel_values,
            top_k=50,
        )
        print("next logit tokens: ", next_token_logits)

        # Sample next tokens from the logits
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probabilities, num_samples=1)

        print("next tokens: ", next_tokens)
        # Update states
        finished_indices = []
        for i, seq_state in enumerate(batch_states):
            if not seq_state.is_finished:
                # Append the new token to generated_tokens
                token = next_tokens[i].unsqueeze(0)  # shape (1,1)
                print("token: ", token)

                seq_state.generated_tokens = torch.cat(
                    [seq_state.generated_tokens.to(model_handler.device), token], dim=1
                )
                attention_mask_tensor = seq_state.inputs["attention_mask"].to(
                    model_handler.device
                )
                ones_tensor = torch.ones(
                    (1, 1),
                    device=model_handler.device,
                    dtype=attention_mask_tensor.dtype,
                )
                seq_state.inputs["attention_mask"] = torch.cat(
                    [attention_mask_tensor, ones_tensor], dim=1
                )

                # Check for end-of-sequence token or max length
                eos_token_id = model_handler.tokenizer.eos_token_id

                if (
                    token.item() == eos_token_id
                    or seq_state.generated_tokens.shape[1] >= seq_state.max_length
                ):
                    # Send the result
                    output_text = model_handler.decode_tokens(
                        seq_state.generated_tokens[0]
                    )
                    message_value = json.dumps(
                        {
                            "request_id": seq_state.request_id,
                            "result": output_text,
                        }
                    ).encode("utf-8")
                    print("sending result text: ", output_text)
                    producer.produce(
                        topic=config.OUTPUT_TOPIC,
                        value=message_value,
                        callback=delivery_report,
                    )
                    # Poll to trigger delivery report callbacks
                    producer.poll(0)
                    seq_state.is_finished = True
                    finished_indices.append(i)

        # Remove finished sequences
        for index in sorted(finished_indices, reverse=True):
            del batch_states[index]

        # Fill up the batch with new requests
        fill_batch(batch_states, batch_size, requests_queue)

        # Small sleep to yield control
        await asyncio.sleep(0.01)


def fill_batch(
    batch_states: List[SequenceState],
    batch_size: int,
    requests_queue: List[Dict[str, Any]],
) -> None:
    while len(batch_states) < batch_size and requests_queue:
        data: Dict[str, Any] = requests_queue.pop(0)
        text: str = data.get("text", "")
        image_url: Optional[str] = data.get("image_url", None)
        request_id: str = data["request_id"]
        request_max_length: int = data.get("max_tokens", config.MAX_LENGTH)

        try:
            # Process inputs
            inputs: BatchEncoding = model_handler.process_inputs(
                text=text, image_url=image_url
            )

            # Initialize generated tokens
            generated_tokens: torch.Tensor = inputs["input_ids"].clone()  # type: ignore

            new_state: SequenceState = SequenceState(
                inputs, generated_tokens, request_id, request_max_length
            )
            batch_states.append(new_state)
        except Exception as e:
            print(f"Error processing inputs for request {request_id}: {e}")
            continue


async def consume_requests() -> None:
    while True:
        try:
            msg = consumer.poll(1.0)  # Poll for messages with a timeout
            if msg is None:
                await asyncio.sleep(0.01)
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            data: Dict[str, Any] = json.loads(msg.value().decode("utf-8"))
            print("got request: ", data)
            base_request_id = f"{msg.topic()}-{msg.partition()}-{msg.offset()}"

            # Process the 'batch' field in the data
            if "batch" in data:
                for idx, message_item in enumerate(data["batch"]):
                    text = ""
                    image_url = None
                    # Handle different message structures
                    if "content" in message_item:
                        for content_item in message_item["content"]:
                            if content_item["type"] == "text":
                                text += content_item["text"] + " "
                            elif content_item["type"] == "image_url":
                                if "url" in content_item["image_url"]:
                                    image_url = content_item["image_url"]["url"]
                    elif "messages" in message_item:
                        # Handle nested messages
                        for message in message_item["messages"]:
                            if message["role"] == "user":
                                text += message["content"] + " "
                    else:
                        print(f"Unknown message format in batch item: {message_item}")
                        continue

                    # Prepare the request data
                    request_data = {
                        "text": text.strip(),
                        "image_url": image_url,
                        "request_id": f"{base_request_id}-{idx}",
                        "max_tokens": data.get("max_tokens", config.MAX_LENGTH),
                    }
                    requests_queue.append(request_data)
            else:
                print(f"No 'batch' field in message: {base_request_id}")
            # Manually commit the message
            consumer.commit(message=msg, asynchronous=False)
        except KafkaException as e:
            print(f"Kafka error: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")


async def main() -> None:
    consumer_task = asyncio.create_task(consume_requests())
    generator_task = asyncio.create_task(generate_responses())
    await asyncio.gather(consumer_task, generator_task)


try:
    asyncio.run(main())
finally:
    # Clean up resources
    consumer.close()
    producer.flush()
