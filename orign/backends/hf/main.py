# main.py
import json
import asyncio
from typing import Any, Dict, List
import traceback
import time
import threading
from queue import Queue, Empty
from collections import defaultdict

from confluent_kafka import Consumer, Producer, KafkaException
import torch
import torch.nn.utils.rnn

from .model_handler import ModelHandler
from .model_factory import get_model_handler
from .config import Config
from .seq import SequenceState

print("Starting main.py")

config: Config = Config()

# Initialize confluent-kafka Consumer
consumer_conf = {
    "bootstrap.servers": ",".join(config.BOOTSTRAP_SERVERS),
    "group.id": config.GROUP_ID,
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
    "max.poll.interval.ms": 900000,
}
consumer: Consumer = Consumer(consumer_conf)
print(f"Initialized Kafka consumer with config: {consumer_conf}")
consumer.subscribe([config.INPUT_TOPIC])
print(f"Subscribed to topic: {config.INPUT_TOPIC}")

# Initialize confluent-kafka Producer
producer_conf = {
    "bootstrap.servers": ",".join(config.BOOTSTRAP_SERVERS),
}
producer: Producer = Producer(producer_conf)
print(f"Initialized Kafka producer with config: {producer_conf}")

# Initialize ModelHandler
model_handler: ModelHandler = get_model_handler(config.MODEL_NAME)
print("Initialized ModelHandler")

# Batch processing variables
batch_size: int = config.BATCH_SIZE
requests_queue: Queue = Queue()
print(f"Batch size set to: {batch_size}")

# State for each sequence in the batch

def delivery_report(err, msg):
    """Callback function for producer to report message delivery."""
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")


def generate_responses() -> None:
    batch_states: List[SequenceState] = []
    print("Starting generate_responses()")
    while True:
        start_time = time.time()
        try:
            print("\n\n-----Entering generate_responses loop-----\n\n")
            # Fill up the batch with available requests
            fill_batch(batch_states, requests_queue, model_handler, config.BATCH_SIZE)
            print(f"Batch states after fill_batch: {batch_states}")

            if not batch_states:
                print("No batch states available, sleeping for 0.01 seconds.")
                time.sleep(0.01)
                continue

            current_batch_size = min(len(batch_states), config.BATCH_SIZE)
            print(f"Current batch size: {current_batch_size}")

            # Collect inputs for all sequences in the batch
            model_inputs_list = []
            past_key_values_list = []
            request_ids = []
            max_lengths = []
            top_k_list = []

            batch_sequences = batch_states[:current_batch_size]
            print(f"Processing batch sequences: {batch_sequences}")

            for seq_state in batch_sequences:
                print(f"Preparing inputs for sequence with request_id: {seq_state.request_id}")
                # Prepare model-specific inputs
                model_inputs = model_handler.prepare_inputs_for_generation(seq_state)
                model_inputs_list.append(model_inputs)
                request_ids.append(seq_state.request_id)
                max_lengths.append(seq_state.max_length)
                top_k_list.append(seq_state.top_k)
                # Collect past_key_values
                past_key_values_list.append(seq_state.past_key_values)

            # Combine model inputs for batching
            combined_inputs = model_handler.combine_inputs_for_batch(model_inputs_list)
            combined_past_key_values = model_handler.combine_past_key_values(past_key_values_list)

            print(f"Combined inputs for batch: {combined_inputs}")
            print(f"Combined past_key_values for batch: {combined_past_key_values}")

            # Generate next logits using the model's generate_next_logits method
            print("Generating next logits")
            next_logits, new_past_key_values = model_handler.generate_next_logits(
                combined_inputs,
                past_key_values=combined_past_key_values,  # Pass combined past_key_values
                top_k=top_k_list,  # Pass list of top_k values
            )
            print(f"Generated next logits with shape: {next_logits.shape}")
            print(f"New past_key_values obtained from model")

            print(f"Type of new_past_key_values: {type(new_past_key_values)}")
            print(f"Length of new_past_key_values: {len(new_past_key_values)}")
            print(f"Sample element: {new_past_key_values[0]}")

            # Split new_past_key_values and update seq_state
            print("Splitting new past_key_values...")
            new_past_key_values_list = model_handler.split_past_key_values(new_past_key_values)
            print(f"Split past_key_values into {len(new_past_key_values_list)} sequences")

            # Sample next tokens
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
            print(f"\n!Sampled next tokens: {next_tokens}\n")

            # Update each sequence in the batch
            finished_indices = []
            for i, seq_state in enumerate(batch_sequences):
                print(f"Updating sequence state for request_id: {seq_state.request_id}")
                token = next_tokens[i].unsqueeze(0)
                print(f"Appending token {token} to generated_tokens")
                seq_state.generated_tokens = torch.cat(
                    [seq_state.generated_tokens, token], dim=1
                )
                # Update attention mask
                seq_state.inputs['attention_mask'] = torch.cat(
                    [
                        seq_state.inputs['attention_mask'],
                        torch.ones((1, 1), device=seq_state.inputs['attention_mask'].device)
                    ],
                    dim=1
                )
                print(f"Updated attention mask for request_id: {seq_state.request_id}")
                # Update past_key_values for each sequence
                seq_state.past_key_values = new_past_key_values_list[i]
                print(f"Updated past_key_values for request_id: {seq_state.request_id}")
                # Let the model handler update any other required states
                model_handler.update_sequence_state(seq_state, token)
                print(f"Updated sequence state with new token for request_id: {seq_state.request_id}")

                # Check for end-of-sequence token or max length
                eos_token_id = model_handler.tokenizer.eos_token_id
                if (
                    token.item() == eos_token_id
                    or seq_state.generated_tokens.shape[1] >= seq_state.max_length
                ):
                    print(f"Sequence finished for request_id: {seq_state.request_id}")
                    # Decode the generated tokens
                    output_text = model_handler.decode_tokens(
                        seq_state.generated_tokens[0]
                    )
                    print(f"\n!Decoded output text for request_id {seq_state.request_id}: {output_text}")
                    message_value = json.dumps(
                        {
                            "type": "generation_response",
                            "request_id": seq_state.request_id,
                            "result": output_text,
                        }
                    ).encode("utf-8")
                    print(f"\n!Sending result for request_id {seq_state.request_id}")
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
                print(f"Removing finished sequence at index {index} (request_id: {batch_states[index].request_id})")
                del batch_states[index]

            if config.DEBUG:
                print("\nSummary of generated tokens in this iteration:")
                for seq_state in batch_sequences:
                    # Get the prompt text
                    prompt_text = seq_state.prompt_text

                    # Get all generated tokens excluding the prompt and the last token
                    if seq_state.generated_tokens.shape[1] > seq_state.prompt_length + 1:
                        previous_token_ids = seq_state.generated_tokens[0, seq_state.prompt_length:-1]
                        previous_tokens_text = model_handler.decode_tokens(previous_token_ids)
                        previous_tokens_list = previous_tokens_text.strip().split()
                    else:
                        previous_tokens_list = []

                    # Get the last generated token
                    last_token_id = seq_state.generated_tokens[0, -1].item()
                    last_token_tensor = torch.tensor(
                        [last_token_id], device=seq_state.generated_tokens.device
                    )
                    token_text = model_handler.decode_tokens(last_token_tensor).strip()

                    # Print the prompt, the previously generated tokens, and the latest token
                    print(f"For prompt '{prompt_text}' we have generated {previous_tokens_list} previously and now generated '{token_text}'\n")

            # Fill up the batch with new requests
            fill_batch(batch_states, requests_queue, model_handler, batch_size)
            print(f"Batch states after refill: {batch_states}")

            end_time = time.time()
            print(f"\n->Time taken for this iteration: {end_time - start_time} seconds")

            # Small sleep to yield control
            time.sleep(0.01)

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error during response generation: {e} -- {error_trace}")

            # Send error messages for all active sequences
            for seq_state in batch_states:
                error_message = json.dumps(
                    {
                        "type": "error",
                        "request_id": seq_state.request_id,
                        "error": str(e),
                    }
                ).encode("utf-8")
                producer.produce(
                    topic=config.OUTPUT_TOPIC,
                    value=error_message,
                    callback=delivery_report,
                )
                producer.poll(0)
            batch_states.clear()
            time.sleep(0.1)


def generate_responses() -> None:
    batch_states: List[SequenceState] = []
    print("Starting generate_responses()")
    while True:
        start_time = time.time()
        try:
            print("\n\n-----Entering generate_responses loop-----\n\n")
            # Fill up the batch with available requests
            fill_batch(batch_states, requests_queue, model_handler, config.BATCH_SIZE)
            print(f"Batch states after fill_batch: {batch_states}")

            if not batch_states:
                print("No batch states available, sleeping for 0.01 seconds.")
                time.sleep(0.01)
                continue

            # Separate sequences into initial and subsequent sequences
            initial_sequences = [seq_state for seq_state in batch_states if seq_state.past_key_values is None]
            subsequent_sequences = [seq_state for seq_state in batch_states if seq_state.past_key_values is not None]

            # Process initial sequences
            if initial_sequences:
                print("Processing initial sequences...")
                # Prepare inputs
                model_inputs_list = [model_handler.prepare_inputs_for_generation(seq_state) for seq_state in initial_sequences]
                combined_inputs = model_handler.combine_inputs_for_batch(model_inputs_list)

                # Generate next logits
                next_logits, new_past_key_values = model_handler.generate_next_logits(
                    combined_inputs,
                    past_key_values=None,
                )

                # Split past_key_values
                new_past_key_values_list = model_handler.split_past_key_values(new_past_key_values)

                # Sample next tokens
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # Shape: [batch_size, 1]

                for seq_state, token, past_key_values in zip(initial_sequences, next_tokens, new_past_key_values_list):
                    update_sequence_state(seq_state, token, past_key_values, model_handler)

            # Process subsequent sequences
            if subsequent_sequences:
                # Group sequences by past_key_values seq_len
                seq_len_to_sequences = defaultdict(list)
                for seq_state in subsequent_sequences:
                    seq_len = seq_state.past_key_values[0][0].shape[2]  # seq_len dimension
                    seq_len_to_sequences[seq_len].append(seq_state)

                for seq_len, sequences in seq_len_to_sequences.items():
                    print(f"Processing sequences with past_key_values seq_len = {seq_len}")

                    # Prepare inputs
                    model_inputs_list = [model_handler.prepare_inputs_for_generation(seq_state) for seq_state in sequences]
                    combined_inputs = model_handler.combine_inputs_for_batch(model_inputs_list)

                    # Combine past_key_values
                    past_key_values = model_handler.combine_past_key_values([seq_state.past_key_values for seq_state in sequences])

                    # Generate next logits
                    next_logits, new_past_key_values = model_handler.generate_next_logits(
                        combined_inputs,
                        past_key_values=past_key_values,
                    )

                    # Split past_key_values
                    new_past_key_values_list = model_handler.split_past_key_values(new_past_key_values)

                    # Sample next tokens
                    next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # Shape: [batch_size, 1]

                    for seq_state, token, past_key_values in zip(sequences, next_tokens, new_past_key_values_list):
                        update_sequence_state(seq_state, token, past_key_values, model_handler)

            # Remove finished sequences
            batch_states = [seq_state for seq_state in batch_states if not seq_state.is_finished]


            if config.DEBUG:
                print("\nSummary of generated tokens in this iteration:")
                for seq_state in batch_states:
                    # Get the prompt text
                    prompt_text = seq_state.prompt_text

                    # Get all generated tokens excluding the prompt
                    if seq_state.generated_tokens.shape[1] > seq_state.prompt_length:
                        generated_token_ids = seq_state.generated_tokens[0, seq_state.prompt_length:]
                        generated_tokens_text = model_handler.decode_tokens(generated_token_ids)
                        generated_tokens_list = generated_tokens_text.strip().split()
                    else:
                        generated_tokens_list = []

                    # Get the last generated token
                    if seq_state.generated_tokens.shape[1] > seq_state.prompt_length:
                        last_token_id = seq_state.generated_tokens[0, -1].unsqueeze(0)
                        token_text = model_handler.decode_tokens(last_token_id).strip()
                    else:
                        token_text = ""

                    # Print the prompt, the previously generated tokens, and the latest token
                    print(f"For prompt '{prompt_text}' we have generated {generated_tokens_list} previously and now generated '{token_text}'\n")


            # Fill up the batch with new requests
            fill_batch(batch_states, requests_queue, model_handler, config.BATCH_SIZE)
            print(f"Batch states after refill: {batch_states}")

            end_time = time.time()
            print(f"\n->Time taken for this iteration: {end_time - start_time} seconds")

            # Small sleep to yield control
            time.sleep(0.01)

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error during response generation: {e} -- {error_trace}")

            # Send error messages for all active sequences
            for seq_state in batch_states:
                error_message = json.dumps(
                    {
                        "type": "error",
                        "request_id": seq_state.request_id,
                        "error": str(e),
                    }
                ).encode("utf-8")
                producer.produce(
                    topic=config.OUTPUT_TOPIC,
                    value=error_message,
                    callback=delivery_report,
                )
                producer.poll(0)
            batch_states.clear()
            time.sleep(0.1)


def update_sequence_state(seq_state, token, new_past_key_values, model_handler):
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
    eos_token_id = model_handler.tokenizer.eos_token_id
    if (
        token.item() == eos_token_id
        or seq_state.generated_tokens.shape[1] >= seq_state.max_length
    ):
        print(f"Sequence finished for request_id: {seq_state.request_id}")
        # Decode only the generated response, excluding the prompt
        response_token_ids = seq_state.generated_tokens[:, seq_state.prompt_length:]
        output_text = model_handler.decode_tokens(
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
            topic=config.OUTPUT_TOPIC,
            value=message_value,
            callback=delivery_report,
        )
        # Poll to trigger delivery report callbacks
        producer.poll(0)
        seq_state.is_finished = True


def fill_batch(
    batch_states: List[SequenceState],
    requests_queue: Queue,
    model_handler: ModelHandler,
    batch_size: int
) -> None:
    while len(batch_states) < batch_size:
        try:
            request = requests_queue.get_nowait()
        except Empty:
            break
        try:
            # Extract request details
            request_id = request['request_id']
            messages = request['messages']

            # Convert messages to prompt text
            prompt_text = model_handler.convert_messages_to_prompt(messages)

            # Preprocess inputs
            inputs = model_handler.preprocess_inputs(prompt_text)
            generated_tokens = inputs["input_ids"]

            # Create new SequenceState with prompt_text
            new_state = SequenceState(
                inputs=inputs,
                generated_tokens=generated_tokens,
                request_id=request_id,
                max_length=request.get('max_tokens', model_handler.config.MAX_LENGTH),
                top_k=request.get('top_k', model_handler.config.TOP_K),
                device=model_handler.device,
                prompt_text=prompt_text
            )

            batch_states.append(new_state)
            print(f"Added SequenceState to batch_states for request_id: {request_id}")

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error processing inputs for request {request_id}: {e} -- {error_trace}")
            # Handle the error (e.g., send an error message back)
            pass
    print("Completed fill_batch()")

def consume_requests() -> None:
    print("Starting consume_requests()")
    while True:
        try:
            msg = consumer.poll(1.0)  # Poll for messages with a timeout
            if msg is None:
                print("No message received, sleeping for 0.01 seconds")
                time.sleep(0.01)
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            data: Dict[str, Any] = json.loads(msg.value().decode("utf-8"))
            print("Received data: ", data)
            base_request_id = f"{msg.topic()}-{msg.partition()}-{msg.offset()}"

            # Extract top-level parameters
            top_level_max_tokens = data.get("max_tokens", config.MAX_LENGTH)
            top_level_top_k = data.get("top_k", config.TOP_K)

            # Process the 'batch' field in the data
            if "batch" in data:
                print(f"Processing batch in message: {base_request_id}")
                for idx, message_item in enumerate(data["batch"]):
                    print(f"Processing batch item {idx}")

                    # Extract per-message parameters, fallback to top-level, then config
                    message_max_tokens = message_item.get("max_tokens", top_level_max_tokens)
                    message_top_k = message_item.get("top_k", top_level_top_k)

                    messages = []
                    if "content" in message_item:
                        print("Found 'content' in message_item")
                        # This format implies a single message
                        messages.append({
                            "role": message_item.get("role", "user"),
                            "content": message_item["content"]
                        })
                    elif "messages" in message_item:
                        print("Found 'messages' in message_item")
                        messages = message_item["messages"]
                    else:
                        print(f"Unknown message format in batch item: {message_item}")
                        continue

                    # Prepare the request data
                    request_data = {
                        "messages": messages,
                        "request_id": f"{base_request_id}-{idx}",
                        "max_tokens": message_max_tokens,
                        "top_k": message_top_k,
                    }
                    print(f"Appending request data to queue: {request_data}")
                    requests_queue.put(request_data)  # Use 'put' instead of 'append'
            else:
                print(f"No 'batch' field in message: {base_request_id}")

            # Manually commit the message
            consumer.commit(message=msg, asynchronous=False)
            print(f"Committed message {base_request_id}")
        except KafkaException as e:
            print(f"Kafka error: {e}")

            # Send error message back to output topic
            error_message = json.dumps(
                {
                    "type": "error",
                    "request_id": base_request_id,
                    "error": str(e),
                }
            ).encode("utf-8")
            print(f"Sending error message for request_id {base_request_id}")
            producer.produce(
                topic=config.OUTPUT_TOPIC,
                value=error_message,
                callback=delivery_report,
            )
            producer.poll(0)
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error processing message: {e} -- {error_trace}")

            # Send error message back to output topic
            error_message = json.dumps(
                {
                    "type": "error",
                    "request_id": base_request_id,
                    "error": str(e),
                    "traceback": error_trace,  # Include the stack trace
                }
            ).encode("utf-8")
            print(f"Sending error message for request_id {base_request_id}")
            producer.produce(
                topic=config.OUTPUT_TOPIC,
                value=error_message,
                callback=delivery_report,
            )
            producer.poll(0)

def main() -> None:
    print("Starting main()")
    consumer_thread = threading.Thread(target=consume_requests)
    generator_thread = threading.Thread(target=generate_responses)
    consumer_thread.start()
    generator_thread.start()
    consumer_thread.join()
    generator_thread.join()
    print("Completed main()")

try:
    print("Running main()")
    asyncio.run(main())
finally:
    print("Closing consumer and producer")
    # Clean up resources
    consumer.close()
    producer.flush()

