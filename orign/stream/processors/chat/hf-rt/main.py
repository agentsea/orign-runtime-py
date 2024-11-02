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
from .util import delivery_report

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
            top_k_list = [seq_state.top_k for seq_state in batch_states]

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
                    top_k=top_k_list,
                )

                # Split past_key_values
                new_past_key_values_list = model_handler.split_past_key_values(new_past_key_values)

                # Sample next tokens
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # Shape: [batch_size, 1]

                for seq_state, token, past_key_values in zip(initial_sequences, next_tokens, new_past_key_values_list):
                    model_handler.update_sequence_state(seq_state, token, past_key_values, producer)

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
                        top_k=top_k_list,
                    )

                    # Split past_key_values
                    new_past_key_values_list = model_handler.split_past_key_values(new_past_key_values)

                    # Sample next tokens
                    next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # Shape: [batch_size, 1]

                    for seq_state, token, past_key_values in zip(sequences, next_tokens, new_past_key_values_list):
                        model_handler.update_sequence_state(seq_state, token, past_key_values, producer)

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

            # Preprocess inputs using the model handler
            preprocessed_input = model_handler.preprocess_inputs(messages)
            inputs = preprocessed_input.inputs
            prompt_length = preprocessed_input.prompt_length
            prompt_text = preprocessed_input.prompt_text
            generated_tokens = inputs["input_ids"]

            # Create new SequenceState with prompt_text
            new_state = SequenceState(
                inputs=inputs,
                generated_tokens=generated_tokens,
                request_id=request_id,
                max_length=request.get('max_tokens', model_handler.config.MAX_LENGTH),
                top_k=request.get('top_k', model_handler.config.TOP_K),
                device=model_handler.device,
                prompt_text=prompt_text,
                prompt_length=prompt_length
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
    main()
finally:
    print("Closing consumer and producer")
    # Clean up resources
    consumer.close()
    producer.flush()

