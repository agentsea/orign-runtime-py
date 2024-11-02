# tests/serve/easyocr_test_redis.py

import json
import time
import subprocess
import sys
import os
import pytest
import redis
import threading
import base64
import requests

# Redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
INPUT_STREAM = f"test_emb_input_stream_{int(time.time())}"
OUTPUT_STREAM = f"test_emb_output_stream_{int(time.time())}"
GROUP_NAME = "test_consumer_group_emb"
MODEL_BACKEND = "sentence_tf"
MODEL = "clip-ViT-B-32"

# Redis configuration dictionary
redis_conf = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
}

@pytest.fixture(scope="module", autouse=True)
def setup_redis_streams():
    # Connect to Redis
    r = redis.Redis(**redis_conf)

    # Delete streams if they already exist
    try:
        r.delete(INPUT_STREAM)
        r.delete(OUTPUT_STREAM)
    except:
        pass

    # Create input and output streams with consumer groups
    try:
        r.xgroup_create(INPUT_STREAM, GROUP_NAME, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if 'BUSYGROUP' in str(e):
            print(f"Consumer group {GROUP_NAME} already exists for stream {INPUT_STREAM}")
        else:
            raise

    try:
        r.xgroup_create(OUTPUT_STREAM, GROUP_NAME, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if 'BUSYGROUP' in str(e):
            print(f"Consumer group {GROUP_NAME} already exists for stream {OUTPUT_STREAM}")
        else:
            raise

    yield

    # Clean up streams after tests
    try:
        r.delete(INPUT_STREAM)
        r.delete(OUTPUT_STREAM)
    except:
        pass

@pytest.fixture(scope="module")
def start_main_process():
    # Start the EasyOCR backend as a subprocess
    env_vars = os.environ.copy()
    env_vars["QUEUE_TYPE"] = "redis"
    env_vars["QUEUE_INPUT_TOPICS"] = INPUT_STREAM
    env_vars["QUEUE_GROUP_ID"] = GROUP_NAME
    env_vars["MODEL_BACKEND"] = MODEL_BACKEND
    env_vars["MODEL"] = MODEL
    env_vars["DEVICE"] = "cuda"
    env_vars["DEBUG"] = "true"
    env_vars["QUEUE_OUTPUT_TOPIC"] = OUTPUT_STREAM

    process = subprocess.Popen(
        [sys.executable, "-m", "orign.stream.processors.embed.sentence_tf.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_vars,
        bufsize=1,
        universal_newlines=True,
    )

    # Function to read and print stdout
    def read_stdout():
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[Server STDOUT] {line}", end='')

    # Function to read and print stderr
    def read_stderr():
        for line in iter(process.stderr.readline, ''):
            if line:
                print(f"[Server STDERR] {line}", end='')

    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    # Wait for a short time to ensure the process starts
    time.sleep(5)

    yield process

    # Terminate the process after tests
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()

    # Ensure all output has been read
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

def test_sentence_tf_backend(start_main_process):
    # Connect to Redis
    r = redis.Redis(**redis_conf)

    # Prepare test image by downloading from a URL and encoding as base64
    image_url = 'https://docelf.com/images/docelf_free_receipt_template.png'
    response = requests.get(image_url)
    if response.status_code == 200:
        image_bytes = response.content
    else:
        pytest.fail(f"Failed to download image from {image_url}")

    try:
        b64_data = base64.b64encode(image_bytes).decode('utf-8')
        image_base64 = f"data:image/png;base64,{b64_data}"
        if not b64_data:
            pytest.fail("Failed to encode image to base64")
    except Exception as e:
        pytest.fail(f"Error encoding image to base64: {str(e)}")

    # Create test Embedding requests
    num_messages = 4
    messages = [
        # Test with text input
        {
            "type": "EmbeddingRequest",
            "request_id": "1",
            "text": "The quick brown fox jumps over the lazy dog.",
            "model": MODEL,
            "output_topic": OUTPUT_STREAM
        },
        # Test with base64-encoded image
        {
            "type": "EmbeddingRequest",
            "request_id": "2",
            "image": image_base64,
            "model": MODEL,
            "output_topic": OUTPUT_STREAM
        },
        # Test with image URL
        {
            "type": "EmbeddingRequest",
            "request_id": "3",
            "image": image_url,
            "model": MODEL,
            "output_topic": OUTPUT_STREAM
        },
        # Test with both text and image
        {
            "type": "EmbeddingRequest",
            "request_id": "4",
            "text": "The quick brown fox jumps over the lazy dog.",
            "image": image_url,
            "model": MODEL,
            "output_topic": OUTPUT_STREAM
        }
    ]

    for msg_dict in messages:
        msg = json.dumps(msg_dict)
        r.xadd(INPUT_STREAM, {'message': msg})
    print(f"Added {num_messages} messages to {INPUT_STREAM}", flush=True)

    # Consume and verify output messages from the output stream
    received_messages = 0
    expected_messages = num_messages
    timeout = time.time() + 60  # Timeout after 60 seconds
    output_results = []

    while received_messages < expected_messages:
        try:
            messages = r.xreadgroup(GROUP_NAME, "test_consumer", {OUTPUT_STREAM: '>'}, count=1, block=1000)
        except redis.exceptions.ResponseError as e:
            print(f"Redis error: {e}")
            continue

        if time.time() > timeout:
            print("Test timed out")
            pytest.fail("Test timed out")
            break

        if not messages:
            continue

        for stream_name, entries in messages:
            for msg_id, msg_data in entries:
                payload = msg_data[b'message'].decode('utf-8')
                output_data = json.loads(payload)
                print(f"\nReceived message: {output_data}")
                
                # Check for error messages
                if output_data.get("type") == "ErrorResponse":
                    pytest.fail(f"Test failed due to error message: {output_data}")

                # Collect results for validation
                output_results.append(output_data)

                # Perform basic validation for Embedding responses
                if output_data.get("type") == "EmbeddingResponse":
                    assert "request_id" in output_data
                    assert "data" in output_data
                    received_messages += 1
                else:
                    print(f"Received unexpected message type: {output_data.get('type')}")

                # Acknowledge the message
                r.xack(OUTPUT_STREAM, GROUP_NAME, msg_id)

        if time.time() > timeout:
            print("Test timed out")
            pytest.fail("Test timed out")
            break

    # Verify that we received the expected number of messages
    assert (
        received_messages == expected_messages
    ), f"Expected {expected_messages} messages, got {received_messages}"

    # Optional: Verify the content of the OCRResponse
    for output_data in output_results:
        if output_data.get("type") == "EmbeddingResponse":
            print("\n----\nEmbeddingResponse:\n")
            print(output_data)
            print("\n----\n")

    print("Test passed successfully.")