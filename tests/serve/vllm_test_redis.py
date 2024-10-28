import json
import time
import subprocess
import sys
import os
import threading
import pytest
from colorama import init, Fore, Style
import redis

# Redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
INPUT_STREAM = "Qwen/Qwen2.5-0.5B-Instruct"
USER_EMAIL = "tom@myspace.com"
OUTPUT_STREAM = f"results:{USER_EMAIL}:{INPUT_STREAM}"
GROUP_NAME = "test_consumer_group_vllm"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Redis configuration dictionary
redis_conf = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
}

@pytest.fixture(scope="module", autouse=True)
def setup_redis_streams():
    # Connect to Redis
    r = redis.Redis(**redis_conf)
    
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

    yield  # Proceed with tests

    # Optional: Clean up streams after tests
    # r.delete(INPUT_STREAM)
    # r.delete(OUTPUT_STREAM)


def test_main():
    # Connect to Redis
    r = redis.Redis(**redis_conf)

    # Produce test messages to the input stream
    num_messages = 2
    for i in range(num_messages):
        msg_dict = {
            "model": MODEL,
            "batch": [
                # {
                #     "messages": [
                #         {
                #             "role": "user",
                #             "content": "What’s in this image?",
                #         },
                #         {
                #             "role": "user",
                #             "content": {
                #                 "type": "image_url",
                #                 "image_url": {
                #                     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                #                 },
                #             },
                #         },
                #     ]
                # },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Think step by step, what's the capital of France?",
                        }
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What's the capital of Germany?"
                        }
                    ]
                },
            ],
            "max_tokens": 50,
            "sampling_params": {
                "n": 4,
                "top_k": 5,
                "logprobs": 5,
            },
            "request_id": str(i),
            "output_topic": OUTPUT_STREAM,
        }
        msg = json.dumps(msg_dict)
        r.xadd(INPUT_STREAM, {'payload': msg})

    # Consume and verify output messages from the output stream
    received_messages = 0
    expected_messages = num_messages * 2  # Adjust if necessary
    timeout = time.time() + 400  # Timeout after 400 seconds
    output_results = []

    while received_messages < expected_messages:
        print("Polling Redis consumer", flush=True)
        try:
            messages = r.xreadgroup(GROUP_NAME, "test_consumer", {OUTPUT_STREAM: '>'}, count=1, block=1000)
        except redis.exceptions.ResponseError as e:
            print(f"Redis error: {e}")
            continue

        if time.time() > timeout:
            print("Test timed out")
            pytest.fail("Test timed out")
            break

        print("Messages expected: ", expected_messages, "Messages received: ", received_messages)
        if not messages:
            continue

        for stream_name, entries in messages:
            for msg_id, msg_data in entries:
                output_data = json.loads(msg_data['payload'])
                print(f"Received output message: {output_data}", flush=True)

                # Check for error messages
                if output_data.get("type") == "ErrorResponse":
                    pytest.fail(f"Test failed due to error message: {output_data}")

                # Collect results for validation
                output_results.append(output_data)

                # Perform basic validation for generation responses
                if output_data.get("type") == "ChatResponse":
                    assert "request_id" in output_data
                    assert "choices" in output_data

                    received_messages += 1
                    print(f"Received message {received_messages} of {expected_messages}")
                else:
                    print(f"Received unexpected message type: {output_data.get('type')}")

                # Acknowledge the message
                r.xack(OUTPUT_STREAM, GROUP_NAME, msg_id)

        print(f"Time: {time.time()}, Timeout: {timeout}")
        if time.time() > timeout:
            print("Test timed out")
            pytest.fail("Test timed out")
            break

    print("Closing Redis consumer... total received messages: ", received_messages, " expected messages: ", expected_messages, flush=True)
    time.sleep(10)

    print("\n-----Results: ", flush=True)
    for i, result in enumerate(output_results):
        print(f"\nResult {i}: {result}")

    # Verify that we received the expected number of messages
    assert (
        received_messages == expected_messages
    ), f"Expected {expected_messages} messages, got {received_messages}"

    print("\n\nValidation passed! Ignore any further errors.\n")