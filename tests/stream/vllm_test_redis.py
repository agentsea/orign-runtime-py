import json
import time
import subprocess
import sys
import os
import pytest
import redis
import time

# Redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
INPUT_STREAM = f"test1,allenai/Molmo-7B-D-0924-{int(time.time())}"
USER_EMAIL = "tom@myspace.com"
OUTPUT_STREAM = f"chat_results:{USER_EMAIL}:{INPUT_STREAM}-{int(time.time())}"
GROUP_NAME = "test_consumer_group_vllm"
MODEL = "allenai/Molmo-7B-D-0924"
INPUT_STREAM_GOOD = INPUT_STREAM.split(",")[1]

# Redis configuration dictionary
redis_conf = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
}

@pytest.fixture(scope="module", autouse=True)
def setup_redis_streams():
    # Connect to Redis
    r = redis.Redis(**redis_conf)
    
    try:
        r.delete(INPUT_STREAM_GOOD)
        r.delete(OUTPUT_STREAM)
    except:
        pass

    # Create input and output streams with consumer groups
    try:
        r.xgroup_create(INPUT_STREAM_GOOD, GROUP_NAME, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if 'BUSYGROUP' in str(e):
            print(f"Consumer group {GROUP_NAME} already exists for stream {INPUT_STREAM_GOOD}")
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
    try:
        r.delete(INPUT_STREAM_GOOD)
        r.delete(OUTPUT_STREAM)
    except:
        pass

@pytest.fixture(scope="module")
def start_main_process():
    import threading
    from colorama import init, Fore, Style

    # Initialize colorama
    init(autoreset=True)

    # Start the main.py script as a subprocess
    env_vars = os.environ.copy()
    env_vars["QUEUE_TYPE"] = "redis"
    env_vars["QUEUE_INPUT_TOPICS"] = INPUT_STREAM
    env_vars["QUEUE_GROUP_ID"] = GROUP_NAME
    env_vars["MODEL_NAME"] = MODEL
    env_vars["DEVICE"] = "cuda"
    env_vars["DEBUG"] = "true"
    env_vars["VLLM_DISABLE_PROMETHEUS"] = "true"
    env_vars["MAX_IMAGES_PER_PROMPT"] = "1"
    env_vars["ACCEPTS"] = "text,image"
    
    process = subprocess.Popen(
        [sys.executable, "-m", "orign_runtime.stream.processors.chat.vllm.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_vars,
        bufsize=1,  # Line-buffered
        universal_newlines=True,  # Text mode
    )

    # Function to read and print stdout
    def read_stdout():
        while True:
            line = process.stdout.readline()
            if line:
                print(f"{Fore.GREEN}[Server STDOUT]{Style.RESET_ALL} {line}", end='')
            else:
                if process.poll() is not None:
                    break

    # Function to read and print stderr
    def read_stderr():
        while True:
            line = process.stderr.readline()
            if line:
                print(f"{Fore.RED}[Server STDERR]{Style.RESET_ALL} {line}", end='')
            else:
                if process.poll() is not None:
                    break

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

    print("Main process output captured.")


def test_main(start_main_process):
    # Connect to Redis
    r = redis.Redis(**redis_conf)

    # Produce test messages to the input stream
    num_messages = 2
    for i in range(num_messages):
        msg_dict = {
            "model": MODEL,
            "batch": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What’s in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                                    },
                                },
                            ],
                        },
                    ],
                },
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
        r.xadd(INPUT_STREAM_GOOD, {'message': msg})

    # Consume and verify output messages from the output stream
    received_messages = 0
    expected_messages = num_messages * 3  # Adjust if necessary
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
                payload = msg_data[b'message'].decode('utf-8')
                print(f"Received message payload: {payload}", flush=True)
                output_data = json.loads(payload)
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

    print("\n====\nResults\n====\n ", flush=True)
    for i, result in enumerate(output_results):
        print(f"\n----Result {i}\n", flush=True)
        for j, choice in enumerate(result.get("choices", [])):
            print(f"\n►Choice {j}: {choice.get('text')}")

    # Verify that we received the expected number of messages
    assert (
        received_messages == expected_messages
    ), f"Expected {expected_messages} messages, got {received_messages}"

    print("\n\nValidation passed! Ignore any further errors.\n")