# test_main.py
import json
import time
import subprocess
import sys
import os
import pytest

from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic

# Kafka configuration
QUEUE_BOOTSTRAP_SERVERS = "localhost:9092"
INPUT_TOPIC = f"vllm-{int(time.time())}"
OUTPUT_TOPIC = f"{INPUT_TOPIC}-results"
GROUP_ID = "test_consumer_group_molmo"
MODEL = "allenai/Molmo-7B-D-0924"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Kafka configuration dictionaries
producer_conf = {
    "bootstrap.servers": QUEUE_BOOTSTRAP_SERVERS,
}
consumer_conf = {
    "bootstrap.servers": QUEUE_BOOTSTRAP_SERVERS,
    "group.id": f"{GROUP_ID}-test-{int(time.time())}",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}
admin_conf = {
    "bootstrap.servers": QUEUE_BOOTSTRAP_SERVERS,
}


@pytest.fixture(scope="module", autouse=True)
def setup_kafka_topics():
    # Create Kafka topics
    admin_client = AdminClient(admin_conf)
    topic_list = [
        NewTopic(INPUT_TOPIC, num_partitions=1, replication_factor=1),
        NewTopic(OUTPUT_TOPIC, num_partitions=1, replication_factor=1),
    ]

    # Create the topics
    fs = admin_client.create_topics(topic_list)

    # Wait for each operation to finish.
    for topic, f in fs.items():
        try:
            f.result()  # The result() will raise an exception if the topic creation failed
            print(f"Topic {topic} created successfully")
        except Exception as e:
            if "Topic already exists" in str(e):
                print(f"Topic {topic} already exists")
            else:
                print(f"Failed to create topic {topic}: {e}")

    yield

    # Optionally, clean up topics after tests
    # admin_client.delete_topics([INPUT_TOPIC, OUTPUT_TOPIC])


@pytest.fixture(scope="module")
def start_main_process():
    import threading
    from colorama import init, Fore, Style

    # Initialize colorama
    init(autoreset=True)

    # Start the main.py script as a subprocess
    env_vars = os.environ.copy()
    env_vars["QUEUE_INPUT_TOPIC"] = INPUT_TOPIC
    env_vars["QUEUE_OUTPUT_TOPIC"] = OUTPUT_TOPIC
    env_vars["QUEUE_GROUP_ID"] = GROUP_ID
    env_vars["HF_MODEL_NAME"] = MODEL
    env_vars["DEBUG"] = "true"

    process = subprocess.Popen(
        [sys.executable, "-m", "orign.server.backends.vllm.main"],
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


def clear_topic(consumer: Consumer, topics: list, timeout: int = 10) -> None:
    """Consume and discard all messages from the provided topics to clear them."""
    print(f"Clearing topics {topics}...", flush=True)
    consumer.subscribe(topics)
    end_time = time.time() + timeout
    while time.time() < end_time:
        msg = consumer.poll(1.0)  # Wait for 1 second
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error while clearing topics: {msg.error()}")
            continue
        print(
            f"Cleared message from {msg.topic()}: {msg.value().decode('utf-8')}",
            flush=True,
        )


def test_main(start_main_process):
    # Create Producer and Consumer instances
    producer = Producer(producer_conf)
    consumer = Consumer(consumer_conf)
    consumer.subscribe([OUTPUT_TOPIC])

    # clear_topic(consumer, [INPUT_TOPIC, OUTPUT_TOPIC])

    # Produce test messages
    num_messages = 2
    for i in range(num_messages):
        msg_dict = {
            "model": MODEL,
            "batch": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What’s in this image?",
                        },
                        {
                            "role": "user",
                            "content": {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                                },
                            },
                        },
                    ]
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
        }
        msg = json.dumps(msg_dict)
        producer.produce(INPUT_TOPIC, msg.encode("utf-8"))

    # Flush producer to ensure all messages are sent
    producer.flush()

    # Consume and verify output messages
    received_messages = 0
    expected_messages = num_messages * 2  # Adjust if necessary
    timeout = time.time() + 400  # Timeout after 120 seconds
    output_results = []

    while received_messages < expected_messages:
        print("Polling consumer", flush=True)
        msg = consumer.poll(1.0)
        if time.time() > timeout:
            print("Test timed out")
            pytest.fail("Test timed out")
            break

        print("messages expected: ", expected_messages, "messages received: ", received_messages)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        # print("Received raw message: ", msg.value(), flush=True)
        output_data = json.loads(msg.value().decode("utf-8"))
        print(f"Received output message: {output_data}", flush=True)

        # **Add this check for error messages**
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

        print(f"Time: {time.time()}, Timeout: {timeout}")
        if time.time() > timeout:
            print("Test timed out")
            pytest.fail("Test timed out")
            break
    
    print("Closing consumer... total recieved messages: ", received_messages, " expected messages: ", expected_messages, flush=True)
    time.sleep(10)
    consumer.close()

    print("\n-----Results: ", flush=True)
    for i, result in enumerate(output_results):
        print(f"\nResult {i}: {result}")

    # Verify that we received the expected number of messages
    assert (
        received_messages == expected_messages
    ), f"Expected {expected_messages} messages, got {received_messages}"

    print("\n\nValidation passed! ignore any further errors\n")

