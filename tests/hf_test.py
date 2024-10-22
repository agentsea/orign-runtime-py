# test_main.py
import json
import time
import subprocess
import sys
import signal
import os

import pytest
from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
INPUT_TOPIC = "qwen"
OUTPUT_TOPIC = "qwen-results"
GROUP_ID = "test_consumer_group"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Kafka configuration dictionaries
producer_conf = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
}
consumer_conf = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
    "group.id": GROUP_ID,
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}
admin_conf = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
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
    # Start the main.py script as a subprocess
    # Adjust the path to main.py according to your project structure

    env_vars = os.environ.copy()
    # Add or modify specific environment variables
    env_vars["KAFKA_INPUT_TOPIC"] = INPUT_TOPIC
    env_vars["KAFKA_OUTPUT_TOPIC"] = OUTPUT_TOPIC
    env_vars["KAFKA_GROUP_ID"] = GROUP_ID
    env_vars["HF_MODEL_NAME"] = MODEL

    process = subprocess.Popen(
        [sys.executable, "-m", "orign.backends.hf.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_vars,
    )

    # Wait for a short time to ensure the process starts
    time.sleep(5)

    yield process

    # Terminate the process after tests
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    stdout, stderr = process.communicate()
    print("Main process output:")
    print(stdout.decode())
    print(stderr.decode())


def clear_topic(consumer: Consumer, topics: list, timeout: int = 10) -> None:
    """Consume and discard all messages from the provided topics to clear them."""
    for topic in topics:
        print(f"Clearing topic {topic}...", flush=True)
        consumer.subscribe([topic])
        end_time = time.time() + timeout
        while time.time() < end_time:
            msg = consumer.poll()
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error while clearing topic {topic}: {msg.error()}")
                continue
            print(
                f"Cleared message from {topic}: {msg.value().decode('utf-8')}",
                flush=True,
            )


def test_main(start_main_process):
    # Create Producer and Consumer instances
    producer = Producer(producer_conf)
    consumer = Consumer(consumer_conf)
    consumer.subscribe([OUTPUT_TOPIC])

    clear_topic(consumer, [INPUT_TOPIC, OUTPUT_TOPIC])

    # Produce test messages
    num_messages = 2
    for i in range(num_messages):
        msg_dict = {
            "model": MODEL,
            "batch": [
                # {
                #     "role": "user",
                #     "content": [
                #         {"type": "text", "text": "What’s in this image?"},
                #         {
                #             "type": "image_url",
                #             "image_url": {
                #                 "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                #             },
                #         },
                #     ],
                # },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Think step by step, whats the capitol of France?",
                        }
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "What's the capitol of the USA?"}
                    ]
                },
            ],
            "max_tokens": 50,
        }
        msg = json.dumps(msg_dict)
        producer.produce(INPUT_TOPIC, msg.encode("utf-8"))

    # Flush producer to ensure all messages are sent
    producer.flush()

    # Consume and verify output messages
    received_messages = 0
    expected_messages = num_messages * 2  # Assuming two responses per input message
    timeout = time.time() + 120  # Timeout after 120 seconds
    output_results = []

    while received_messages < expected_messages and time.time() < timeout:
        print("polling consumer", flush=True)
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        print("Received raw message: ", msg.value(), flush=True)
        output_data = json.loads(msg.value().decode("utf-8"))
        print(f"Received output message: {output_data}", flush=True)

        # Collect results for validation
        output_results.append(output_data)

        # Perform basic validation
        assert "request_id" in output_data
        assert "result" in output_data
        assert isinstance(output_data["result"], str)

        received_messages += 1

    consumer.close()

    # Verify that we received the expected number of messages
    assert (
        received_messages == expected_messages
    ), f"Expected {expected_messages} messages, got {received_messages}"

    # Additional validation can be done here based on expected results
