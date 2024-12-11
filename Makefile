
.PHONY: test
test:
	poetry run pytest -s -vvv

.PHONY: test-molmo
test-molmo:
	poetry run pytest -s -vvv tests/stream/hf_test_molmo.py

.PHONY: test-qwen
test-qwen:
	poetry run pytest -s -vvv tests/stream/hf_test_qwen.py

.PHONY: test-vllm
test-vllm:
	poetry run pytest -s -vvv tests/stream/vllm_test.py

.PHONY: test-vllm-redis-qwen
test-vllm-redis-qwen:
	poetry run pytest -s -vvv tests/stream/vllm_test_redis_qwen.py

.PHONY: test-vllm-redis
test-vllm-redis-molmo:
	poetry run pytest -s -vvv tests/stream/vllm_test_redis_molmo.py

.PHONY: test-vllm-redis-stream
test-vllm-redis-stream:
	poetry run pytest -s -vvv tests/stream/vllm_test_redis_stream.py

.PHONY: serve-molmo
serve-molmo:
	REDIS_HOST="localhost" REDIS_PORT=6379 \
	USER_EMAIL="tom@myspace.com" OUTPUT_STREAM="chat_results:tom@myspace.com:allenai/Molmo-7B-D-0924-$(date +%s)" \
	GROUP_NAME="test_consumer_group_vllm" \
	QUEUE_TYPE="redis" \
	QUEUE_INPUT_TOPICS="test1,allenai/Molmo-7B-D-0924-$(date +%s)" QUEUE_GROUP_ID="test_consumer_group_vllm" MODEL_NAME="allenai/Molmo-7B-D-0924" \
	MODEL_TYPE="molmo" DEVICE="cuda" DEBUG="true" ACCEPTS="text,image" \
	poetry run python -m orign_runtime.stream.processors.chat.vllm.main

.PHONY: test-easyocr
test-easyocr:
	poetry run pytest -s -vvv tests/stream/easyocr_test.py

.PHONY: test-doctr
test-doctr:
	poetry run pytest -s -vvv tests/stream/doctr_test.py

.PHONY: test-sentence-tf
test-sentence-tf:
	poetry run pytest -s -vvv tests/stream/sentence_tf_test.py

.PHONY: test-litellm
test-litellm:
	poetry run pytest -s -vvv tests/stream/litellm_test.py

.PHONY: build-vllm
build-vllm:
	docker build --platform=linux/amd64 -f ./orign_runtime/stream/processors/chat/vllm/Dockerfile -t orign-vllm .

.PHONY: run-redis
run-redis:
	docker run -d --name redis -p 6379:6379 redis:latest
