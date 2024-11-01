
.PHONY: test
test:
	poetry run pytest -s -vvv

.PHONY: test-molmo
test-molmo:
	poetry run pytest -s -vvv tests/serve/hf_test_molmo.py

.PHONY: test-qwen
test-qwen:
	poetry run pytest -s -vvv tests/serve/hf_test_qwen.py

.PHONY: test-vllm
test-vllm:
	poetry run pytest -s -vvv tests/serve/vllm_test.py

.PHONY: test-vllm-redis
test-vllm-redis:
	poetry run pytest -s -vvv tests/serve/vllm_test_redis.py

.PHONY: test-vllm-redis-stream
test-vllm-redis-stream:
	poetry run pytest -s -vvv tests/serve/vllm_test_redis_stream.py

.PHONY: build-vllm
build-vllm:
	docker build --platform=linux/amd64 -f ./orign/server/backends/vllm/Dockerfile -t orign-vllm .

.PHONY: run-redis
run-redis:
	docker run -d --name redis -p 6379:6379 redis:latest
