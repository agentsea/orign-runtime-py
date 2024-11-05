FROM vllm/vllm-openai:latest

RUN pip3 install orign-runtime

COPY . /app

WORKDIR /app

ENTRYPOINT ["python3", "main.py"]