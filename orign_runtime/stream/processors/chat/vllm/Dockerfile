FROM vllm/vllm-openai:latest

COPY . /app

WORKDIR /app

RUN pip3 install orign-runtime # $(date +%s)

ENTRYPOINT ["sh", "-c", "pip3 install flash-attn --no-build-isolation && python3 main.py"]