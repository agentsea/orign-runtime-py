FROM vllm/vllm-openai:latest

COPY . /app

WORKDIR /app

RUN pip3 install orign-runtime # $(date +%s)

ENTRYPOINT ["python3", "main.py"]