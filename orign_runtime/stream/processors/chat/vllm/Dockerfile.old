FROM vllm/vllm-openai:v0.6.4

COPY . /app

WORKDIR /app

RUN pip3 install orign-runtime # $(date +%s)

# This oddly fails on L40s boxes on EC2
# ENTRYPOINT ["sh", "-c", "pip3 install flash-attn --no-build-isolation || echo 'flash-attn install failed, continuing...' && python3 main.py"]
ENTRYPOINT ["python3", "main.py"]