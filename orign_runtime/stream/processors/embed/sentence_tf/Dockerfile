FROM huggingface/transformers-pytorch-gpu:latest

COPY . /app

WORKDIR /app

RUN pip install sentence-transformers orign-runtime # $(date +%s)

CMD ["python3", "main.py"]