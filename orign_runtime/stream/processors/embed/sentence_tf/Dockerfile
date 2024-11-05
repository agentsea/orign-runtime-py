FROM huggingface/transformers-pytorch-gpu:latest

RUN pip install sentence-transformers orign-runtime

COPY . /app

WORKDIR /app

CMD ["python3", "main.py"]