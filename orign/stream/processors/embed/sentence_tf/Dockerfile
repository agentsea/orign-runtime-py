FROM huggingface/transformers-pytorch-gpu:latest

RUN pip install sentence-transformers
COPY . /app

WORKDIR /app

CMD ["python", "main.py"]