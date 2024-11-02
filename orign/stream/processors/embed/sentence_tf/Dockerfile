FROM huggingface/transformers-pytorch-gpu:latest

RUN pip install sentence-transformers
RUN pip install orign==0.1.2

COPY . /app

WORKDIR /app

CMD ["python3", "main.py"]