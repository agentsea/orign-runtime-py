FROM litellm/litellm:latest

COPY . /app

WORKDIR /app

RUN pip3 install orign-runtime # $(date +%s)

CMD ["python3", "main.py"]
