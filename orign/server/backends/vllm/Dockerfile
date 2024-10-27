FROM thehale/python-poetry:1.7.1-py3.12-slim

COPY . /app
WORKDIR /app

RUN poetry install

EXPOSE 8080

CMD ["poetry", "run", "python", "-m", "orign.server.backends.vllm.main"]
