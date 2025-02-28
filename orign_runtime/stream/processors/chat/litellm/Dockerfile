FROM litellm/litellm:latest

COPY . /app

WORKDIR /app

ARG CACHE_BUST=unknown
RUN pip3 install orign-runtime # $(date +%s)

ENTRYPOINT []

CMD ["python3", "main.py"]
