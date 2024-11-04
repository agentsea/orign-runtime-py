FROM thehale/python-poetry:1.7.1-py3.12-slim

# Copy only the Poetry configuration files
COPY pyproject.toml poetry.lock /app/
WORKDIR /app

# Install dependencies
RUN poetry install --no-root

# Now copy the rest of the application files
COPY . /app

# Expose the port and specify the command
EXPOSE 8080
CMD ["poetry", "run", "python", "-m", "orign_runtime.stream.processors.chat.vllm.main"]
