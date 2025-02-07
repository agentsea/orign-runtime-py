FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && apt-get install -y git wget

# # Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # Run the installer then remove it
# RUN sh /uv-installer.sh && rm /uv-installer.sh

# # Ensure the installed binary is on the `PATH`
# ENV PATH="/root/.local/bin/:$PATH"

# uv venv --python 3.12.8
# source .venv/bin/activate
RUN pip install "git+https://github.com/ywang96/vllm@qwen2_5_vl"
RUN pip install "git+https://github.com/huggingface/transformers"

COPY . /app

WORKDIR /app

RUN pip3 install qwen-vl-utils
RUN pip3 install orign-runtime # $(date +%s)

# This oddly fails on L40s boxes on EC2
# ENTRYPOINT ["sh", "-c", "pip3 install flash-attn --no-build-isolation || echo 'flash-attn install failed, continuing...' && python3 main.py"]
ENTRYPOINT ["python3", "main.py"]