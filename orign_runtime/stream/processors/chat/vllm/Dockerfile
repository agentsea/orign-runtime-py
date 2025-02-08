# FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && apt-get install -y git wget

# RUN pip install "git+https://github.com/ywang96/vllm@qwen2_5_vl"
RUN pip install "git+https://github.com/huggingface/transformers"

# Install pre-built vLLM to get compiled CUDA code
RUN pip install --pre --extra-index-url https://wheels.vllm.ai/nightly vllm

# Clone your specific fork and branch
RUN git clone --branch qwen2_5_vl https://github.com/ywang96/vllm.git /vllm_custom

# Replace only the Python code from the custom fork
RUN cp -r /vllm_custom/vllm /opt/conda/lib/python3.11/site-packages/

COPY . /app

WORKDIR /app

RUN pip3 install qwen-vl-utils
RUN pip3 install orign-runtime # $(date +%s)

# This oddly fails on L40s boxes on EC2
# ENTRYPOINT ["sh", "-c", "pip3 install flash-attn --no-build-isolation || echo 'flash-attn install failed, continuing...' && python3 main.py"]
ENTRYPOINT ["python3", "main.py"]