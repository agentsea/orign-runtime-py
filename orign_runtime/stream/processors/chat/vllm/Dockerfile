# FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && apt-get install -y git wget

# RUN pip install "git+https://github.com/ywang96/vllm@qwen2_5_vl"


COPY . /app

WORKDIR /app

ARG CACHE_BUST=unknown
RUN pip3 install qwen-vl-utils
RUN pip3 install orign-runtime # $(date +%s)

# Install pre-built vLLM to get compiled CUDA code
RUN pip install vllm==0.7.3
# RUN pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl


# RUN pip install "git+https://github.com/huggingface/transformers"
RUN pip install transformers==4.49.0

# This oddly fails on L40s boxes on EC2
# ENTRYPOINT ["sh", "-c", "pip3 install flash-attn --no-build-isolation || echo 'flash-attn install failed, continuing...' && python3 main.py"]
ENTRYPOINT ["python3", "main.py"]