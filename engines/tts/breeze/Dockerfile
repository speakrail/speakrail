# Speakrail TTS: Breeze TTS 2 from the speakrail fork (int8 weights, slim CUDA-graph profile, disconnect fix).
#
#   docker build -t speakrail-tts engines/tts/breeze
#   docker run --gpus all -v ./models:/models -p 127.0.0.1:7860:7860 speakrail-tts
#
# The Breeze weights and everything they generate are for research and non-commercial use only (BreezeBlue license).

ARG BASE_IMAGE=docker.io/pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

# branch "speakrail" of the fork = upstream 008f769 + one commit
ARG BREEZE_REPO=https://github.com/speakrail/breeze-tts.git
ARG BREEZE_COMMIT=be25971d96612319f7b65c94f92e0a653fdae239

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# gcc: torch.compile (Triton) builds small launcher modules at run time; sox: qwen-tts expects it
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl git libsndfile1 sox && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/breeze
RUN git clone "$BREEZE_REPO" . && git checkout "$BREEZE_COMMIT"

# torch 2.9.1 comes with the base image; the test tools in requirements.txt are not needed to serve
RUN grep -v -E '^(pytest|ruff)' requirements.txt > /tmp/requirements.txt && \
    python -m pip install -r /tmp/requirements.txt "torchao==0.13.0" && \
    python -c "import torch, torchao, transformers, qwen_tts, fastapi; import breeze_infer.api; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'torchao', torchao.__version__)"

COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

ENV MODELS_DIR=/models TTS_HOST=0.0.0.0 TTS_PORT=7860 \
    BREEZE_FAST_CONFIG=/opt/breeze/configs/slim.json BREEZE_INT8=backbone,text_encoder,depth
EXPOSE 7860
# warm-up captures CUDA graphs and compiles kernels: allow a few minutes
HEALTHCHECK --interval=10s --timeout=3s --start-period=600s CMD curl -sf http://127.0.0.1:${TTS_PORT}/health || exit 1
ENTRYPOINT ["/opt/start.sh"]
