# speakrail/vllm-openai:0.26.0-ple, the image of llm/vllm-docker: stock vLLM 0.26 plus the Gemma 4 PLE offload patch (patch_ple.py).
#   docker build -t speakrail/vllm-openai:0.26.0-ple engines/llm/vllm-docker
# Run with
#   --cpu-offload-gb 6 --cpu-offload-params embed_tokens_per_layer
# to keep E2B/E4B's per-layer embedding table in pinned host memory.
FROM docker.io/vllm/vllm-openai:v0.26.0
COPY patch_ple.py /tmp/patch_ple.py
RUN python3 /tmp/patch_ple.py && rm /tmp/patch_ple.py
