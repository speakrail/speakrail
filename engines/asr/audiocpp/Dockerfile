# Speakrail ASR: audio.cpp (speakrail fork, branch turn-head) built for Voxtral realtime only, plus the bridge.
#
#   docker build -t speakrail-asr engines/asr/audiocpp
#   docker run --gpus all -v ./models:/models -e ASR_HOST=0.0.0.0 -p 127.0.0.1:8765:8765 speakrail-asr

ARG CUDA_VERSION=12.9.2
ARG UBUNTU_VERSION=24.04

# ---- build audiocpp_server ---------------------------------------------------------------------------------------
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG AUDIOCPP_REPO=https://github.com/speakrail/audio.cpp.git
ARG AUDIOCPP_COMMIT=ba2ae40d4630874121d2dfadddd10fe97c358b89
# 86 = RTX 3090 / A10 / A5000, 89 = RTX 4090 / L4 / L40 (add 120 for the RTX 5090)
ARG CUDA_ARCHS="86;89"
# parallel compile jobs: nvcc takes 1-2 GB of RAM per job
ARG BUILD_JOBS=16

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates gcc-14 g++-14 cmake && \
    rm -rf /var/lib/apt/lists/*
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /src
RUN git clone "$AUDIOCPP_REPO" audio.cpp && cd audio.cpp && git checkout "$AUDIOCPP_COMMIT"

RUN cd audio.cpp && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxtral_realtime \
        -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON \
        -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_CUDA_GRAPHS=ON -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_OPENMP=ON \
        -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --parallel "$BUILD_JOBS" --target audiocpp_server && \
    mkdir -p /out/lib && cp build/bin/audiocpp_server /out/ && find build -name "*.so*" -exec cp -P {} /out/lib/ \;

# ---- runtime -------------------------------------------------------------------------------------------------------
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /out/audiocpp_server /app/audiocpp_server
# ggml loads its backends (CUDA, CPU variants) as plugins from the executable's directory
COPY --from=build /out/lib/ /app/
COPY --from=build /src/audio.cpp/model_specs/ /app/model_specs/
ENV LD_LIBRARY_PATH=/app

COPY requirements.txt /app/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt
ENV PATH=/opt/venv/bin:$PATH

COPY bridge.py start.sh /app/
RUN chmod +x /app/start.sh

ENV MODELS_DIR=/models ASR_HOST=0.0.0.0 ASR_PORT=8765
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=3s --start-period=300s CMD curl -sf http://127.0.0.1:8090/health || exit 1
ENTRYPOINT ["/app/start.sh"]
