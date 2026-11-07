# Engines

Each engine is a process the harness talks to over the local network, in `engines/<role>/<name>/`, where the role is `llm`, `asr` or `tts`. An `engine.toml` says how to run it and what it can do; `src/speakrail/manifest.py` documents the format.

| Engine | What it is | Runs on |
|---|---|---|
| `llm/mlx` | Gemma E4B + LoRA on native MLX | Mac |
| `llm/vllm-docker` | Gemma + LoRA on vLLM 0.26, from the image its Dockerfile builds (stock 0.26 + a patch that can keep E2B/E4B's per-layer embeddings in host memory); a config picks the model | Linux, NVIDIA |
| `asr/audiocpp` | Voxtral + turn head on audio.cpp, behind the bridge | Mac |
| `asr/audiocpp-docker` | The same, from the image `asr/audiocpp/Dockerfile` builds | Linux, NVIDIA |
| `tts/pocket` | Pocket TTS on audio.cpp, on the CPU | Mac |
| `tts/pocket-docker` | The same, on the CPU, from the ASR engine's image | Linux |
| `tts/breeze-docker` | Breeze TTS 2 (non-commercial), from the image its Dockerfile builds | Linux, NVIDIA |
| `llm/stub`, `asr/stub`, `tts/stub` | No model and no GPU: for tests, and templates to copy | Mac, Linux |

`audiocpp.toml` pins the audio.cpp commit and build flags behind `asr/audiocpp` and `tts/pocket`. The ASR Dockerfile pins the same commit.

A process in a manifest with `image` runs in a container on the host network (`src/speakrail/containers.py`), so ports and health checks work as for a local process. GPU access goes through the NVIDIA Container Toolkit, and `speakrail doctor` says what to do if Docker doesn't have its runtime.

    python -m speakrail engine list
    python -m speakrail engine install asr/audiocpp
    python -m speakrail engine test asr/audiocpp
