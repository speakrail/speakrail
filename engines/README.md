# Engines

Each engine is a process the harness talks to over the local network, in `engines/<role>/<name>/`, where the role is `llm`, `asr` or `tts`. An `engine.toml` says how to run it and what it can do; `src/speakrail/manifest.py` documents the format.

| Engine | What it is | Runs on |
|---|---|---|
| `llm/mlx` | Gemma E4B + LoRA on native MLX | Mac |
| `asr/audiocpp` | Voxtral + turn head on audio.cpp, behind the bridge | Mac; Linux through its Docker image |
| `tts/pocket` | Pocket TTS on audio.cpp, on the CPU | Mac |
| `tts/breeze` | Breeze TTS 2 Docker image, no manifest yet | Linux |
| `llm/stub`, `asr/stub`, `tts/stub` | No model and no GPU: for tests, and templates to copy | Mac, Linux |

`audiocpp.toml` pins the audio.cpp commit and build flags behind `asr/audiocpp` and `tts/pocket`.

    python -m speakrail engine list
    python -m speakrail engine install asr/audiocpp
    python -m speakrail engine test asr/audiocpp
