# Engines

Each engine is a process the harness talks to over the local network, in `engines/<role>/<name>/`, where the role is `llm`, `asr` or `tts`.

The manifest format, the contracts and how to add an engine arrive with the engine foundation. For now this folder holds the existing Docker services: `asr/audiocpp` and `tts/breeze`.
