# Adding an engine

An engine is a process the harness talks to over the local network: a speech recognizer (`asr`), a speech synthesizer (`tts`) or a language model (`llm`). You describe how to start it in a manifest, `engine.toml`, and the supervisor starts it, checks its health, and stops it. The harness doesn't care what runs inside, as long as it speaks its role's protocol: the **contract**.

This page goes from nothing to an engine that passes its checks. Then [Adding a config](adding-a-config.md) puts it to use.

## 1. Start from a stub

Every role has a stub: no model, no GPU, the smallest server that passes the role's checks. Copy one:

```bash
cp -r engines/tts/stub engines/tts/myvoice
```

The folder names are the engine's id, `tts/myvoice`, and must match `role` and `name` in the manifest. Engines that run in Docker end in `-docker` (`tts/myvoice-docker`).

You don't have to change the repository at all: Speakrail also finds engines in `~/.speakrail/engines/<role>/<name>/` and in every folder of `$SPEAKRAIL_ENGINES_PATH`. One there with the same id replaces a shipped one.

## 2. Write the manifest

`src/speakrail/manifest.py` documents every field. The parts you will touch:

```toml
contract = 1
role = "tts"
name = "myvoice"
description = "My voice model, on the CPU"
capabilities = ["stream", "cancel"]     # only what it really does: each one is checked
platforms = ["linux", "darwin"]         # sys.platform values
url = "http://127.0.0.1:${port_main}/v1/audio/speech"
startup_timeout_s = 120
needs = ["${models}/myvoice/model.bin"]  # files that must exist first; `doctor` checks them too

[ports]
main = 8095                             # each port becomes ${port_<name>}

[params]                                # defaults; a config can change any of them
model_dir = "${models}/myvoice"

[harness]
worker = "pocket"                       # which client the harness uses for it (TTS: "pocket" or "breeze")

[[process]]                             # one or more, started in order; any one dying fails the engine
name = "server"
cmd = ["${python}", "-u", "${engine_dir}/server.py", "--port", "${port_main}", "--model", "${model_dir}"]
health = "http://127.0.0.1:${port_main}/health"
```

A Python engine gets its own environment from a `[python]` table with a `requirements.txt` next to the manifest; `speakrail engine install tts/myvoice` creates it. A Docker engine instead sets `image` on its process (plus `gpu = true` for the NVIDIA Container Toolkit, `mounts`, `entrypoint`, `docker_args`), runs on the host network, and keeps its Dockerfile in the engine's folder. Pin everything: an image by tag, a git source by commit.

## 3. Speak the contract

What each role must do. `src/speakrail/contracts/conformance.py` is the exact definition: it is what checks your engine.

**LLM** (`/v1/completions`, vLLM's dialect):

- prompts as token ids; `max_tokens`, `temperature`, `stop_token_ids` (finishing with `stop_reason`), `logit_bias`, `skip_special_tokens`;
- decisions: `allowed_token_ids` restricts the answer, and `logprobs` with `return_tokens_as_token_ids` reports it as `token_id:N` with the top alternatives. The speak token (106) is also an end-of-turn token, and a decision must still be able to answer with it;
- streaming with `stream_options.include_usage`;
- a 4xx for an unknown `model` and for a prompt longer than the context;
- `model` picks the base model or the LoRA per request (`speakrail-base` / `speakrail`).

Capabilities it may declare: `chat` (`/v1/chat/completions`, used by the listening notes), `parallel`, `per_request_adapter`, `cancel`, `prefix_cache`, `sampling`.

**ASR** (a WebSocket at the engine's URL):

- the client connects with `?encoding=pcm_s16le&sample_rate=16000&channels=1&delay_ms=480&turn_detection=none&client_id=...&turn_probs=true` and sends 16 kHz mono PCM; the engine answers `session.created` first;
- `transcript` events carry timed words; `turn` events, one per 80 ms frame, carry the turn head's probabilities (`p`): speaking, finished, pausing mid-thought, backchannel, ... An engine without a real turn head can derive them from a voice activity detector, but it must send them: the harness ends turns from these frames (`turn_head` is required);
- `{"type": "input_audio.end"}` gets `session.ended`;
- `{"type": "peek", "frames": N, "id": ...}` (capability `peek`) answers with `transcript.peek`: the words still in the delay line, while audio keeps flowing.

Capabilities: `transcribe`, `turn_head`, `anticipation` (three more outputs that predict the end of a turn), `peek`.

**TTS** (`worker = "pocket"`: POST JSON `{"model", "input", "response_format": "pcm", "stream_format": "audio", "voice"}`, raw 24 kHz 16-bit PCM back; `worker = "breeze"`: a form with a reference voice):

- speech comes back as PCM and isn't silent;
- capabilities: `stream` (audio arrives while it is generated: needed for a fast first word), `cancel` (closing the connection stops it), `clone` (`voice_ref`, a voice sent with the request).

## 4. Test it

```bash
speakrail engine show tts/myvoice      # what the manifest says, after its params are filled in
speakrail engine run tts/myvoice       # start it in the foreground: its log shows why it won't start
speakrail engine test tts/myvoice      # the contract checks; --url tests one that is already running
```

Every check in `engine test` must pass, including one for every capability the manifest claims.

An LLM engine running a Speakrail model also has to make the same decisions as the reference engine did with the same model files. `verify/fidelity/` holds those references: about 120 decisions each, taken from real conversations on the test tapes, with the token the reference chose and its logprobs. Point the check at your engine, loading the same model (with `--param` as your config sets it), or a port of it to your engine's format:

```bash
speakrail engine fidelity llm/myllm --reference verify/fidelity/nvidia-12b-24gb.json   # the 12B
speakrail engine fidelity llm/myllm --reference verify/fidelity/nvidia-e4b-16gb.json   # E4B
```

It passes when at least 95% of the decisions are the same (`--agree` changes that; a config sets its own bar in `[fidelity]`, see [Adding a config](adding-a-config.md)): a near-tie can flip between engines and GPUs through bf16 rounding, so a few differences are normal, many are a bug (a wrong chat template, a missing token row, a LoRA not applied). `speakrail verify` runs the same check by itself when a reference's model files are all in the config. A new checkpoint gets its reference from `python verify/record_fidelity.py <config>`.

Then put the engine in a config ([Adding a config](adding-a-config.md)) and run `speakrail verify --config NAME`: the whole stack with the test tapes, which shows whether the engine behaves in a real conversation, not only in its checks.
