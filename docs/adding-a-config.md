# Adding a config

A config is one setup: which engine fills each role, the models they load, the harness features that go with those models, and how much memory it all needs. `configs/README.md` lists the shipped ones; `src/speakrail/resolver.py` reads them.

## 1. Write it

Copy the config closest to yours, give it a new `name` (the file name should match), and set `status = "experimental"`: it runs only when asked for by name until it has evidence. Configs in `~/.speakrail/configs/` are found too, so you can keep yours outside the repository.

```toml
name = "nvidia-e4b-12gb"
status = "experimental"
description = "Gemma E4B on vLLM, Voxtral q4, Pocket on the CPU, on one 12 GB NVIDIA GPU"   # shown when picking a setup
platform = "cuda"                  # "apple" or "cuda": a config is offered only on its platform
priority = 5                       # among tested configs that fit, the highest is recommended
memory_peak_gib = 10.2             # all processes together, at their peak: measure it (step 2)
disk_gib = 15.0                    # the models

[engines]
asr = "asr/audiocpp-docker"
tts = "tts/pocket-docker"
llm = "llm/vllm-docker"

[harness]                          # features and the settings that belong to the model
turn_k = 1
peek_frames = 10
notes = true

[params."llm/vllm-docker"]         # any engine param, for this config only
gpu_share = "0.50"
max_model_len = "65536"

[[download]]                       # every file, from a repository at an exact commit
repo = "speakrail/gemma-4-E4B-it-qat-Speakrail-181026"
revision = "41d5549664bd1422f679bb32c3e83ebee9b896ee"
include = ["adapter_model.safetensors", "adapter_config.json"]
to = "speakrail-e4b/adapter"
```

Rules that keep a config reproducible:

- **Pin every download to a commit**, never a branch: `revision = "main"` changes under you. `from` picks a folder in the repository, `include` the files, `to` where they go under the models folder.
- **Settings that belong to a checkpoint stay in the config**: `reply_temp`, `reply_top_p`, `reply_top_k` and `decision_bias` go in `[harness]` when the model was benchmarked with them. No environment variable or `--set` reaches them.
- **One vLLM image, one audio.cpp image**: if your engine needs a different build, that is a new engine (see [Adding an engine](adding-an-engine.md)), not a param.
- **Name the decision reference when you port a model.** `verify` compares the LLM's decisions with a reference recorded from the same model (see [Adding an engine](adding-an-engine.md)). It finds one by itself when your config downloads exactly the files a reference came from. A port to other files (a GGUF of the 12B for llama.cpp, another quantization) has to name it, and may set a lower bar, because rounding flips a few more near-ties:

  ```toml
  [fidelity]
  reference = "nvidia-12b-24gb"     # verify/fidelity/<name>.json: the model you ported
  agree = 0.90                      # the share of decisions that must match; 0.95 if left out
  ```

- **Leave the desktop room.** The resolver keeps 1 GB free on top of `memory_peak_gib`. On a GPU that also drives a screen, aim for more.

## 2. Measure and verify it

```bash
speakrail install --config nvidia-e4b-12gb      # fetches what it needs
speakrail verify --config nvidia-e4b-12gb
```

`install --config` also makes it the config `speakrail start` runs from then on; `speakrail install` again lets you pick another.

`verify` starts the config the way `speakrail run` does, checks every engine against its contract, then plays five synthetic conversations (`verify/tapes/`) through the whole stack at real time and checks what the assistant does:

| tape | what it checks |
|---|---|
| everyday | it answers, and gets 17 × 23 right |
| pauses | it doesn't cut in during a pause in the middle of a sentence, and converts units correctly |
| backchannels | "mm-hmm" while it talks doesn't stop it |
| interruption | the user cutting in stops it, and the new question gets answered |
| tools | it calls the shopping list and notes tools when asked |

Pass or fail is about that behaviour and those answers. Latency and memory depend on the hardware, so `verify` measures and reports them but never fails on them: use its peak memory for `memory_peak_gib` (round up). `--quick` plays only the first tape, for a fast check that an install works. `--param ENGINE:KEY=VALUE` changes an engine param for that run, and the report lists it: for example `--param llm/vllm-docker:gpu_share=0.240` gives vLLM on a 24 GB card the GiB that `0.48` gives it on a 12 GB one, for a first measurement before a real 12 GB card.

The report lands in `~/.speakrail/verify/<config>-<time>/`: `report.md` (this machine, the exact versions of everything that ran, every check with its detail) and `report.json`, plus each tape's events for debugging.

## 3. Make it tested

A config becomes `tested` through a pull request:

1. the config file, with `status = "tested"`;
2. its `[evidence]`: `report.md` ends with a block to paste in;
3. `report.md` itself, attached to the pull request: a verify that passed, from a clean checkout (the report says so if the code had uncommitted changes).

A maintainer reruns it where they can. Tested configs are what auto mode recommends, so the evidence has to come from the same machine class the config is for: a 12 GB config verified on a 24 GB card doesn't show it fits in 12 GB.
