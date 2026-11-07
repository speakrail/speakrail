# Configs

One file per setup: which engines run, and which features they start with. Auto mode only picks `status = "tested"`, and a tested config has to say what it was tested on, in `[evidence]`. Everything else is `experimental`: it runs only when asked for by name, with `speakrail run --config NAME`.

    speakrail config            what auto mode would choose here, and why
    speakrail config list       every config, and whether it fits this machine

Replies are greedy and decisions have no bias unless a config says otherwise. These belong to the checkpoint, so a config that uses one benchmarked with them sets them in `[harness]`; no environment variable or `--set` reaches them:

    reply_temp = 0.7            # replies and interjections; decisions stay greedy. Needs an LLM engine with `sampling`
    reply_top_p = 0.95
    reply_top_k = 64
    decision_bias = "106:1.0"   # <token id>:<logit bias>[,...]; 106 is speak

| Config | For | Models | Peak, all processes | Last word to ear, p50/p90 |
|---|---|---|---|---|
| `mac-e4b` | Apple Silicon, 16 GB and up | E4B on MLX, Voxtral q4, Pocket on the CPU | 10.7 GB | 954/1351 ms |
| `nvidia-12b-24gb` | NVIDIA, 24 GB and up | 12B, Voxtral q8, Breeze (non-commercial) | 22.9 GB | 736/1327 ms |
| `nvidia-e4b-16gb` | NVIDIA, 16 GB and up | E4B with its per-layer embeddings in host memory, Voxtral q8, Pocket on the CPU | 11.9 GB | 537/1130 ms |
