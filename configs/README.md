# Configs

One file per setup: which engines run, and which features they start with. Auto mode only picks `status = "tested"`, and a tested config has to say what it was tested on, in `[evidence]`. Everything else is `experimental`: it runs only when asked for by name, with `speakrail run --config NAME`.

    speakrail config            what auto mode would choose here, and why
    speakrail config list       every config, and whether it fits this machine

Replies are greedy and decisions have no bias unless a config says otherwise. These belong to the checkpoint, so a config that uses one benchmarked with them sets them in `[harness]`; no environment variable or `--set` reaches them:

    reply_temp = 0.7            # replies and interjections; decisions stay greedy. Needs an LLM engine with `sampling`
    reply_top_p = 0.95
    reply_top_k = 64
    decision_bias = "106:1.0"   # <token id>:<logit bias>[,...]; 106 is speak
