# Configs

One file per setup: which engines run, and which features they start with. Auto mode only picks `status = "tested"`, and a tested config has to say what it was tested on, in `[evidence]`. Everything else is `experimental`: it runs only when asked for by name, with `speakrail run --config NAME`.

    speakrail config            what auto mode would choose here, and why
    speakrail config list       every config, and whether it fits this machine
