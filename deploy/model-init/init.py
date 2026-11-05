#!/usr/bin/env python3
"""Speakrail model-init: download every model at a pinned revision, write the 22 trained token rows into Google's
Gemma, and verify everything. Idempotent: finished artifacts are skipped, so a re-run only fetches what is missing.

Layout under $MODELS_DIR (default /models):
    speakrail-gemma/model/       Google's gemma-4-12B-it-qat-w4a16-ct with rows 6-27 of embed_tokens and lm_head
                                 replaced, plus tokenizer v1.2 (served by vLLM as speakrail-base)
    speakrail-gemma/adapter/     the LoRA (served by vLLM as speakrail)
    speakrail-gemma/tokenizer/   tokenizer v1.2 (read by the harness)
    voxtral/voxtral-mini-4b-realtime-2602-q8_0.gguf
    turn_head/turn_head.vxth (+ .json)
    breeze-tts-2/                Breeze TTS 2 (research and non-commercial use only)

The patch overwrites only the bytes of rows 6-27 in the two bf16 tensors, so it needs no torch and little memory.
The patched file is checked against the SHA-256 of the exact model that was benchmarked.
"""
import hashlib, json, os, shutil, struct, sys, time

from huggingface_hub import hf_hub_download

MODELS = os.environ.get("MODELS_DIR", "/models")
CACHE = os.path.join(MODELS, ".download-cache")
KEEP_CACHE = os.environ.get("KEEP_DOWNLOAD_CACHE") == "1"

GEMMA = ("google/gemma-4-12B-it-qat-w4a16-ct", "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee")
SPEAKRAIL = ("speakrail/gemma-4-12B-it-qat-Speakrail", "f1004b2763dafee266f108bb2f5167554b984c48")
TURN_HEAD = ("speakrail/Voxtral-Mini-4B-Realtime-2602-TurnHead", "ba89de14c626a362d4c7bc35094b0972c534e9d7")
VOXTRAL = ("audio-cpp/audio.cpp-gguf", "e36610ac69b5262e914a52635324050bee8f1ad2")
BREEZE = ("BreezeBlue/Breeze-TTS-2", "3e28c5151381a722f1d8661b4118c298caa77aa4")

# repo file -> destination under MODELS, size, sha256 (None: small git file, checked by size only)
FILES = [
    (GEMMA, "config.json", "speakrail-gemma/model/config.json", 6183, None),
    (GEMMA, "generation_config.json", "speakrail-gemma/model/generation_config.json", 255, None),
    (GEMMA, "processor_config.json", "speakrail-gemma/model/processor_config.json", 1382, None),
    (SPEAKRAIL, "adapter_model.safetensors", "speakrail-gemma/adapter/adapter_model.safetensors", 262373840,
     "913d0ecf6e5598592a4c4243ec013e9749cec0ad2be0d6103ea717c5bd6856e8"),
    (SPEAKRAIL, "adapter_config.json", "speakrail-gemma/adapter/adapter_config.json", None, None),
    (SPEAKRAIL, "token_rows.safetensors", "speakrail-gemma/token_rows.safetensors", 169208, "1b8e4089a112a4a2605ce6eee58c43c787e2749bcb8c856276dc519d57e30972"),
    (SPEAKRAIL, "tokenizer/tokenizer.json", "speakrail-gemma/tokenizer/tokenizer.json", None, "f7fa088e59a7c7d935ddff8911cf0a901ccdb927ca9cc677c5c1de4bcee22b48"),
    (SPEAKRAIL, "tokenizer/tokenizer_config.json", "speakrail-gemma/tokenizer/tokenizer_config.json", None, None),
    (SPEAKRAIL, "tokenizer/chat_template.jinja", "speakrail-gemma/tokenizer/chat_template.jinja", None, None),
    (TURN_HEAD, "turn_head.vxth", "turn_head/turn_head.vxth", 9455706,
     "ccb8dcee3024d0d3b9758955fc9dd22a34a3be84604ebbc2a2ea83a65c6e1a32"),
    (TURN_HEAD, "turn_head.vxth.json", "turn_head/turn_head.vxth.json", None, None),
    (VOXTRAL, "Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf",
     "voxtral/voxtral-mini-4b-realtime-2602-q8_0.gguf", 5104567264, "0312a5ceafc6ee4a19a32da458cab6485e3b86b1b563c7bb7150aeae295c1769"),
]
BREEZE_FILES = ["LICENSE", "README.md", "config.json", "generation_config.json", "model.safetensors.index.json",
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "special_tokens_map.json",
                "tokenizer.json", "tokenizer_config.json", "audio_tokenizer/config.json",
                "audio_tokenizer/configuration.json", "audio_tokenizer/model.safetensors",
                "audio_tokenizer/preprocessor_config.json"]

GEMMA_WEIGHTS = (GEMMA, "model.safetensors", "speakrail-gemma/model/model.safetensors", 10264229896, "60b6e3989502969d8ae04185d72ecbbc7db63978d5af747a493d53895aa6bfa3")
PATCHED_SHA256 = "cea2cc956b76af496c85637a6fb87a7ed8f53e028729e8c92db0d698908af6ec"            # Google's file with our rows: the benchmarked model, byte for byte
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
ROWS = range(6, 28)
HIDDEN = 3840


def log(msg):
    print(f"[model-init] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(repo_rev, filename, dest, size=None, digest=None):
    """download one file to MODELS/dest unless it is already there with the right size; verify the digest"""
    path = os.path.join(MODELS, dest)
    if os.path.exists(path) and (size is None or os.path.getsize(path) == size):
        return path
    repo, rev = repo_rev
    t0 = time.time()
    src = hf_hub_download(repo, filename, revision=rev, cache_dir=CACHE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    shutil.copyfile(src, tmp)
    if size is not None and os.path.getsize(tmp) != size:
        sys.exit(f"[model-init] {repo}/{filename}: size {os.path.getsize(tmp)}, expected {size}")
    if digest and sha256(tmp) != digest:
        sys.exit(f"[model-init] {repo}/{filename}: checksum mismatch")
    os.replace(tmp, path)
    log(f"{repo}/{filename} -> {dest} ({os.path.getsize(path) / 2**30:.2f} GiB, {time.time() - t0:.0f} s)")
    return path


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def patch_rows(model_path, rows_path):
    """write rows 6-27 from token_rows.safetensors into embed_tokens and lm_head of model_path, in place"""
    base, hdr = safetensors_header(rows_path)
    t = hdr["embed_tokens"]
    if t["dtype"] != "BF16" or t["shape"] != [len(ROWS), HIDDEN]:
        sys.exit(f"[model-init] unexpected token rows {t}")
    with open(rows_path, "rb") as f:
        f.seek(base + t["data_offsets"][0])
        rows = f.read(len(ROWS) * HIDDEN * 2)
    base, hdr = safetensors_header(model_path)
    with open(model_path, "r+b") as f:
        for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            m = hdr[name]
            if m["dtype"] != "BF16" or m["shape"][1] != HIDDEN:
                sys.exit(f"[model-init] unexpected {name} {m}")
            f.seek(base + m["data_offsets"][0] + ROWS[0] * HIDDEN * 2)
            f.write(rows)


def gemma():
    final = os.path.join(MODELS, GEMMA_WEIGHTS[2])
    marker = final + ".speakrail-patched"
    if os.path.exists(marker) and os.path.exists(final):
        return
    repo_rev, filename, dest, size, digest = GEMMA_WEIGHTS
    log("downloading Gemma 4 12B QAT (int4, ~9.6 GiB)")
    src = hf_hub_download(repo_rev[0], filename, revision=repo_rev[1], cache_dir=CACHE)
    tmp = final + ".part"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    shutil.copyfile(src, tmp)
    if os.path.getsize(tmp) != size or sha256(tmp) != digest:
        sys.exit("[model-init] Gemma base checkpoint does not match the pinned file")
    patch_rows(tmp, os.path.join(MODELS, "speakrail-gemma/token_rows.safetensors"))
    if sha256(tmp) != PATCHED_SHA256:
        sys.exit("[model-init] patched Gemma does not match the benchmarked model")
    os.replace(tmp, final)
    open(marker, "w").write(PATCHED_SHA256 + "\n")
    if not KEEP_CACHE:
        os.remove(os.path.realpath(src))
    log("Gemma patched with the 22 speakrail token rows and verified")


def main():
    os.makedirs(MODELS, exist_ok=True)
    for f in FILES:
        fetch(*f)
    for name in TOKENIZER_FILES:                              # vLLM serves the model dir: it needs tokenizer v1.2 too
        shutil.copyfile(os.path.join(MODELS, "speakrail-gemma/tokenizer", name),
                        os.path.join(MODELS, "speakrail-gemma/model", name))
    gemma()
    for name in BREEZE_FILES:
        fetch(BREEZE, name, os.path.join("breeze-tts-2", name), BREEZE_SIZES.get(name), BREEZE_SHA.get(name))
    if not KEEP_CACHE:
        shutil.rmtree(CACHE, ignore_errors=True)
    log(f"all models ready in {MODELS}")


BREEZE_SIZES = {
 "LICENSE": 18719,
 "README.md": 9818,
 "config.json": 10161,
 "generation_config.json": 251,
 "model.safetensors.index.json": 109004,
 "model-00001-of-00002.safetensors": 4961989890,
 "model-00002-of-00002.safetensors": 2004567152,
 "special_tokens_map.json": 886,
 "tokenizer.json": 33386945,
 "tokenizer_config.json": 1157960,
 "audio_tokenizer/config.json": 2336,
 "audio_tokenizer/configuration.json": 76,
 "audio_tokenizer/model.safetensors": 682293092,
 "audio_tokenizer/preprocessor_config.json": 234
}
BREEZE_SHA = {
 "model-00001-of-00002.safetensors": "abf813781256e10cbe81f2dbb415f897556225d4dfa0282d67aa8ea164e114a9",
 "model-00002-of-00002.safetensors": "36aa73b1a11361e1774db90d9c63c63303b294c022de112aa51904d940edcef1",
 "tokenizer.json": "d3ec9ac3eb2392389b9f5112e85d8b43316494addb587ba7b7a9d61eac23af96",
 "audio_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
}

if __name__ == "__main__":
    main()
