#!/usr/bin/env bash
# Speakrail ASR: audiocpp_server (Voxtral Mini 4B Realtime q8_0 + turn head) behind the bridge (bridge.py).
#
#   MODELS_DIR       where model-init put the weights (default /models)
#   ASR_GGUF         Voxtral GGUF       (default $MODELS_DIR/voxtral/voxtral-mini-4b-realtime-2602-q8_0.gguf)
#   ASR_TURN_HEAD    turn head .vxth     (default $MODELS_DIR/turn_head/turn_head.vxth; its .json sidecar sits next to it)
#   AUDIOCPP_BIN     audiocpp_server binary built from github.com/speakrail/audio.cpp, branch turn-head (default /app/audiocpp_server)
#   ASR_HOST/PORT    where the bridge listens for the harness (default 127.0.0.1:8765)
#   ACPP_PORT        audiocpp_server's own port, local only (default 8090)
#   ASR_DRY_RUN=1    print the audio.cpp config and exit
#   OMP_NUM_THREADS  CPU threads for audio.cpp (default 4; the GPU does the heavy work)
set -euo pipefail

MODELS_DIR=${MODELS_DIR:-/models}
ASR_GGUF=${ASR_GGUF:-$MODELS_DIR/voxtral/voxtral-mini-4b-realtime-2602-q8_0.gguf}
ASR_TURN_HEAD=${ASR_TURN_HEAD:-$MODELS_DIR/turn_head/turn_head.vxth}
AUDIOCPP_BIN=${AUDIOCPP_BIN:-/app/audiocpp_server}
ASR_HOST=${ASR_HOST:-127.0.0.1}
ASR_PORT=${ASR_PORT:-8765}
ACPP_PORT=${ACPP_PORT:-8090}
HERE=$(cd "$(dirname "$0")" && pwd)
CONFIG=${ASR_CONFIG:-/tmp/speakrail_audiocpp.json}

cat > "$CONFIG" <<EOF
{
  "host": "127.0.0.1",
  "port": $ACPP_PORT,
  "backend": "cuda",
  "device": 0,
  "threads": 4,
  "lazy_load": false,
  "busy_timeout_ms": 0,
  "live_ingest": {
    "idle_timeout_ms": 120000,
    "total_timeout_ms": 14400000,
    "max_body_bytes": 4294967296,
    "max_chunk_bytes": 8388608,
    "send_timeout_ms": 30000
  },
  "models": [
    {
      "id": "voxtral-rt",
      "family": "voxtral_realtime",
      "path": "$ASR_GGUF",
      "task": "asr",
      "mode": "streaming",
      "session_options": {
        "voxtral_realtime.turn_head": "$ASR_TURN_HEAD",
        "voxtral_realtime.stream_chunk_samples": "128"
      }
    }
  ]
}
EOF
if [ "${ASR_DRY_RUN:-0}" = 1 ]; then cat "$CONFIG"; exit 0; fi

for f in "$ASR_GGUF" "$ASR_TURN_HEAD" "$ASR_TURN_HEAD.json" "$AUDIOCPP_BIN"; do
  [ -e "$f" ] || { echo "asr: missing $f" >&2; exit 1; }
done

# audio.cpp is built with OpenMP, which otherwise starts one spinning worker per core (~3 cores busy on a 32-core box)
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE} "$AUDIOCPP_BIN" --config "$CONFIG" &
ACPP_PID=$!
for i in $(seq 1 300); do
  kill -0 "$ACPP_PID" 2>/dev/null || { echo "asr: audiocpp_server exited during start-up" >&2; exit 1; }
  curl -s -m 1 "http://127.0.0.1:$ACPP_PORT/health" >/dev/null && break
  sleep 1
done

python3 "$HERE/bridge.py" --host "$ASR_HOST" --port "$ASR_PORT" --acpp "127.0.0.1:$ACPP_PORT" --model voxtral-rt \
  --head "$ASR_TURN_HEAD.json" &
BRIDGE_PID=$!

# either process dying takes the service down, so the container restarts cleanly
trap 'kill "$ACPP_PID" "$BRIDGE_PID" 2>/dev/null' EXIT INT TERM
wait -n "$ACPP_PID" "$BRIDGE_PID"
