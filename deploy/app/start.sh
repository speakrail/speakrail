#!/usr/bin/env bash
# Speakrail app: render the phrase cache for the voice once (needs the TTS service), then serve the UI.
set -euo pipefail
cd /app
TTS_ENGINE=${TTS_ENGINE:-breeze}
[ "$TTS_ENGINE" = breeze ] && VOICE=${VOICE:-female_a.wav}
case "${VOICE:-}" in */*|"") ;; *.wav) VOICE="/app/src/speakrail/harness/voices/$VOICE" ;; esac   # a bare wav name = one of the shipped voices
export VOICE TTS_ENGINE
case " $* " in *" --stub "*) exec python -m speakrail.harness.server "$@" ;; esac          # no TTS in stub mode
python -m speakrail.harness.phrase_cache render --engine "$TTS_ENGINE" --voice "${VOICE:-}" --urls "${TTS_URL:-http://127.0.0.1:7860/v1/audio/speech}" || \
  echo "app: phrase cache not rendered (replies then start without pre-rendered first clauses)" >&2
exec python -m speakrail.harness.server "$@"
