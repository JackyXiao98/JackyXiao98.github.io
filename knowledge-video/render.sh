#!/usr/bin/env bash
# Render one episode and verify its subtitles against speech recognition.
# usage: ./render.sh src/ep07_flow.py out/ep07
# env:   PYTHON (default python3), VOICE (default matcha), MODELS (default ./models)
set -euo pipefail
cd "$(dirname "$0")"
script=$1; out=$2
PY=${PYTHON:-python3}
M=${MODELS:-$(pwd)/models}
mkdir -p "$out"
TTS_MODELS=$M ASR_MODEL=$M/sherpa-onnx-paraformer-zh-small-2024-03-09 VOICE=${VOICE:-matcha} \
  $PY "$script" "$out" 2>&1 | grep -v '^frame '
$PY src/check_align.py "$M/sherpa-onnx-paraformer-zh-small-2024-03-09" "$out" 2>/dev/null > "$out/align.txt"
ok=$(grep -c '^OK' "$out/align.txt" || true); all=$(grep -c . "$out/align.txt")
echo "subtitle check: $ok / $all lines matched; review the '??' lines below (usually ASR homophones):"
grep '^??' "$out/align.txt" || true
