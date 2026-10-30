#!/usr/bin/env bash
# Download the offline speech models into knowledge-video/models/ (about 330 MB):
#   matcha-icefall-zh-baker + vocos vocoder  – Chinese TTS (the female voice used in the series)
#   sherpa-onnx-paraformer-zh-small          – Chinese ASR, used to verify narration and subtitle timing
# Optional alternatives for narrate.py: kokoro-multi-lang-v1_1 ("kokoro:<sid>"), vits-melo-tts-zh_en ("melo").
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
get() {  # url, check-path
  [ -e "models/$2" ] && { echo "have $2"; return; }
  echo "downloading $1"
  case "$1" in
    *.tar.bz2) curl -fSL "$1" | tar xj -C models ;;
    *) curl -fSL -o "models/$2" "$1" ;;
  esac
}
get $B/tts-models/matcha-icefall-zh-baker.tar.bz2 matcha-icefall-zh-baker
get $B/vocoder-models/vocos-22khz-univ.onnx vocos-22khz-univ.onnx
get $B/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2 sherpa-onnx-paraformer-zh-small-2024-03-09
if [ "${WITH_ALT_VOICES:-0}" = 1 ]; then
  get $B/tts-models/kokoro-multi-lang-v1_1.tar.bz2 kokoro-multi-lang-v1_1
  get $B/tts-models/vits-melo-tts-zh_en.tar.bz2 vits-melo-tts-zh_en
fi
echo "models ready in $(pwd)/models"
