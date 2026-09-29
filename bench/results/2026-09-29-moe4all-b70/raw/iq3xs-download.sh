#!/usr/bin/env bash
set -u
BASE="https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/resolve/main/IQ3_XS"
DEST="$HOME/.bongo/models/ukisai-Swift-1.5-Qwen3.8-Flash-Next-GGUF/IQ3_XS"
for f in Swift-1.5-Qwen3.8-Flash-Next-IQ3_XS-00001-of-00003.gguf Swift-1.5-Qwen3.8-Flash-Next-IQ3_XS-00002-of-00003.gguf Swift-1.5-Qwen3.8-Flash-Next-IQ3_XS-00003-of-00003.gguf; do
  echo "[$(date -u +%FT%TZ)] downloading $f"
  curl -sL --retry 5 --retry-delay 5 -C - -o "$DEST/$f" "$BASE/$f" || { echo "[$(date -u +%FT%TZ)] FAILED $f"; exit 1; }
  echo "[$(date -u +%FT%TZ)] done $f"
done
echo "[$(date -u +%FT%TZ)] ALL DONE"
