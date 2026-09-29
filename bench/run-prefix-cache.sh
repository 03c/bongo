#!/usr/bin/env bash
# Measure the prefix-cache (agentic turn) path on the pinned Stage 0 backend.
#
#   ./bench/run-prefix-cache.sh
#   BONGO_PREFIXES=4096,31744,65536 ./bench/run-prefix-cache.sh
#
# It starts its own llama-server through ./bongo.sh (which now warms the server
# and exposes --slot-save-path), waits for health, runs
# `bench/measure-prefix-cache.py` (cold / hit / grow / repeat plus a timed slot
# save -> erase -> restore that proves the restored KV is reused), then stops
# only the server it started. The GPU is exclusive: do not run two at once.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "${here}")"
# shellcheck source=bench/gpu-lock.sh
. "$here/gpu-lock.sh"
scratch="${PAPERCLIP_RUN_SCRATCH_DIR:-${PAPERCLIP_SCRATCH_DIR:-$(mktemp -d)}}"
mkdir -p "$scratch"

REV="${BONGO_LLAMA_REV:-b11223}"
HOST="${BONGO_HOST:-127.0.0.1}"
PORT="${BONGO_PORT:-8080}"
CTX="${BONGO_CTX:-131072}"
N_CPU_MOE="${BONGO_N_CPU_MOE:-16}"
PREFIXES="${BONGO_PREFIXES:-4096,31744}"
DELTA="${BONGO_DELTA:-512}"
SLOT_ID="${BONGO_SLOT_ID:-0}"
OUT_DIR="${BONGO_PREFIX_OUT:-$repo/bench/results/$(date +%Y-%m-%d)-prefix-cache-m3.0a}"
# BAS-86: persist context checkpoints alongside the slot KV and grade the
# restored slot with the post-restore needle. Both stay opt-in so the default
# baseline run is byte-for-byte what it was before.
SAVE_SLOT_CHECKPOINTS="${BONGO_SAVE_SLOT_CHECKPOINTS:-0}"
NEEDLE="${BONGO_NEEDLE:-0}"
CKPT_FLAGS=()
NEEDLE_FLAGS=()
if [[ "$SAVE_SLOT_CHECKPOINTS" == "1" || "$SAVE_SLOT_CHECKPOINTS" == "true" ]]; then
  CKPT_FLAGS+=(--save-slot-checkpoints)
fi
if [[ "$NEEDLE" == "1" || "$NEEDLE" == "true" ]]; then
  NEEDLE_FLAGS+=(--needle)
fi

# State (pid/log/slots) stays private so a concurrently running server is not touched.
STATE_HOME="${BONGO_STATE_HOME:-$scratch/bongo-home}"
SLOT_DIR="${BONGO_SLOT_DIR:-$scratch/slots}"
LLAMA_BIN="${BONGO_LLAMA_BIN:-$HOME/.bongo/llama/$REV/vulkan}"
RUNTIME_DIR="${BONGO_RUNTIME_DIR:-$HOME/.bongo/runtime}"
GGUF_DIR="${BONGO_GGUF_DIR:-$HOME/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs}"

[ -x "$LLAMA_BIN/llama-server" ] || { echo "llama-server not found under $LLAMA_BIN" >&2; exit 1; }
[ -d "$GGUF_DIR" ] || { echo "GGUF dir not found at $GGUF_DIR" >&2; exit 1; }
# A locally built engine (--llama-bin) may not carry an $ORIGIN RPATH, so make
# sure its sibling shared libraries are found. Harmless for the prebuilt asset.
export LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if curl -sf -m 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "A server is already listening on http://$HOST:$PORT; refusing to disturb it." >&2
  echo "Stop it (or set BONGO_PORT) before running this measurement." >&2
  exit 3
fi

mkdir -p "$STATE_HOME" "$SLOT_DIR" "$OUT_DIR"
start_log="$scratch/prefix-cache-bongo-start.log"

# Serialise on the single GPU for the whole measured run (BAS-80). bongo.sh
# inherits BONGO_GPU_LOCK_HELD=1 and will not try to take the lock again.
bongo_gpu_lock_acquire "run-prefix-cache ctx=$CTX port=$PORT" || exit 3

echo "starting bongo.sh: Vulkan, n-cpu-moe=$N_CPU_MOE, ctx=$CTX, port=$PORT, slots=$SLOT_DIR"
BONGO_HOME="$STATE_HOME" "$repo/bongo.sh" \
  --backend vulkan --llama-bin "$LLAMA_BIN" --gguf-dir "$GGUF_DIR" \
  --runtime dir --runtime-dir "$RUNTIME_DIR" \
  --ctx "$CTX" --n-cpu-moe "$N_CPU_MOE" --port "$PORT" \
  --slot-save-path "$SLOT_DIR" "${CKPT_FLAGS[@]}" --detach >"$start_log" 2>&1
grep -E 'Warmup|Server ready|Slot KV|Endpoint' "$start_log" | sed 's/^/  /' || true

spid="$(cat "$STATE_HOME/run/llama-server.pid" 2>/dev/null || true)"
cleanup() { if [ -n "$spid" ]; then kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; fi; bongo_gpu_lock_release; }
trap cleanup EXIT

if [ -z "$spid" ] || ! kill -0 "$spid" 2>/dev/null; then
  echo "server pid not found; tail of $start_log:" >&2
  tail -30 "$start_log" >&2
  exit 1
fi
echo "server pid=$spid; running measurement"

BONGO_BASE_URL="http://$HOST:$PORT/v1" BONGO_MODEL=bongo-iq2_xs BONGO_SERVER_PID="$spid" \
  python3 "$here/measure-prefix-cache.py" \
  --prefixes "$PREFIXES" --delta "$DELTA" --slot-id "$SLOT_ID" \
  --slot-save-dir "$SLOT_DIR" "${NEEDLE_FLAGS[@]}" --out "$OUT_DIR"

echo "saved $OUT_DIR/prefix-cache.json"
