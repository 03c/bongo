#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-TBD
#
# bench/run-prefix-cache-m3.0c.sh — BAS-145 / M3.0c same-window A/B at ~128K.
#
# Runs `bench/run-prefix-cache.sh` twice under **one** hold of the single-GPU
# flock (BAS-80), so the measured sidecar reuse and its stock control come from
# the same GPU window:
#
#   1. patched  the reproducible patched engine from
#               `tools/build-llama-vulkan.sh` with `--save-slot-checkpoints`
#               (the BAS-86 checkpoint sidecar)
#   2. stock    the pinned b11223 Vulkan baseline, flag off
#
# `run-prefix-cache.sh` does cold / hit / grow / grow-repeat and then
# prime -> save -> erase -> restore -> after at the last prefix size, so the
# `after` case reports `cache_n` for a slot restored from disk.
#
# The 128K cold prefill is ~16 min and the stock post-restore verify turn is a
# second full prefill, so the A/B is ~50-70 min of GPU time. Launch it detached
# behind a blocking lock wait instead of polling:
#
#   mkdir -p bench/results/2026-09-29-prefix-cache-m3.0c
#   BONGO_GPU_LOCK_TIMEOUT=-1 setsid nohup bench/run-prefix-cache-m3.0c.sh \
#       > bench/results/2026-09-29-prefix-cache-m3.0c/runner.log 2>&1 &
#
# Safe to re-run: `--plan` prints the two invocations and touches nothing.
#
# Environment:
#   BONGO_CTX                 server context (default 131072)
#   BONGO_PREFIXES            token prefix target (default 128000, leaves room
#                             for the chat template inside a 131072 context)
#   BONGO_M30C_PATCHED_BIN    patched llama-server dir (default the pinned
#                             build tree: ~/.bongo/engine/llama.cpp-pin/build-vulkan/bin)
#   BONGO_M30C_STOCK_BIN      stock llama-server dir (default ~/.bongo/llama/b11223/vulkan)
#   BONGO_M30C_OUT            output root (default bench/results/<date>-prefix-cache-m3.0c)
#   BONGO_M30C_STATE          durable state root (default ~/.bongo/prefix-cache-m3.0c)
#   BONGO_M30C_NEEDLE         run the post-restore needle on the patched run (default 1)
#   BONGO_GPU_LOCK_TIMEOUT    passed through (default -1: queue behind the holder)
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
# shellcheck source=bench/gpu-lock.sh
. "$here/gpu-lock.sh"

DATE="${BONGO_DATE:-$(date -u +%Y-%m-%d)}"
CTX="${BONGO_CTX:-131072}"
PREFIXES="${BONGO_PREFIXES:-128000}"
PATCHED_BIN="${BONGO_M30C_PATCHED_BIN:-$HOME/.bongo/engine/llama.cpp-pin/build-vulkan/bin}"
STOCK_BIN="${BONGO_M30C_STOCK_BIN:-$HOME/.bongo/llama/b11223/vulkan}"
OUT_ROOT="${BONGO_M30C_OUT:-$repo/bench/results/${DATE}-prefix-cache-m3.0c}"
STATE="${BONGO_M30C_STATE:-$HOME/.bongo/prefix-cache-m3.0c}"
NEEDLE="${BONGO_M30C_NEEDLE:-1}"
MODEL="${BONGO_MODEL:-bongo-iq2_xs}"
HOST="${BONGO_HOST:-127.0.0.1}"
PORT="${BONGO_PORT:-8080}"

for arg in "$@"; do
  case "$arg" in
    --plan)
      cat <<PLAN
plan (nothing started; no GPU taken)
  out root    $OUT_ROOT
  state root  $STATE   (durable: outlives the heartbeat, holds the slot files)
  context     ctx=$CTX prefix=$PREFIXES
  1. patched  BONGO_LLAMA_BIN=$PATCHED_BIN
              BONGO_SAVE_SLOT_CHECKPOINTS=1 BONGO_NEEDLE=$NEEDLE
              -> $OUT_ROOT/ctx128k-patched/prefix-cache.json
  2. stock    BONGO_LLAMA_BIN=$STOCK_BIN
              BONGO_SAVE_SLOT_CHECKPOINTS=0 BONGO_NEEDLE=0
              -> $OUT_ROOT/ctx128k-stock/prefix-cache.json
  lock        BONGO_GPU_LOCK_TIMEOUT=${BONGO_GPU_LOCK_TIMEOUT:--1} (queues behind the holder)
PLAN
      exit 0;;
    -h|--help) sed -n '2,45p' "$0"; exit 0;;
    *) echo "unknown argument: $arg" >&2; exit 2;;
  esac
done

log() { printf '[m3.0c %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

mkdir -p "$OUT_ROOT" "$STATE"

# Do not let the detached run depend on the heartbeat's scratch directory: the
# harness removes PAPERCLIP_RUN_SCRATCH_DIR after this heartbeat ends, and a
# queued run may not start until long after that.
export BONGO_STATE_ROOT="$STATE"
export TMPDIR="${BONGO_M30C_TMP:-$STATE/tmp}"
mkdir -p "$TMPDIR"
unset PAPERCLIP_RUN_SCRATCH_DIR PAPERCLIP_SCRATCH_DIR PAPERCLIP_TASK_SCRATCH_DIR

[ -x "$PATCHED_BIN/llama-server" ] || { log "no patched llama-server at $PATCHED_BIN"; exit 5; }
[ -x "$STOCK_BIN/llama-server" ]   || { log "no stock llama-server at $STOCK_BIN"; exit 5; }

PATCHED_SHA="$(sha256sum "$PATCHED_BIN/llama-server" 2>/dev/null | awk '{print $1}')"
STOCK_SHA="$(sha256sum "$STOCK_BIN/llama-server" 2>/dev/null | awk '{print $1}')"
if ! LD_LIBRARY_PATH="$PATCHED_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$PATCHED_BIN/llama-server" --help 2>&1 | grep -q -- '--save-slot-checkpoints'; then
  log "patched binary does not advertise --save-slot-checkpoints; refusing to measure ($PATCHED_BIN)"
  exit 5
fi

LOCK_T0=$SECONDS
BONGO_GPU_LOCK_TIMEOUT="${BONGO_GPU_LOCK_TIMEOUT:--1}"
bongo_gpu_lock_acquire "run-prefix-cache-m3.0c (BAS-145)" || exit 3
LOCK_WAIT_S=$(( SECONDS - LOCK_T0 ))
trap 'bongo_gpu_lock_release' EXIT INT TERM
log "holding the single-GPU lock after ${LOCK_WAIT_S}s of queueing"

# A server this run did not start means somebody is outside the lock protocol
# (BAS-80); refuse rather than evict it.
if pgrep -x llama-server >/dev/null 2>&1; then
  log "another llama-server is already running; refusing to overlap (BAS-80)"
  pgrep -a -x llama-server >&2 || true
  exit 4
fi

# Provenance: the two engine hashes and the exact date this session started.
{
  echo "{"
  echo "  \"schema\": \"bongo.prefix-cache-m3.0c-runner.v1\","
  echo "  \"started_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"gpu_lock_wait_s\": $LOCK_WAIT_S,"
  echo "  \"ctx\": $CTX,"
  echo "  \"prefixes\": \"$PREFIXES\","
  echo "  \"patched_bin\": \"$PATCHED_BIN\","
  echo "  \"patched_sha256\": \"$PATCHED_SHA\","
  echo "  \"stock_bin\": \"$STOCK_BIN\","
  echo "  \"stock_sha256\": \"$STOCK_SHA\""
  echo "}"
} >"$OUT_ROOT/runner-start.json"

rc_patched=0
rc_stock=0

# run_one <label> <bin> <save_flag> <needle> <out_dir>
run_one() {
  local label="$1" bin="$2" save="$3" needle="$4" out="$5"
  local slot_dir="$STATE/slots-$label"
  mkdir -p "$slot_dir" "$out"
  log "=== $label: engine=$bin save_slot_checkpoints=$save needle=$needle ==="
  local t0=$SECONDS
  BONGO_GPU_LOCK_HELD=1 \
  BONGO_STATE_HOME="$STATE/home-$label" \
  BONGO_SLOT_DIR="$slot_dir" \
  BONGO_LLAMA_BIN="$bin" \
  BONGO_SAVE_SLOT_CHECKPOINTS="$save" \
  BONGO_NEEDLE="$needle" \
  BONGO_PREFIXES="$PREFIXES" \
  BONGO_CTX="$CTX" \
  BONGO_HOST="$HOST" BONGO_PORT="$PORT" \
  BONGO_PREFIX_OUT="$out" \
  "$here/run-prefix-cache.sh" >>"$OUT_ROOT/$label.run.log" 2>&1
  local rc=$?
  log "=== $label rc=$rc elapsed=$((SECONDS-t0))s ==="
  return "$rc"
}

run_one patched "$PATCHED_BIN" 1 "$NEEDLE" "$OUT_ROOT/ctx128k-patched" || rc_patched=$?
run_one stock   "$STOCK_BIN"   0 0        "$OUT_ROOT/ctx128k-stock"   || rc_stock=$?

log "done patched_rc=$rc_patched stock_rc=$rc_stock; results under $OUT_ROOT"
BONGO_M30C_PATCHED_RC="$rc_patched" BONGO_M30C_STOCK_RC="$rc_stock" \
  python3 - "$OUT_ROOT" <<'PY'
import json, os, sys
out = sys.argv[1]
summary = {
    "patched_rc": int(os.environ.get("BONGO_M30C_PATCHED_RC", "1")),
    "stock_rc": int(os.environ.get("BONGO_M30C_STOCK_RC", "1")),
    "finished_at": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cases": {},
}
for label in ("patched", "stock"):
    p = os.path.join(out, "ctx128k-" + label, "prefix-cache.json")
    if not os.path.isfile(p):
        summary["cases"][label] = None
        continue
    d = json.load(open(p))
    slot = d.get("slot") or {}
    after = slot.get("after_restore") or {}
    summary["cases"][label] = {
        "cache_n": after.get("cache_n"),
        "after_restore_prompt_tokens": after.get("prompt_tokens"),
        "restore_ttft_ms": slot.get("restore_ttft_ms"),
        "restore_reuse": slot.get("restore_reuse"),
        "restore_verified": slot.get("restore_verified"),
        "after_restore": after,
    }
with open(os.path.join(out, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
PY
exit $(( rc_patched != 0 || rc_stock != 0 ))
