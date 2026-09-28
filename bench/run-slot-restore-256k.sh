#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-TBD
#
# bench/run-slot-restore-256k.sh — M3.0a 256K q8 slot save/restore (BAS-83).
#
# One command for the 256K point of the slot-KV-persistence measurement: how long
# does it take to persist and reload a ~3-4 GiB KV cache, and is the reloaded KV
# actually reused by the next request?  The 31K point is in
# bench/results/2026-09-28-prefix-cache-m3.0a/.
#
# The run is deliberately two stages with a real server restart in between, so
# the restore has to come off disk into a process that never saw the original
# prefill:
#
#   1. start the pinned 256K server, cold-prefill 262144 tokens, save the slot,
#      erase it (proves the save, and leaves the server holding nothing)
#   2. stop the server, start a fresh one on the same slot directory, restore,
#      then send the identical prompt and report cache_n / prompt_n
#
# A cold 256K prefill is ~50 min on the reference box (~85 prompt tok/s), and the
# post-restore request costs another ~50 min when the restored KV is *not*
# reused, so expect 1.5-2.5 h of GPU time once the lock is held.
#
#   ./bench/run-slot-restore-256k.sh            # queue for the GPU, then run
#   ./bench/run-slot-restore-256k.sh --plan     # print the plan, touch nothing
#
# The script holds the single-GPU lock (BAS-80) for the whole run, so sibling
# measurements serialise instead of corrupting each other.  It is safe to launch
# detached: `setsid nohup ./bench/run-slot-restore-256k.sh >> <log> 2>&1 &`.
#
# Environment overrides: BONGO_LLAMA_REV, BONGO_PORT, BONGO_CTX, BONGO_N_CPU_MOE,
# BONGO_GGUF_DIR, BONGO_RUNTIME_DIR, BONGO_OUT, BONGO_SLOT_DIR,
# BONGO_GPU_LOCK_TIMEOUT, BONGO_SR_SKIP_DELTA (default 1 at 256K).
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
# shellcheck source=bench/gpu-lock.sh
. "$here/gpu-lock.sh"

PLAN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --plan) PLAN_ONLY=1;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;;
    *) echo "unknown argument: $arg" >&2; exit 2;;
  esac
done

DATE="${BONGO_DATE:-$(date -u +%Y-%m-%d)}"
REV="${BONGO_LLAMA_REV:-b11223}"
HOST="${BONGO_HOST:-127.0.0.1}"
PORT="${BONGO_PORT:-8080}"
CTX="${BONGO_CTX:-262144}"
N_CPU_MOE="${BONGO_N_CPU_MOE:-18}"
GGUF_DIR="${BONGO_GGUF_DIR:-$HOME/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs}"
RUNTIME_DIR="${BONGO_RUNTIME_DIR:-$HOME/.bongo/runtime}"
LLAMA_BIN="${BONGO_LLAMA_BIN:-$HOME/.bongo/llama/$REV/vulkan}"
# Deliberately *not* under PAPERCLIP_RUN_SCRATCH_DIR: the run outlives the
# heartbeat that launches it, and the slot file has to survive the restart.
OUT="${BONGO_OUT:-$repo/bench/results/${DATE}-prefix-cache-256k}"
# The default --slot-save-path for a run at this size, so the two stages agree.
SLOT_DIR="${BONGO_SLOT_DIR:-$HOME/.bongo/slot-restore-256k/run/slots}"
BONGO_HOME_DIR="${BONGO_SR_HOME:-$HOME/.bongo/slot-restore-256k}"
SKIP_DELTA="${BONGO_SR_SKIP_DELTA:-1}"
MODEL="${BONGO_MODEL:-bongo-iq2_xs}"

log() { printf '[slot256 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

server_pid() { cat "$BONGO_HOME_DIR/run/llama-server.pid" 2>/dev/null || true; }

# start_server <stage-name>
# Starts the pinned Stage 0 server on the shared slot directory and records the
# exact argv + /props for provenance.
start_server() {
  local stage="$1" spid
  mkdir -p "$OUT"
  log "=== stage $stage: starting server (ctx=$CTX n_cpu_moe=$N_CPU_MOE) ==="
  BONGO_HOME="$BONGO_HOME_DIR" "$repo/bongo.sh" \
    --backend vulkan --llama-bin "$LLAMA_BIN" --gguf-dir "$GGUF_DIR" \
    --runtime dir --runtime-dir "$RUNTIME_DIR" \
    --ctx "$CTX" --n-cpu-moe "$N_CPU_MOE" --port "$PORT" \
    --slot-save-path "$SLOT_DIR" --detach >"$OUT/server-$stage.log" 2>&1
  spid="$(server_pid)"
  if [[ -z "$spid" ]] || ! kill -0 "$spid" 2>/dev/null; then
    log "stage $stage: server failed to start; see $OUT/server-$stage.log"
    tail -20 "$OUT/server-$stage.log" >&2
    return 1
  fi
  log "stage $stage: server pid=$spid"
  echo "$spid" >"$OUT/server-$stage.pid"
  curl -s "http://$HOST:$PORT/props" >"$OUT/props-$stage.json" 2>/dev/null || true
  cp "$BONGO_HOME_DIR/run/bongo-config.json" "$OUT/bongo-config-$stage.json" 2>/dev/null || true
}

stop_server() {
  local spid
  spid="$(server_pid)"
  [[ -n "$spid" ]] || return 0
  log "stopping server pid=$spid"
  kill "$spid" 2>/dev/null || true
  for _ in $(seq 1 120); do kill -0 "$spid" 2>/dev/null || break; sleep 1; done
  kill -9 "$spid" 2>/dev/null || true
  rm -f "$BONGO_HOME_DIR/run/llama-server.pid"
  sleep 3
}

measure() {
  local stage="$1"; shift
  log "=== stage $stage: measure-slot-restore.py $* ==="
  BONGO_BASE_URL="http://$HOST:$PORT/v1" BONGO_MODEL="$MODEL" \
  BONGO_SLOT_SAVE_PATH="$SLOT_DIR" \
    python3 "$repo/bench/measure-slot-restore.py" \
      --stage "$stage" --ctx "$CTX" --out "$OUT" "$@" \
      >"$OUT/measure-$stage.log" 2>&1
  local rc=$?
  log "stage $stage: rc=$rc ($(tail -1 "$OUT/measure-$stage.log" 2>/dev/null | head -c 120))"
  return "$rc"
}

if (( PLAN_ONLY )); then
  cat <<PLAN
plan (nothing started; no GPU taken)
  out         $OUT
  slot dir    $SLOT_DIR   (--slot-save-path, shared by both stages)
  server      bongo.sh --backend vulkan --llama-bin $LLAMA_BIN
              --ctx $CTX --n-cpu-moe $N_CPU_MOE --port $PORT
              --slot-save-path $SLOT_DIR --detach
  model       $GGUF_DIR (iq2_xs, alias $MODEL)
  stage 1     measure-slot-restore.py --stage save --ctx $CTX --out $OUT
  restart     stop_server + start_server restore   (real process restart)
  stage 2     measure-slot-restore.py --stage restore --ctx $CTX --out $OUT
              $( [[ "$SKIP_DELTA" == 1 ]] && echo '--skip-delta' || echo '(with the 131K delta turn)')
  expect      cold prefill ~50 min; save ~1 s; restore ~1-3 s; post-restore
              turn ~50 min when the restored KV is not reused
PLAN
  exit 0
fi

mkdir -p "$OUT" "$SLOT_DIR" "$BONGO_HOME_DIR/run"

# Queue for the single GPU.  The box runs one measurement at a time, so this
# blocks in flock until the current holder is done rather than evicting it.
LOCK_T0=$SECONDS
BONGO_GPU_LOCK_TIMEOUT="${BONGO_GPU_LOCK_TIMEOUT:-86400}"
bongo_gpu_lock_acquire "run-slot-restore-256k slot-restore-256k (BAS-83)" || exit 3
LOCK_WAIT_S=$(( SECONDS - LOCK_T0 ))
trap 'stop_server; bongo_gpu_lock_release' EXIT INT TERM
log "holding the single-GPU lock after ${LOCK_WAIT_S}s of queueing"

# Refuse to measure against a server this run did not start.  The box has 32 GiB
# and the 256K server needs ~30 GiB, so a second llama-server device-loses and can
# take the first one with it (BAS-75).  A healthy endpoint or a stray llama-server
# under the lock means somebody is running outside the protocol (BAS-80).
code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://$HOST:$PORT/v1/models" 2>/dev/null || true)"
if [[ "$code" == "200" ]]; then
  log "a llama-server is already healthy on $HOST:$PORT and is not ours; refusing to overlap"
  exit 4
fi
if pgrep -x llama-server >/dev/null 2>&1; then
  log "a llama-server is already running and is not ours; refusing to start a second one"
  pgrep -a -x llama-server >&2 || true
  exit 4
fi

[ -x "$LLAMA_BIN/llama-server" ] || { log "no llama-server at $LLAMA_BIN"; exit 5; }

START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rc=0
start_server save || exit 6
measure save || { rc=7; stop_server; exit "$rc"; }
stop_server
start_server restore || exit 8
measure restore --skip-delta || rc=9
stop_server
END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cp "$OUT/bongo-config-restore.json" "$OUT/bongo-config.json" 2>/dev/null || true
python3 - "$OUT" "$START_ISO" "$END_ISO" "$LOCK_WAIT_S" "$CTX" "$N_CPU_MOE" "$REV" "$SLOT_DIR" "$rc" <<'PY'
import json, pathlib, sys

out, start, end, lock_wait, ctx, n_cpu_moe, rev, slot_dir, rc = sys.argv[1:10]
out = pathlib.Path(out)


def load(name):
    path = out / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


save, restore = load("slot-restore-save.json"), load("slot-restore-restore.json")
config = load("bongo-config.json") or {}
summary = {
    "schema": "bongo.slot-restore-256k.v1",
    "started_at": start,
    "finished_at": end,
    "gpu_lock_wait_s": int(lock_wait),
    "runner_rc": int(rc),
    "engine": {
        "revision": rev,
        "commit": (config.get("llama_cpp") or {}).get("commit"),
        "backend": (config.get("llama_cpp") or {}).get("backend"),
        "binary": (config.get("llama_cpp") or {}).get("binary"),
    },
    "server_flags": (config.get("server") or {}).get("flags"),
    "placement": {"ctx": int(ctx), "n_cpu_moe": int(n_cpu_moe), **(config.get("placement") or {})},
    "slot_dir": slot_dir,
    "save": {
        "elapsed_ms": (save or {}).get("save_elapsed_ms"),
        "file_bytes": (save or {}).get("slot_file_bytes"),
        "n_past_after_prefill": (save or {}).get("slot_n_past_after_prefill"),
        "prefill_prompt_tokens": (save or {}).get("prefill_prompt_tokens"),
        "prefill_ms": (save or {}).get("prefill_ms"),
        "prefill_tps": (save or {}).get("prefill_tps"),
        "body": ((save or {}).get("save") or {}).get("body"),
        "erase_elapsed_ms": (save or {}).get("erase_elapsed_ms"),
        "n_past_after_erase": (save or {}).get("slot_n_past_after_erase"),
    },
    "restore": {
        "elapsed_ms": (restore or {}).get("restore_elapsed_ms"),
        "file_bytes_before": (restore or {}).get("slot_file_bytes_before_restore"),
        "n_past_after_restore": (restore or {}).get("slot_n_past_after_restore"),
        "n_restored": (restore or {}).get("restore_n_restored"),
        "body": ((restore or {}).get("restore") or {}).get("body"),
    },
    "reuse": {
        "prompt_n": (restore or {}).get("verify_hit_prompt_tokens"),
        "cache_n": (restore or {}).get("verify_hit_cache_n"),
        "cached_tokens": (restore or {}).get("verify_hit_cached_tokens"),
        "prompt_ms": (restore or {}).get("verify_hit_prompt_ms"),
        "ttft_ms": (restore or {}).get("verify_hit_ttft_ms"),
        "restore_verified": (restore or {}).get("restore_verified"),
    },
}
(out / "slot-restore-256k.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

log "done rc=$rc; summary at $OUT/slot-restore-256k.json"
exit "$rc"
