#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-TBD
#
# bench/run-prefix-longctx.sh — M3.5b: the 128K/256K cached 512-token delta turn (BAS-132).
#
# One command for the product-path metric at the two long contexts:
#
#   * ctx128k: ~128K cached prefix at the pinned Stage 0 placement
#              (q8 KV, --n-cpu-moe 16)
#   * ctx256k: ~256K cached prefix at the shipped long-context default
#              (q8 KV, --n-cpu-moe 18)
#
# Per context the script starts the pinned bongo.sh server, waits for it (and
# its shader warmup), then runs `bench/measure-prefix-cache.py --cached-only
# --no-slot`:
#
#   prime_p<P>   first request on the empty slot; this is the one full prefill
#   hit_p<P>     the identical prompt again; must be a full KV reuse
#   grow_p<P>    prefix + 512 tokens; the delta-turn TTFT (the product metric)
#   repeat       the grown prompt again; must be a full hit
#
# Peak VRAM is sampled per case (MemorySampler, fdinfo) so the run records what
# the context actually costs on the box.
#
#   ./bench/run-prefix-longctx.sh            # queue for the GPU, then run
#   ./bench/run-prefix-longctx.sh --plan     # print the plan, touch nothing
#
# The script takes the single-GPU flock (BAS-80) once for both contexts, so a
# sibling measurement queues instead of overlapping (two llama-servers cannot
# fit the 32 GiB card). It is safe to launch detached:
#
#   setsid nohup ./bench/run-prefix-longctx.sh >> <log> 2>&1 &
#
# Budget: one 128K prefill (~25 min) + one 256K prefill (~50 min) plus the fast
# hit/delta turns. Expect ~1.5 h of GPU once the lock is held.
#
# The slot save/restore path is deliberately skipped (--no-slot): this model
# does not reuse a restored slot (BAS-86), so the post-restore re-prefill would
# add ~25/50 min per context for no product number.
#
# Environment overrides: BONGO_LLAMA_REV, BONGO_PORT, BONGO_LLAMA_BIN,
# BONGO_GGUF_DIR, BONGO_RUNTIME_DIR, BONGO_OUT, BONGO_PLC_HOME, BONGO_DELTA,
# BONGO_CASE_TIMEOUT, BONGO_GPU_LOCK_TIMEOUT.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
# shellcheck source=bench/gpu-lock.sh
. "$here/gpu-lock.sh"

PLAN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --plan) PLAN_ONLY=1;;
    -h|--help) sed -n '2,48p' "$0"; exit 0;;
    *) echo "unknown argument: $arg" >&2; exit 2;;
  esac
done

DATE="${BONGO_DATE:-$(date -u +%Y-%m-%d)}"
REV="${BONGO_LLAMA_REV:-b11223}"
HOST="${BONGO_HOST:-127.0.0.1}"
PORT="${BONGO_PORT:-8080}"
LLAMA_BIN="${BONGO_LLAMA_BIN:-$HOME/.bongo/llama/$REV/vulkan}"
RUNTIME_DIR="${BONGO_RUNTIME_DIR:-$HOME/.bongo/runtime}"
GGUF_DIR="${BONGO_GGUF_DIR:-$HOME/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs}"
OUT="${BONGO_OUT:-$repo/bench/results/${DATE}-prefix-cache-longctx}"
# Durable state: the run outlives the heartbeat that launches it, and the
# Paperclip run scratch is reaped when the run ends. Never keep it there.
STATE_HOME="${BONGO_PLC_HOME:-$HOME/.bongo/prefix-longctx}"
SLOT_DIR="${BONGO_PLC_SLOT_DIR:-$STATE_HOME/run/slots}"
CASE_TIMEOUT="${BONGO_CASE_TIMEOUT:-7200}"
DELTA="${BONGO_DELTA:-512}"
MODEL="${BONGO_MODEL:-bongo-iq2_xs}"

# label ctx n_cpu_moe prefix_tokens
CONFIGS=(
  "ctx128k 131072 16 128000"
  "ctx256k 262144 18 256000"
)

log() { printf '[prefix-longctx %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

server_pid() { cat "$STATE_HOME/run/llama-server.pid" 2>/dev/null || true; }

if (( PLAN_ONLY )); then
  echo "plan (nothing started; no GPU taken)"
  echo "  out         $OUT"
  echo "  state       $STATE_HOME   (server log/pid; durable, not run scratch)"
  echo "  engine      $LLAMA_BIN"
  echo "  model       $GGUF_DIR (iq2_xs, alias $MODEL)"
  echo "  method      measure-prefix-cache.py --cached-only --no-slot --delta $DELTA"
  echo "  timeout     $CASE_TIMEOUT s per request"
  for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    printf '  %-9s ctx=%-7s --n-cpu-moe %-2s prefixes=%s\n' "$1" "$2" "$3" "$4"
  done
  exit 0
fi

mkdir -p "$OUT" "$SLOT_DIR" "$STATE_HOME/run"

# Queue for the single GPU instead of evicting the current holder (BAS-80).
LOCK_T0=$SECONDS
BONGO_GPU_LOCK_TIMEOUT="${BONGO_GPU_LOCK_TIMEOUT:-86400}"
bongo_gpu_lock_acquire "run-prefix-longctx (BAS-132)" || exit 3
LOCK_WAIT_S=$(( SECONDS - LOCK_T0 ))
log "holding the single-GPU lock after ${LOCK_WAIT_S}s of queueing"

start_server() {
  local stage="$1" ctx="$2" n_cpu_moe="$3" spid
  mkdir -p "$OUT/$stage"
  log "=== $stage: starting server (ctx=$ctx, n-cpu-moe=$n_cpu_moe) ==="
  BONGO_HOME="$STATE_HOME" "$repo/bongo.sh" \
    --backend vulkan --llama-bin "$LLAMA_BIN" --gguf-dir "$GGUF_DIR" \
    --runtime dir --runtime-dir "$RUNTIME_DIR" \
    --ctx "$ctx" --n-cpu-moe "$n_cpu_moe" --port "$PORT" \
    --slot-save-path "$SLOT_DIR" --detach >"$OUT/$stage/server-start.log" 2>&1
  spid="$(server_pid)"
  if [[ -z "$spid" ]] || ! kill -0 "$spid" 2>/dev/null; then
    log "$stage: server failed to start; see $OUT/$stage/server-start.log"
    tail -20 "$OUT/$stage/server-start.log" >&2
    return 1
  fi
  log "$stage: server pid=$spid"
  echo "$spid" >"$OUT/$stage/server.pid"
  curl -s -m 10 "http://$HOST:$PORT/props" >"$OUT/$stage/props.json" 2>/dev/null || true
  cp "$STATE_HOME/run/bongo-config.json" "$OUT/$stage/bongo-config.json" 2>/dev/null || true
  cp "$STATE_HOME/run/llama-server.log" "$OUT/$stage/llama-server.log" 2>/dev/null || true
  return 0
}

stop_server() {
  local spid
  spid="$(server_pid)"
  [[ -n "$spid" ]] || return 0
  log "stopping server pid=$spid"
  kill "$spid" 2>/dev/null || true
  for _ in $(seq 1 180); do kill -0 "$spid" 2>/dev/null || break; sleep 1; done
  kill -9 "$spid" 2>/dev/null || true
  rm -f "$STATE_HOME/run/llama-server.pid"
  sleep 3
}

cleanup() { stop_server; bongo_gpu_lock_release; }
trap cleanup EXIT INT TERM

# A server this run did not start must not overlap: the box holds one model at
# 256K and a second server would device-lose (BAS-75/BAS-80).
if pgrep -x llama-server >/dev/null 2>&1; then
  log "a llama-server is already running and is not ours; refusing to start a second one"
  pgrep -a -x llama-server >&2 || true
  exit 4
fi
[ -x "$LLAMA_BIN/llama-server" ] || { log "no llama-server at $LLAMA_BIN"; exit 5; }

START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rc=0
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  stage="$1"; ctx="$2"; n_cpu_moe="$3"; prefix="$4"
  start_server "$stage" "$ctx" "$n_cpu_moe" || { rc=6; break; }
  BONGO_BASE_URL="http://$HOST:$PORT/v1" BONGO_MODEL="$MODEL" \
  BONGO_SERVER_PID="$(server_pid)" \
    python3 "$here/measure-prefix-cache.py" \
      --prefixes "$prefix" --delta "$DELTA" --cached-only --no-slot \
      --timeout "$CASE_TIMEOUT" --slot-id 0 \
      --out "$OUT/$stage" >"$OUT/$stage/measure.log" 2>&1
  measure_rc=$?
  log "$stage: measure rc=$measure_rc ($(tail -1 "$OUT/$stage/measure.log" 2>/dev/null | head -c 140))"
  (( measure_rc == 0 )) || rc=$measure_rc
  stop_server
  (( rc == 0 )) || break
done
END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# One reviewable summary across both contexts plus the per-context raw JSON.
python3 - "$OUT" "$START_ISO" "$END_ISO" "$LOCK_WAIT_S" "$DELTA" "$rc" "${CONFIGS[@]}" <<'PY'
import json
import pathlib
import sys

out, start, end, lock_wait, delta, rc = sys.argv[1:7]
configs = sys.argv[7:]
out = pathlib.Path(out)


def load_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return None


contexts = []
for cfg in configs:
    label, ctx, n_cpu_moe, prefix = cfg.split()
    d = load_json(out / label / "prefix-cache.json") or {}
    runs = {r.get("label"): r for r in d.get("runs", [])}
    hit = runs.get(f"hit_p{int(prefix)}", {})
    grow = runs.get(f"grow_p{int(prefix)}_d{delta}", {})
    repeat = runs.get(f"grow_p{int(prefix)}_repeat", {})
    config = load_json(out / label / "bongo-config.json") or {}
    contexts.append(
        {
            "label": label,
            "ctx_flag": int(ctx),
            "n_cpu_moe": int(n_cpu_moe),
            "prefix_target": int(prefix),
            "prefix_actual": grow.get("cache_n"),
            "delta_tokens": int(delta),
            "full_hit_ttft_ms": hit.get("ttft_ms"),
            "delta_turn_ttft_ms": grow.get("ttft_ms"),
            "delta_prompt_tokens": grow.get("prompt_tokens"),
            "delta_prompt_ms": grow.get("prompt_ms"),
            "delta_prompt_tps": grow.get("prompt_tps"),
            "delta_output_tps": grow.get("output_tps"),
            "delta_repeat_ttft_ms": repeat.get("ttft_ms"),
            "vram_peak_bytes": (grow.get("memory") or {}).get("vram_peak_bytes")
            or (d.get("memory") or {}).get("vram_peak_bytes"),
            "vram_method": (d.get("memory") or {}).get("vram_method"),
            "server_flags": (config.get("server") or {}).get("flags"),
            "engine": {
                "revision": ((config.get("llama_cpp") or {}).get("revision")),
                "commit": ((config.get("llama_cpp") or {}).get("commit")),
                "backend": ((config.get("llama_cpp") or {}).get("backend")),
            },
            "raw": f"{label}/prefix-cache.json",
        }
    )

summary = {
    "schema": "bongo.prefix-cache-longctx.v1",
    "issue": "BAS-132",
    "started_at": start,
    "finished_at": end,
    "gpu_lock_wait_s": int(lock_wait),
    "runner_rc": int(rc),
    "delta_tokens": int(delta),
    "mode": "cached-only",
    "contexts": contexts,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

log "done rc=$rc; summary at $OUT/summary.json"
exit "$rc"
