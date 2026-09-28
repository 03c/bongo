#!/usr/bin/env bash
# M3.0 backend A/B — warm SYCL vs Vulkan for the shipped agentic profile (BAS-72).
#
# Holds the single-GPU lock (BAS-80) for the WHOLE run so no sibling can
# interleave, then for each backend starts the pinned bongo server with the
# shipped flags and runs bench/harness.py cold at 4096/131072 (3 repeats) plus
# the prefix-cache (cached-turn) path.
#
#   ./bench/run-backend-ab.sh
#
# Environment overrides: BONGO_LLAMA_REV, BONGO_PORT, BONGO_CTX, BONGO_N_CPU_MOE,
# BONGO_GGUF_DIR, BONGO_RUNTIME_DIR, BONGO_CONTEXTS, BONGO_REPEATS,
# BONGO_PREFIXES, BONGO_GPU_LOCK_TIMEOUT, BONGO_BACKENDS.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "${here}")"
# shellcheck source=bench/gpu-lock.sh
. "$here/gpu-lock.sh"

scratch="${PAPERCLIP_RUN_SCRATCH_DIR:-${PAPERCLIP_SCRATCH_DIR:-$(mktemp -d)}}"
mkdir -p "$scratch"
out="$repo/bench/results/2026-09-28-backend-ab"
mkdir -p "$out"

REV="${BONGO_LLAMA_REV:-b11223}"
HOST="${BONGO_HOST:-127.0.0.1}"
PORT="${BONGO_PORT:-8080}"
CTX="${BONGO_CTX:-131072}"
N_CPU_MOE="${BONGO_N_CPU_MOE:-16}"
GGUF_DIR="${BONGO_GGUF_DIR:-$HOME/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs}"
RUNTIME_DIR="${BONGO_RUNTIME_DIR:-$HOME/.bongo/runtime}"
CONTEXTS="${BONGO_CONTEXTS:-4096,131072}"
REPEATS="${BONGO_REPEATS:-3}"
PREFIXES="${BONGO_PREFIXES:-4096,31744}"
BACKENDS="${BONGO_BACKENDS:-vulkan sycl}"

# NOTE: no LD_LIBRARY_PATH / ZEL_LIBRARY_PATH export here on purpose. bongo.sh's
# setup_runtime_env() owns both (it adds the IGC/LLVM lib dir the Level Zero
# probe needs and keeps ZEL_LIBRARY_PATH a single directory, BAS-72), so the A/B
# measures the shipped path rather than a runner-local workaround.

log() { printf '[ab %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

BONGO_GPU_LOCK_TIMEOUT="${BONGO_GPU_LOCK_TIMEOUT:-21600}"
bongo_gpu_lock_acquire "run-backend-ab backend-ab (BAS-72)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM

run_backend() {
  local backend="$1"
  local bin="$HOME/.bongo/llama/$REV/$backend"
  local home="$scratch/home-$backend"
  local be_out="$out/$backend"
  local rc=0
  mkdir -p "$home" "$be_out"

  [ -x "$bin/llama-server" ] || { log "$backend: no llama-server at $bin"; return 1; }

  # One retry: the first start after another server released the GPU can still
  # lose the Level Zero device. bongo.sh's probe failure prints its own output.
  local spid="" attempt
  for attempt in 1 2; do
    log "=== $backend: starting server (ctx=$CTX n_cpu_moe=$N_CPU_MOE) attempt $attempt ==="
    : > "$be_out/bongo-start.log"
    BONGO_HOME="$home" "$repo/bongo.sh" \
      --backend "$backend" --llama-bin "$bin" --gguf-dir "$GGUF_DIR" \
      --runtime dir --runtime-dir "$RUNTIME_DIR" \
      --ctx "$CTX" --n-cpu-moe "$N_CPU_MOE" --port "$PORT" \
      --detach >"$be_out/bongo-start.log" 2>&1
    spid="$(cat "$home/run/llama-server.pid" 2>/dev/null || true)"
    if [ -n "$spid" ] && kill -0 "$spid" 2>/dev/null; then
      break
    fi
    spid=""
    if [ "$backend" = "sycl" ]; then
      log "$backend: device probe failed; direct helper output follows"
      LD_LIBRARY_PATH= "$bin/llama-ls-sycl-device" 2>&1 | head -4 | sed 's/^/    /' || true
    fi
    log "$backend: server did not come up; retrying in 30s"
    sleep 30
  done
  if [ -z "$spid" ]; then
    log "$backend: server failed to start; see $be_out/bongo-start.log"
    tail -20 "$be_out/bongo-start.log" >&2
    return 1
  fi
  log "$backend: server pid=$spid"
  curl -s "http://$HOST:$PORT/props" > "$be_out/props.json" 2>/dev/null || true
  cp "$home/run/bongo-config.json" "$be_out/bongo-config.json" 2>/dev/null || true

  log "$backend: harness contexts=$CONTEXTS repeats=$REPEATS (cold)"
  python3 "$repo/bench/harness.py" --repo-root "$repo" \
    --contexts "$CONTEXTS" --repeats "$REPEATS" --deep-threshold 0 \
    --tier iq2_xs --no-cache-prompt --hash-mode sampled --skip-error-cases \
    --server-pid "$spid" --gguf-dir "$GGUF_DIR" \
    --out-dir "$be_out" >"$be_out/harness.log" 2>&1
  rc=$?
  log "$backend: harness rc=$rc ($(tail -1 "$be_out/harness.log" 2>/dev/null | head -c 120))"

  log "$backend: prefix-cache prefixes=$PREFIXES"
  BONGO_BASE_URL="http://$HOST:$PORT/v1" BONGO_MODEL=bongo-iq2_xs \
    python3 "$repo/bench/measure-prefix-cache.py" \
    --prefixes "$PREFIXES" --delta 512 --server-pid "$spid" \
    --slot-save-dir "$home/run/slots" \
    --out "$be_out/prefix-cache" >"$be_out/prefix-cache.log" 2>&1 || true

  log "$backend: stopping server pid=$spid"
  kill "$spid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$spid" 2>/dev/null || break; sleep 1; done
  kill -9 "$spid" 2>/dev/null || true
  sleep 3
  return "$rc"
}

for backend in $BACKENDS; do
  run_backend "$backend" || log "$backend: run finished with errors (recorded)"
done
log "A/B complete; results under $out"
