#!/usr/bin/env bash
# M4.3 (BAS-158) — verify the *shipped* bongo.sh default reaches the M4.2 numbers.
#
# `bongo.sh` default since M4.3 is: the M4.2-patched Vulkan engine, `--load-mode
# none`, and `GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1`.
# This runner measures exactly that (the `shipped_default` harness config), the
# `bongo.sh --engine stage0` opt-out, the needle check, and the 256K fit/load
# spot check.  One GPU-lock hold for the whole session (BAS-80).
#
# Resumable: a stage whose raw file already exists is skipped.
#
# Usage:
#   bench/run-m4.3-shipped-default.sh                 # all stages
#   BONGO_M43_STAGE=turns bench/run-m4.3-shipped-default.sh
#   BONGO_M43_DRY_RUN=1 bench/run-m4.3-shipped-default.sh
#
# Environment:
#   BONGO_M43_OUT         results root (default bench/results/2026-09-29-m4.3-shipped-default)
#   BONGO_M43_STAGE       all | turns | needle | ctx256 (default all)
#   BONGO_LLAMA_SERVER    patched engine binary (default the cached M4.2 build)
#   BONGO_GGUF_DIR        tier dir (default $BONGO_HOME/models/.../iq2_xs)
#   BONGO_DEVICE          Vulkan device (default Vulkan1)
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
# shellcheck source=bench/gpu-lock.sh
. "$repo/bench/gpu-lock.sh"

tier="${BONGO_SWEEP_TIER:-iq2_xs}"
bongo_home="${BONGO_HOME:-$HOME/.bongo}"
llama_bin="${BONGO_LLAMA_SERVER:-$bongo_home/engine/llama.cpp-pin/build-vulkan/bin/llama-server}"
gguf_dir="${BONGO_GGUF_DIR:-$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier}"
host="${BONGO_HOST:-127.0.0.1}"
port="${BONGO_PORT:-8080}"
device="${BONGO_DEVICE:-Vulkan1}"
out="${BONGO_M43_OUT:-bench/results/2026-09-29-m4.3-shipped-default}"
stage="${BONGO_M43_STAGE:-all}"
dry_run="${BONGO_M43_DRY_RUN:-0}"
load_timeout="${BONGO_M43_LOAD_TIMEOUT:-1500}"
base_url="http://$host:$port/v1"

log() { printf '[m4.3 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

stop_server() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$pid" 2>/dev/null || true
}

shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
if (( ! dry_run )); then
  [[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }
  (( ${#shards[@]} > 0 )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
  if ! grep -qa -m1 'GGML_VK_HOST_BUFT_PER_DEVICE' "$(dirname "$llama_bin")"/libggml-vulkan.so* 2>/dev/null; then
    log "ERROR: $llama_bin is not the M4.2-patched engine; build it with BONGO_APPLY_M42_PATCH=1 tools/build-llama-vulkan.sh"
    exit 2
  fi
fi
model="${shards[0]:-}"

run_turns() {
  log "turns: shipped_default (16K), shipped_default_128k, stage0_optout (16K)"
  BONGO_HOST_SPLIT=1 \
  BONGO_LLAMA_SERVER="$llama_bin" \
  BONGO_PROFILE_OUT="$out/turns" \
  BONGO_PROFILE_LOAD_TIMEOUT="$load_timeout" \
  BONGO_PROFILE_DELTAS="512" \
  BONGO_PROFILE_CONFIGS="shipped_default shipped_default_128k stage0_optout" \
    bench/run-warm-prefix-profile.sh
}

run_needle() {
  log "needle: shipped_default"
  BONGO_NEEDLE_OUT="$out/needle" \
  BONGO_NEEDLE_CONFIGS="shipped_default" \
  BONGO_LLAMA_SERVER="$llama_bin" \
    bench/run-needle-check.sh
}

# 256K fit/load spot check: start the shipped default at the 256K placement
# (--ctx 262144 --n-cpu-moe 18) with the upload levers, wait for health, send one
# tiny request, and record VRAM + RAM. A full 256K prefill is not run (30+ min).
run_ctx256() {
  local out_dir="$out/ctx256"
  local rec="$out_dir/ctx256-fit.json"
  mkdir -p "$out_dir"
  if [[ -f "$rec" ]]; then log "ctx256 already checked; skipping"; return 0; fi
  local server_log="$out_dir/llama-server.log"
  local argv=(--model "$model" --ctx-size 262144 --jinja
    --cache-type-k q8_0 --cache-type-v q8_0
    --host "$host" --port "$port" --parallel 1 --alias "bongo-$tier"
    --metrics --device "$device" --spec-type none
    --flash-attn on --n-cpu-moe 18 --n-gpu-layers 99 --load-mode none)
  log "ctx256: starting llama-server (n_cpu_moe=18, load-mode none, upload levers)"
  printf '%s\n' "$llama_bin ${argv[*]}" > "$out_dir/command.txt"
  env GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1 \
    setsid "$llama_bin" "${argv[@]}" > "$server_log" 2>&1 &
  local pid=$!
  echo "$pid" > "$out_dir/llama-server.pid"
  local waited=0 started=0
  while (( waited < load_timeout )); do
    if healthy; then started=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; waited=$(( waited + 5 ))
  done
  if (( started == 0 )); then
    log "ctx256: server failed to become healthy"
    tail -n 5 "$server_log" >&2
    stop_server "$pid"
    return 3
  fi
  log "ctx256: healthy after ${waited}s"
  python3 - "$tier" "$base_url" "$pid" "$rec" "$waited" <<'PY'
import json, sys, time
sys.path.insert(0, "bench")
from bench_lib import MemorySampler, post_json
tier, base, pid, out, waited = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
sampler = MemorySampler(server_pids=[pid], interval=0.25)
sampler.start()
time.sleep(1.5)
res = post_json(base, "/completions", {"model": f"bongo-{tier}", "prompt": "bongo ctx256 fit",
                                       "max_tokens": 4, "temperature": 0.0, "cache_prompt": False}, 900)
time.sleep(1.5)
sampler.stop()
now = time.time()
peak = sampler.window_peak(0, now)
json.dump({
    "schema": "bongo.ctx256-fit.v1",
    "label": "ctx256-n18-shipped-default",
    "ctx_size": 262144,
    "n_cpu_moe": 18,
    "load_mode": "none",
    "m42_env": ["GGML_VK_HOST_BUFT_PER_DEVICE=1", "GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1"],
    "loaded": True,
    "load_seconds": waited,
    "request_status": res.status,
    "vram_peak_gib": round(peak["vram_peak_bytes"] / 2**30, 3) if peak["vram_peak_bytes"] else None,
    "vram_method": peak["vram_method"],
    "rss_peak_gib": round(peak["system_ram_process_hwm_bytes"] / 2**30, 3) if peak["system_ram_process_hwm_bytes"] else None,
    "host_ram_available_gib": round((sampler.host_mem().get("MemAvailable") or 0) / 2**20, 3),
}, open(out, "w"), indent=2)
print(f"ctx256 fit: status={res.status} vram_peak_gib={json.load(open(out))['vram_peak_gib']}")
PY
  local rc=$?
  stop_server "$pid"
  return $rc
}

if [[ "$dry_run" == "1" ]]; then
  log "dry run; results root $out"
  log "engine: $llama_bin"
  BONGO_HOST_SPLIT=1 BONGO_PROFILE_DRY_RUN=1 \
  BONGO_PROFILE_OUT="$out/turns" \
  BONGO_PROFILE_CONFIGS="shipped_default shipped_default_128k stage0_optout" \
    bench/run-warm-prefix-profile.sh
  exit 0
fi

bongo_gpu_lock_acquire "run-m4.3-shipped-default ($stage)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
export BONGO_GPU_LOCK_HELD=1
if healthy; then
  log "ERROR: port $port already serving after acquiring the GPU lock; refusing to contaminate."
  exit 3
fi

rc=0
if [[ "$stage" == "all" || "$stage" == "turns" ]]; then run_turns || rc=$?; fi
if [[ "$stage" == "all" || "$stage" == "ctx256" ]]; then run_ctx256 || rc=$?; fi
if [[ "$stage" == "all" || "$stage" == "needle" ]]; then run_needle || rc=$?; fi
log "done (rc=$rc); results in $out"
exit $rc
