#!/usr/bin/env bash
# Needle correctness check for the M4.1 host/CPU lever configs (BAS-144).
#
# Starts one llama-server per config (same pinned Stage 0 base flags as
# bench/run-warm-prefix-profile.sh), plants the bongo sentinel in a short-context
# document, and records whether the model recalls it.  This is the "correctness
# unchanged" guard for flags that move host-resident MoE weights (--load-mode,
# --no-op-offload) or resize the CPU thread pool.
#
# The single GPU is held for the whole run (BAS-80); the check is resumable.
#
# Usage:
#   bench/run-needle-check.sh                          # baseline + no_op_offload
#   BONGO_NEEDLE_CONFIGS="baseline no_op_offload threads16 lm_none" bench/run-needle-check.sh
#   BONGO_NEEDLE_PREFIX=4096 bench/run-needle-check.sh
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
# shellcheck source=bench/gpu-lock.sh
. "$repo/bench/gpu-lock.sh"

tier="${BONGO_SWEEP_TIER:-iq2_xs}"
bongo_home="${BONGO_HOME:-$HOME/.bongo}"
llama_bin="${BONGO_LLAMA_SERVER:-$bongo_home/llama/b11223/vulkan/llama-server}"
gguf_dir="${BONGO_GGUF_DIR:-$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier}"
host="${BONGO_HOST:-127.0.0.1}"
port="${BONGO_PORT:-8080}"
device="${BONGO_DEVICE:-Vulkan1}"
ctx="${BONGO_CTX:-131072}"
prefix="${BONGO_NEEDLE_PREFIX:-8192}"
out_root="${BONGO_NEEDLE_OUT:-bench/results/2026-09-29-host-cpu/needle}"
base_url="http://$host:$port/v1"

declare -A CFG_FLAGS
CFG_FLAGS[baseline]=""
CFG_FLAGS[lm_none]="--load-mode none"
CFG_FLAGS[no_op_offload]="--no-op-offload"
CFG_FLAGS[threads16]="--threads 16 --threads-batch 16"
# M4.3 (BAS-158): the shipped bongo.sh default (patched engine + --load-mode
# none + the two GGML_VK upload env vars). Requires BONGO_LLAMA_SERVER pointed
# at the M4.2-patched engine build.
declare -A CFG_ENV
CFG_FLAGS[shipped_default]="--load-mode none"
CFG_ENV[shipped_default]="GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1"
selected="${BONGO_NEEDLE_CONFIGS:-baseline no_op_offload}"
selected="${selected//,/ }"

log() { printf '[needle %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
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
[[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }
(( ${#shards[@]} > 0 )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
model="${shards[0]}"

bongo_gpu_lock_acquire "run-needle-check (BAS-144)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
if healthy; then
  log "ERROR: port $port is already serving after acquiring the GPU lock"
  exit 3
fi

rc_total=0
for name in $selected; do
  flags="${CFG_FLAGS[$name]:-}"
  if [[ -z "${CFG_FLAGS[$name]+x}" ]]; then
    log "unknown config '$name'; skipping"
    rc_total=2
    continue
  fi
  out_dir="$out_root/$name"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/needle.json" ]]; then
    log "$name already checked; skipping"
    continue
  fi
  # shellcheck disable=SC2206
  extra=($flags)
  argv=(--model "$model" --ctx-size "$ctx" --jinja
    --cache-type-k q8_0 --cache-type-v q8_0
    --host "$host" --port "$port" --parallel 1 --alias "bongo-$tier"
    --metrics --device "$device" --spec-type none --n-cpu-moe 16 --n-gpu-layers 99 --flash-attn on)
  argv+=("${extra[@]}")
  env_flags="${CFG_ENV[$name]:-}"
  log "$name: starting llama-server flags: ${flags:-<baseline>} env: ${env_flags:-<none>}"
  # shellcheck disable=SC2086
  env $env_flags setsid "$llama_bin" "${argv[@]}" > "$out_dir/llama-server.log" 2>&1 &
  pid=$!
  waited=0; started=0
  while (( waited < 900 )); do
    if healthy; then started=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; waited=$(( waited + 5 ))
  done
  if (( started == 0 )); then
    log "$name: server failed to become healthy"
    stop_server "$pid"
    rc_total=3
    continue
  fi
  python3 bench/check-needle.py --prefix "$prefix" --ctx "$ctx" \
    --out "$out_dir/needle.json" --label "$name" --server-pid "$pid" \
    --flags-note "engine=llama.cpp b11223 backend=Vulkan tier=$tier flags=${flags:-<baseline>}" \
    >> "$out_dir/harness.log" 2>&1
  rc=$?
  [[ $rc -ne 0 ]] && rc_total=$rc
  log "$name: needle exit $rc"
  stop_server "$pid"
done
log "done (rc=$rc_total); results in $out_root"
exit $rc_total
