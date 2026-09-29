#!/usr/bin/env bash
# M4.4 (BAS-159) turn-regression + multi-context decode check.
#
# The decode lever must not move the already-met cached-turn targets (>2%), and
# the milestone also needs the 128K decode number.  `bench/profile-warm-prefix.py`
# measures the 512-token cached delta turn and a decode case at each prefix, so
# one server hold with prefixes 16K and ~128K gives all four numbers:
#
#   grow_p16384_d512    -> the 16K cached delta turn (shipped target <=3 s)
#   grow_p127999_d512   -> the 128K cached delta turn (shipped target <=5 s)
#   decode_p16384       -> decode tok/s at 16K
#   decode_p127999      -> decode tok/s at 128K
#
# One server restart for the whole run, under the shared single-GPU flock (BAS-80).
#
# Usage:
#   bench/run-m4.4-turn-check.sh shipped_default          # M4.2 levers only
#   BONGO_M44_TURN_ENV="GGML_OP_OFFLOAD_MIN_BATCH=1" bench/run-m4.4-turn-check.sh lever
#
# Environment:
#   BONGO_M44_TURN_ENV     extra env assignments for the server (the lever)
#   BONGO_M44_TURN_FLAGS   extra server flags
#   BONGO_M44_TURN_OUT     results root (default bench/results/2026-09-29-m4.4-decode/turns)
#   BONGO_M44_TURN_PREFIXES  prefixes (default 16384,127999)
#   BONGO_PROFILE_LOAD_TIMEOUT  seconds to wait for model load (default 1500)
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
# shellcheck source=bench/gpu-lock.sh
. "$repo/bench/gpu-lock.sh"

label="${1:-shipped_default}"
tier="${BONGO_SWEEP_TIER:-iq2_xs}"
bongo_home="${BONGO_HOME:-$HOME/.bongo}"
llama_bin="${BONGO_LLAMA_SERVER:-$bongo_home/engine/llama.cpp-pin/build-vulkan/bin/llama-server}"
gguf_dir="${BONGO_GGUF_DIR:-$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier}"
host="${BONGO_HOST:-127.0.0.1}"
port="${BONGO_PORT:-8080}"
device="${BONGO_DEVICE:-Vulkan1}"
out_root="${BONGO_M44_TURN_OUT:-bench/results/2026-09-29-m4.4-decode/turns}"
prefixes="${BONGO_M44_TURN_PREFIXES:-16384,127999}"
extra_env="${BONGO_M44_TURN_ENV:-}"
extra_flags="${BONGO_M44_TURN_FLAGS:-}"
base_url="http://$host:$port/v1"

log() { printf '[m4.4-turn %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

out_dir="$out_root/$label"
mkdir -p "$out_dir"
profile_out="$out_dir/profile-host-split.json"
if [[ -f "$profile_out" ]]; then
  log "$label already measured ($profile_out); nothing to do"
  exit 0
fi

shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
[[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }
(( ${#shards[@]} > 0 )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
model="${shards[0]}"

bongo_gpu_lock_acquire "run-m4.4-turn-check $label (BAS-159)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
if healthy; then
  log "ERROR: port $port already serving after acquiring the GPU lock; refusing to contaminate."
  exit 3
fi

argv=(--model "$model" --ctx-size 131072 --jinja
  --cache-type-k q8_0 --cache-type-v q8_0
  --host "$host" --port "$port" --parallel 1 --alias "bongo-$tier"
  --metrics --device "$device" --spec-type none
  --flash-attn on --n-cpu-moe 16 --n-gpu-layers 99 --load-mode none)
# shellcheck disable=SC2206
argv+=($extra_flags)
server_env="GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1 $extra_env"

printf '%s\n' "$llama_bin ${argv[*]}" > "$out_dir/command.txt"
printf '%s\n' "$server_env" > "$out_dir/env.txt"
printf '%s\n' "$prefixes" > "$out_dir/prefixes.txt"

log "$label: starting llama-server env='$server_env' flags='$extra_flags'"
# shellcheck disable=SC2086
env $server_env setsid "$llama_bin" "${argv[@]}" > "$out_dir/llama-server.log" 2>&1 &
pid=$!
echo "$pid" > "$out_dir/llama-server.pid"

waited=0
load_timeout="${BONGO_PROFILE_LOAD_TIMEOUT:-1500}"
while (( waited < load_timeout )); do
  healthy && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 5; waited=$(( waited + 5 ))
done
if ! healthy; then
  log "$label: server failed to become healthy"
  tail -n 8 "$out_dir/llama-server.log" >&2
  kill "$pid" 2>/dev/null || true
  printf '{"label":"%s","fatal_error":"server load timeout or crash"}\n' "$label" > "$out_dir/FAILED.json"
  exit 3
fi
log "$label: healthy after ${waited}s; profiling prefixes=$prefixes"

rc=0
python3 bench/profile-warm-prefix.py \
  --prefixes "$prefixes" --deltas 512 --ctx 131072 \
  --out "$profile_out" --label "$label" \
  --server-pid "$pid" \
  --flags-note "engine=llama.cpp b11223 Vulkan tier=$tier ctx=131072 flags=--load-mode none + M4.2 upload levers; extra_env='$extra_env'; extra_flags='$extra_flags'" \
  >> "$out_dir/harness.log" 2>&1 || rc=$?

# Correctness guard: the same server, same weights, same split.  A placement or
# offload lever only changes where the expert bytes come from, so the recall
# answer must be unchanged.
if [[ "${BONGO_M44_TURN_NEEDLE:-1}" == "1" && "$rc" == "0" ]]; then
  python3 bench/check-needle.py --prefix "${BONGO_M44_TURN_NEEDLE_PREFIX:-8192}" --ctx 131072 \
    --out "$out_dir/needle.json" --label "$label" --server-pid "$pid" \
    --flags-note "flags=--load-mode none + M4.2 upload levers; extra_env='$extra_env'; extra_flags='$extra_flags'" \
    >> "$out_dir/harness.log" 2>&1 || rc=$?
fi

kill "$pid" 2>/dev/null || true
for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
kill -9 "$pid" 2>/dev/null || true
log "$label: done (rc=$rc); results in $out_dir"
exit $rc
