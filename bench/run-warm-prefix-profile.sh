#!/usr/bin/env bash
# M3.6 warm-prefix (cached-turn) profile — BAS-130.
#
# Runs ``bench/profile-warm-prefix.py`` against a freshly-started llama-server
# for each *ablation config*, so the delta-turn cost can be attributed to a
# component (attention / MoE experts / GatedDeltaNet / hyper-connections /
# batch / host) instead of one profiler dump.  Every config is a server restart
# with one changed flag; the measured workload is identical.
#
# The Stage 0 Vulkan baseline stays pinned: the ``baseline`` config is exactly
# the shipped flags (`--n-cpu-moe 16`, q8 KV, Vulkan, flash-attn on).  Ablations
# only *change* one thing relative to it.
#
# The single GPU is serialised with the shared flock (BAS-80) for the whole
# measured run.  Resumable: a config whose output JSON already exists is skipped.
#
# Usage:
#   bench/run-warm-prefix-profile.sh                 # all configs, in order
#   bench/run-warm-prefix-profile.sh baseline        # one config (name)
#   bench/run-warm-prefix-profile.sh baseline ncmoe24
#   BONGO_PROFILE_CONFIGS=baseline,fa_off bench/run-warm-prefix-profile.sh
#   BONGO_PROFILE_DRY_RUN=1 bench/run-warm-prefix-profile.sh
#
# Environment:
#   BONGO_PROFILE_OUT      results root (default bench/results/2026-09-28-warm-prefix-profile)
#   BONGO_PROFILE_CONFIGS  comma list of config names to run (default: all)
#   BONGO_PROFILE_PREFIXES override prefixes for every config (comma list)
#   BONGO_PROFILE_DELTAS   override deltas (default 128,512,1024)
#   BONGO_GPU_LOCK_TIMEOUT flock wait, seconds (default 0 = fail fast)
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
out_root="${BONGO_PROFILE_OUT:-bench/results/2026-09-28-warm-prefix-profile}"
deltas="${BONGO_PROFILE_DELTAS:-128,512,1024}"
dry_run="${BONGO_PROFILE_DRY_RUN:-0}"

base_url="http://$host:$port/v1"

# ---------------------------------------------------------------------------
# Config table.  name | ctx-size | prefixes | extra server flags
# ---------------------------------------------------------------------------
# The single changed variable per config is the point; do not stack changes.
declare -A CFG_CTX CFG_PREFIXES CFG_FLAGS CFG_NOTE CFG_PERF

CFG_CTX[baseline]="131072"; CFG_PREFIXES[baseline]="16384,128000"
CFG_FLAGS[baseline]=""
CFG_NOTE[baseline]="shipped Stage 0 Vulkan: --n-cpu-moe 16, q8 KV, flash-attn on"

CFG_CTX[baseline_perf]="131072"; CFG_PREFIXES[baseline_perf]="16384"
CFG_FLAGS[baseline_perf]=""
CFG_NOTE[baseline_perf]="shipped baseline with GGML_VK_PERF_LOGGER=1 (per-op Vulkan timings)"
CFG_PERF[baseline_perf]=1

CFG_CTX[ncmoe24]="131072"; CFG_PREFIXES[ncmoe24]="16384"
CFG_FLAGS[ncmoe24]="--n-cpu-moe 24"
CFG_NOTE[ncmoe24]="8 more MoE layers' experts moved GPU->CPU (placement)"

CFG_CTX[ncmoe8]="131072"; CFG_PREFIXES[ncmoe8]="16384"
CFG_FLAGS[ncmoe8]="--n-cpu-moe 8"
CFG_NOTE[ncmoe8]="8 fewer MoE layers' experts on CPU, i.e. more experts on GPU (placement)"

CFG_CTX[fa_off]="131072"; CFG_PREFIXES[fa_off]="16384"
CFG_FLAGS[fa_off]="--flash-attn off"
CFG_NOTE[fa_off]="flash-attn disabled: isolates the 12 full-attention layers' kernel"

CFG_CTX[ub128]="131072"; CFG_PREFIXES[ub128]="16384"
CFG_FLAGS[ub128]="--ubatch-size 128"
CFG_NOTE[ub128]="physical batch 128 x4 over the 512-token delta (batch/chunking)"

CFG_CTX[ub1024]="131072"; CFG_PREFIXES[ub1024]="16384"
CFG_FLAGS[ub1024]="--ubatch-size 1024"
CFG_NOTE[ub1024]="physical batch 1024 for a 512-token delta (one large batch)"

CFG_CTX[attn_cpu]="131072"; CFG_PREFIXES[attn_cpu]="16384"
CFG_FLAGS[attn_cpu]="-ot attn_qkv=CPU,attn_output=CPU,attn_gate=CPU,attn_q=CPU,attn_k=CPU,attn_v=CPU"
CFG_NOTE[attn_cpu]="full-attention weights/compute forced to CPU (12 layers)"

CFG_CTX[ssm_cpu]="131072"; CFG_PREFIXES[ssm_cpu]="16384"
CFG_FLAGS[ssm_cpu]="-ot ssm_out=CPU,ssm_conv1d=CPU,ssm_alpha=CPU,ssm_beta=CPU"
CFG_NOTE[ssm_cpu]="GatedDeltaNet (recurrent) weights forced to CPU (36 layers)"

CFG_CTX[hc_cpu]="131072"; CFG_PREFIXES[hc_cpu]="16384"
CFG_FLAGS[hc_cpu]="-ot hc_.*=CPU"
CFG_NOTE[hc_cpu]="hyper-connection tensors forced to CPU (48 layers, BF16)"

ALL_CONFIGS="baseline baseline_perf ncmoe24 ncmoe8 fa_off ub128 ub1024 attn_cpu ssm_cpu hc_cpu"

log() { printf '[warm-prefix %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

stop_server() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
if (( ! dry_run )); then
  [[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }
  (( ${#shards[@]} > 0 )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
fi
model="${shards[0]:-}"

selected="${BONGO_PROFILE_CONFIGS:-${*:-$ALL_CONFIGS}}"
selected="${selected//,/ }"

if [[ "$dry_run" == "1" ]]; then
  for name in $selected; do
    echo "config=$name ctx=${CFG_CTX[$name]:-?} prefixes=${BONGO_PROFILE_PREFIXES:-${CFG_PREFIXES[$name]:-?}} deltas=$deltas"
    echo "  flags: ${CFG_FLAGS[$name]:-}"
    echo "  note:  ${CFG_NOTE[$name]:-}"
  done
  exit 0
fi

bongo_gpu_lock_acquire "run-warm-prefix-profile (BAS-130)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
# Now that the lock is held the lock holder's server must be gone. A server
# still answering here is a stale one that did not hold the lock; refuse.
if healthy; then
  log "ERROR: port $port is already serving after acquiring the GPU lock. Refusing to contaminate a measurement."
  exit 3
fi

run_config() {
  local name="$1"
  local out_dir="$out_root/$name"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/profile.json" ]]; then
    log "$name already measured ($out_dir/profile.json); skipping"
    return 0
  fi
  local ctx="${CFG_CTX[$name]:?unknown config $name}"
  local prefixes="${BONGO_PROFILE_PREFIXES:-${CFG_PREFIXES[$name]}}"
  local flags="${CFG_FLAGS[$name]:-}"
  local note="${CFG_NOTE[$name]:-}"
  local server_log="$out_dir/llama-server.log"
  local pidfile="$out_dir/llama-server.pid"

  # shellcheck disable=SC2206
  local extra=($flags)
  local perf="${CFG_PERF[$name]:-0}"
  local argv=(--model "$model" --ctx-size "$ctx" --jinja
    --cache-type-k q8_0 --cache-type-v q8_0
    --host "$host" --port "$port" --parallel 1 --alias "bongo-$tier"
    --metrics --device "$device" --spec-type none)
  # The base flags are added only when the ablation does not redefine them, so a
  # config changes exactly one thing relative to the shipped baseline.
  if [[ "$flags" != *"--flash-attn"* ]]; then argv+=(--flash-attn on); fi
  if [[ "$flags" != *"--n-cpu-moe"* ]]; then argv+=(--n-cpu-moe 16); fi
  if [[ "$flags" != *"--n-gpu-layers"* ]]; then argv+=(--n-gpu-layers 99); fi
  argv+=("${extra[@]}")

  python3 - "$out_dir/server-flags.json" "$perf" "${argv[@]}" <<'PY'
import json, sys
json.dump({"perf_logger": sys.argv[2] == "1", "argv": sys.argv[3:]}, open(sys.argv[1], "w"), indent=2)
PY

  log "$name: starting llama-server (ctx=$ctx perf=$perf) flags: $flags"
  if [[ "$perf" == "1" ]]; then
    GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
      setsid "$llama_bin" "${argv[@]}" > "$server_log" 2>&1 &
  else
    setsid "$llama_bin" "${argv[@]}" > "$server_log" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$pidfile"

  local waited=0
  local started=0
  while (( waited < 900 )); do
    if healthy; then started=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 5; waited=$(( waited + 5 ))
  done
  if (( started == 0 )); then
    local why
    why="$(grep -iE 'out of memory|failed to allocate|error|abort|ggml_vk|vk::' "$server_log" | tail -n 3 | tr '\n' ' ' | cut -c1-400)"
    log "$name: server failed to become healthy: ${why:-timeout}"
    stop_server "$pid"
    python3 - "$out_dir/profile.json" "$name" "$tier" "$flags" "${why:-timeout}" <<'PY'
import json, sys, datetime
out, name, tier, flags, why = sys.argv[1:6]
json.dump({"schema": "bongo.warm-prefix-profile.v1", "label": name, "tier": tier,
           "flags": flags, "fit": False, "fatal_error": {"stage": "server load", "message": why},
           "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open(out, "w"), indent=2)
PY
    return 3
  fi
  log "$name: server healthy after ${waited}s; pid=$pid"

  # Warm the CPU-resident experts' page cache with a discarded 4K prefill so the
  # measured delta is not a cold-page number (the sweep scripts do the same).
  python3 - "$tier" "$base_url" <<'PY' >> "$out_dir/harness.log" 2>&1 || log "$name: warmup prefill failed; continuing"
import sys
sys.path.insert(0, "bench")
from bench_lib import Tokenizer, CORPUS, post_json
tier, base = sys.argv[1], sys.argv[2]
tk = Tokenizer(base, timeout=900)
prompt = tk.size_to(4096, CORPUS, hard_max=4096)
res = post_json(base, "/completions", {"model": f"bongo-{tier}", "prompt": prompt,
                                       "max_tokens": 1, "temperature": 0.0,
                                       "cache_prompt": False}, 900)
print(f"warmup prefill status={res.status}")
PY

  local note_full="engine=llama.cpp b11223 backend=Vulkan tier=$tier ctx=$ctx flags=${flags:-<baseline>} note=$note"
  log "$name: profiling prefixes=$prefixes deltas=$deltas"
  python3 bench/profile-warm-prefix.py \
    --prefixes "$prefixes" --deltas "$deltas" --ctx "$ctx" \
    --out "$out_dir/profile.json" --label "$name" \
    --server-pid "$pid" --flags-note "$note_full" \
    >> "$out_dir/harness.log" 2>&1
  local rc=$?
  log "$name: profile exit $rc"

  stop_server "$pid"
  # Keep the last server log for the record.
  cp "$server_log" "$out_dir/llama-server.log" 2>/dev/null || true
  return $rc
}

rc_total=0
for name in $selected; do
  if [[ -z "${CFG_CTX[$name]:-}" ]]; then
    log "unknown config '$name'; skipping"
    rc_total=2
    continue
  fi
  run_config "$name" || rc_total=$?
done
log "done; results in $out_root"
exit $rc_total
