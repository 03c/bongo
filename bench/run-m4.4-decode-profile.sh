#!/usr/bin/env bash
# M4.4 (BAS-159) — 4K decode GPU profile and lever screen.
#
# Decode is the last unmet target (>=25 tok/s at a 4096-token context, measured
# ~18).  M4.1/M4.2 showed it is flat to the host-upload fix, so it is bound by
# the GPU decode graph.  This runner does two things, both under the shared
# single-GPU flock (BAS-80):
#
#   1. ``perf`` configs start llama-server with ``GGML_VK_PERF_LOGGER=1`` and run
#      one 4096-token prompt + a short decode, so the per-op Vulkan device
#      timestamps can be split into MoE experts / dense / attention / DeltaNet /
#      norms for a single decode step.  Parse with ``bench/analyze-decode-perf.py``.
#   2. ``decode`` configs run the shipped 4K decode measurement (4096-token
#      prompt, N generated tokens, 3 repeats) so a lever can be compared on the
#      real tok/s number.
#
# One server restart per config, exactly one changed variable per config, so the
# profile table and every ablation are attributable.
#
# Usage:
#   bench/run-m4.4-decode-profile.sh                       # default config list
#   bench/run-m4.4-decode-profile.sh profile16 decode16    # named configs
#   BONGO_M44_CONFIGS=profile16,decode16 bench/run-m4.4-decode-profile.sh
#
# Environment:
#   BONGO_M44_OUT        results root (default bench/results/2026-09-29-m4.4-decode)
#   BONGO_M44_CONFIGS    comma list of config names (default: see ALL_CONFIGS)
#   BONGO_M44_CTX        prompt tokens (default 4096)
#   BONGO_M44_MAX        generated tokens per measured run (default 128)
#   BONGO_M44_PERF_MAX   generated tokens for a perf config (default 16)
#   BONGO_M44_REPS       measured repeats (default 3)
#   BONGO_LLAMA_SERVER   server binary (default the M4.2-lever engine build)
#   BONGO_PROFILE_LOAD_TIMEOUT  seconds to wait for model load (default 1500)
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
out_root="${BONGO_M44_OUT:-bench/results/2026-09-29-m4.4-decode}"
ctx_tokens="${BONGO_M44_CTX:-4096}"
max_tokens="${BONGO_M44_MAX:-128}"
perf_max_tokens="${BONGO_M44_PERF_MAX:-16}"
reps="${BONGO_M44_REPS:-3}"
base_url="http://$host:$port/v1"

log() { printf '[m4.4 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

# ---------------------------------------------------------------------------
# Config table.  Every config keeps the shipped decode flags and the M4.2
# levers; a config changes exactly one thing on top of that.
# ---------------------------------------------------------------------------
declare -A CFG_CTX CFG_FLAGS CFG_ENV CFG_PERF CFG_REPS CFG_NOTE

# Shipped decode flags, minus the lever-under-test.  --load-mode none is the
# M4.1 lever; the two env vars are the M4.2 upload levers.
M42_ENV="GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1"

CFG_CTX[profile16]="131072"; CFG_FLAGS[profile16]=""; CFG_ENV[profile16]="$M42_ENV"
CFG_PERF[profile16]=1; CFG_REPS[profile16]=1
CFG_NOTE[profile16]="shipped decode config (--n-cpu-moe 16, --load-mode none) + GGML_VK_PERF_LOGGER"

CFG_CTX[decode16]="131072"; CFG_FLAGS[decode16]=""; CFG_ENV[decode16]="$M42_ENV"
CFG_PERF[decode16]=0; CFG_REPS[decode16]=$reps
CFG_NOTE[decode16]="shipped decode config + M4.2 upload levers (the M4.4 reference)"

CFG_CTX[decode_nc12]="131072"; CFG_FLAGS[decode_nc12]="--n-cpu-moe 12"; CFG_ENV[decode_nc12]="$M42_ENV"
CFG_PERF[decode_nc12]=0; CFG_REPS[decode_nc12]=$reps
CFG_NOTE[decode_nc12]="placement: 4 more expert layers on the GPU (less per-token host upload)"

CFG_CTX[decode_nc8]="8192"; CFG_FLAGS[decode_nc8]="--n-cpu-moe 8"; CFG_ENV[decode_nc8]="$M42_ENV"
CFG_PERF[decode_nc8]=0; CFG_REPS[decode_nc8]=$reps
CFG_NOTE[decode_nc8]="decode-optimal placement probe: n-cpu-moe 8 at a small 8K context only"

CFG_CTX[decode_forcemmvq]="131072"; CFG_FLAGS[decode_forcemmvq]=""; CFG_ENV[decode_forcemmvq]="$M42_ENV GGML_VK_FORCE_MMVQ=1"
CFG_PERF[decode_forcemmvq]=0; CFG_REPS[decode_forcemmvq]=$reps
CFG_NOTE[decode_forcemmvq]="force the explicit mul_mat_vec path for every dense matmul"

CFG_CTX[decode_notq]="131072"; CFG_FLAGS[decode_notq]=""; CFG_ENV[decode_notq]="GGML_VK_HOST_BUFT_PER_DEVICE=1"
CFG_PERF[decode_notq]=0; CFG_REPS[decode_notq]=$reps
CFG_NOTE[decode_notq]="transfer-queue switch off: does the decode path overlap the expert copies?"

ALL_CONFIGS="profile16 decode16"

want="${BONGO_M44_CONFIGS:-$ALL_CONFIGS}"
shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
[[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }
(( ${#shards[@]} > 0 )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
model="${shards[0]}"

IFS=',' read -r -a configs <<<"$want"

# Dry run: print the plan without taking the GPU.
if [[ "${BONGO_M44_DRY_RUN:-0}" == "1" ]]; then
  for name in "${configs[@]}"; do
    [[ -n "${CFG_CTX[$name]:-}" ]] || { log "unknown config $name"; exit 2; }
    printf '%-20s ctx=%-7s perf=%s flags=%s env=%s\n' \
      "$name" "${CFG_CTX[$name]}" "${CFG_PERF[$name]}" "${CFG_FLAGS[$name]:-<shipped>}" "${CFG_ENV[$name]:-}"
  done
  exit 0
fi

bongo_gpu_lock_acquire "run-m4.4-decode-profile (BAS-159)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
if healthy; then
  log "ERROR: port $port already serving after acquiring the GPU lock; refusing to contaminate."
  exit 3
fi

stop_server() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$pid" 2>/dev/null || true
}

run_config() {
  local name="$1"
  local out_dir="$out_root/$name"
  local perf="${CFG_PERF[$name]}"
  local cfg_ctx="${CFG_CTX[$name]}"
  local flags="${CFG_FLAGS[$name]}"
  local env_flags="${CFG_ENV[$name]}"
  local nreps="${CFG_REPS[$name]}"
  mkdir -p "$out_dir"
  local server_log="$out_dir/llama-server.log"

  # shellcheck disable=SC2206
  local extra=($flags)
  local argv=(--model "$model" --ctx-size "$cfg_ctx" --jinja
    --cache-type-k q8_0 --cache-type-v q8_0
    --host "$host" --port "$port" --parallel 1 --alias "bongo-$tier"
    --metrics --device "$device" --spec-type none)
  if [[ "$flags" != *"--flash-attn"* ]]; then argv+=(--flash-attn on); fi
  if [[ "$flags" != *"--n-cpu-moe"* ]]; then argv+=(--n-cpu-moe 16); fi
  if [[ "$flags" != *"--n-gpu-layers"* ]]; then argv+=(--n-gpu-layers 99); fi
  if [[ "$flags" != *"--load-mode"* ]]; then argv+=(--load-mode none); fi
  argv+=("${extra[@]}")

  local marker="$out_dir/decode4k.json"
  [[ "$perf" == "1" ]] && marker="$out_dir/vk-perf-decode.json"
  if [[ -f "$marker" ]]; then
    log "$name already measured ($marker); skipping"
    return 0
  fi

  printf '%s\n' "$llama_bin ${argv[*]}" > "$out_dir/command.txt"
  printf '%s\n' "$env_flags" > "$out_dir/env.txt"
  printf '%s\n' "${CFG_NOTE[$name]}" > "$out_dir/note.txt"

  log "$name: starting llama-server (ctx=$cfg_ctx perf=$perf)"
  if [[ "$perf" == "1" ]]; then
    # shellcheck disable=SC2086
    env $env_flags GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
      setsid "$llama_bin" "${argv[@]}" > "$server_log" 2>&1 &
  else
    # shellcheck disable=SC2086
    env $env_flags setsid "$llama_bin" "${argv[@]}" > "$server_log" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$out_dir/llama-server.pid"

  local waited=0 load_timeout="${BONGO_PROFILE_LOAD_TIMEOUT:-1500}"
  while (( waited < load_timeout )); do
    healthy && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; waited=$(( waited + 5 ))
  done
  if ! healthy; then
    log "$name: server failed to become healthy"
    tail -n 8 "$server_log" >&2
    stop_server "$pid"
    printf '{"label":"%s","fatal_error":"server load timeout or crash"}\n' "$name" > "$out_dir/FAILED.json"
    return 3
  fi
  log "$name: healthy after ${waited}s"

  local rc=0
  if [[ "$perf" == "1" ]]; then
    # Two requests.  The first primes the slot with the full 4096-token prompt
    # (cache_prompt=false).  The second re-sends the same prompt with
    # cache_prompt=true, so llama.cpp reuses the cached KV and every graph it
    # computes is a one-token graph -- a decode step at a 4096-token context.
    # The analyzer attributes the perf blocks after this request's
    # ``launch_slot_`` line, which is the last launch in the log.
    python3 - "$tier" "$base_url" "$ctx_tokens" "$perf_max_tokens" "$out_dir" <<'PY' || rc=$?
import json, sys
sys.path.insert(0, "bench")
from bench_lib import Tokenizer, CORPUS
from harness import streaming_measure
tier, base, ctx_tokens, max_tokens, out = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
tk = Tokenizer(base, timeout=900)
prompt = tk.size_to(ctx_tokens, CORPUS, hard_max=ctx_tokens)
r0 = streaming_measure(base, f"bongo-{tier}", prompt, 1, 900, cache_prompt=False)
print(f"prime: status={r0.get('status')} prompt_tokens={r0.get('prompt_tokens')} "
      f"prompt_ms={r0.get('prompt_ms')}")
r = streaming_measure(base, f"bongo-{tier}", prompt, max_tokens, 900, cache_prompt=True)
rec = {"label": out, "ctx_tokens": ctx_tokens, "max_tokens": max_tokens,
       "prime": r0, "run": r,
       "decode_steps": (r.get("output_tokens") or 0) + 1}
json.dump(rec, open(f"{out}/perf-run.json", "w"), indent=2)
print(f"profile run: status={r.get('status')} cache_n={r.get('cache_n')} "
      f"prompt_tokens={r.get('prompt_tokens')} prompt_ms={r.get('prompt_ms')} "
      f"output_tokens={r.get('output_tokens')} output_tps={r.get('output_tps')}")
PY
  else
    python3 - "$tier" "$base_url" "$ctx_tokens" "$max_tokens" "$nreps" "$out_dir/decode4k.json" <<'PY' || rc=$?
import json, sys
sys.path.insert(0, "bench")
from bench_lib import Tokenizer, CORPUS
from harness import streaming_measure
tier, base, ctx_tokens, max_tokens, reps, out = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
tk = Tokenizer(base, timeout=900)
prompt = tk.size_to(ctx_tokens, CORPUS, hard_max=ctx_tokens)
recs = []
# discarded warmup: page-cache warm, not measured
streaming_measure(base, f"bongo-{tier}", prompt, 8, 900, cache_prompt=False)
for i in range(reps):
    r = streaming_measure(base, f"bongo-{tier}", prompt, max_tokens, 900, cache_prompt=False)
    recs.append(r)
    print(f"run{i}: prompt_ms={r.get('prompt_ms')} output_tokens={r.get('output_tokens')} "
          f"output_tps={r.get('output_tps')} status={r.get('status')}")
json.dump({"label": out, "ctx_tokens": ctx_tokens, "max_tokens": max_tokens, "runs": recs},
          open(out, "w"), indent=2)
PY
  fi

  stop_server "$pid"
  log "$name: done (rc=$rc)"
  return $rc
}

overall=0
for name in "${configs[@]}"; do
  [[ -n "${CFG_CTX[$name]:-}" ]] || { log "unknown config $name"; overall=2; continue; }
  run_config "$name" || overall=$?
done
log "done; results in $out_root"
exit $overall
