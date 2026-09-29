#!/usr/bin/env bash
# M4.2 (BAS-155) 4K decode A/B.
#
# The primary M4.2 target is the 512-token warm-prefix delta turn, but the
# milestone also reports the 4K decode change (target >= 25 tok/s).  This runner
# measures a 4096-token prompt with 128 generated tokens for two configs, one
# server restart each, in one GPU-lock hold (BAS-80):
#
#   baseline  `--load-mode none` (device-0 host expert buffer, the M4.1 lever)
#   lever     `--load-mode none` + device-local host expert buffer + transfer queue
#
# Results are written under <out>/<label>/{decode4k.json,llama-server.log,flags.txt}.
#
# Usage:
#   bench/run-m4.2-decode4k.sh
#   BONGO_LLAMA_SERVER=<dir>/llama-server bench/run-m4.2-decode4k.sh
#
# Environment:
#   BONGO_DECODE4K_OUT   results root (default bench/results/2026-09-29-m4.2-upload/decode4k)
#   BONGO_DECODE4K_CTX   prompt tokens (default 4096)
#   BONGO_DECODE4K_MAX   generated tokens (default 128)
#   BONGO_DECODE4K_REPS  measured repeats (default 3)
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
out_root="${BONGO_DECODE4K_OUT:-bench/results/2026-09-29-m4.2-upload/decode4k}"
ctx_tokens="${BONGO_DECODE4K_CTX:-4096}"
max_tokens="${BONGO_DECODE4K_MAX:-128}"
reps="${BONGO_DECODE4K_REPS:-3}"
base_url="http://$host:$port/v1"

log() { printf '[decode4k %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
[[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }
(( ${#shards[@]} > 0 )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
model="${shards[0]}"

bongo_gpu_lock_acquire "run-m4.2-decode4k (BAS-155)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
if healthy; then
  log "ERROR: port $port already serving after acquiring the GPU lock; refusing to contaminate."
  exit 3
fi

run_config() {
  local label="$1"; shift
  local out_dir="$out_root/$label"
  mkdir -p "$out_dir"
  local server_log="$out_dir/llama-server.log"
  local argv=(--model "$model" --ctx-size 131072 --jinja
    --cache-type-k q8_0 --cache-type-v q8_0
    --host "$host" --port "$port" --parallel 1 --alias "bongo-$tier"
    --metrics --device "$device" --spec-type none
    --flash-attn on --n-cpu-moe 16 --n-gpu-layers 99 --load-mode none)

  if [[ -f "$out_dir/decode4k.json" ]]; then
    log "$label already measured; skipping"
    return 0
  fi

  printf '%s\n' "$llama_bin ${argv[*]}" > "$out_dir/command.txt"
  printf '%s\n' "$*" > "$out_dir/env.txt"

  log "$label: starting llama-server"
  # shellcheck disable=SC2086
  env "$@" setsid "$llama_bin" "${argv[@]}" > "$server_log" 2>&1 &
  local pid=$!
  echo "$pid" > "$out_dir/llama-server.pid"

  local waited=0 load_timeout="${BONGO_PROFILE_LOAD_TIMEOUT:-1500}"
  while (( waited < load_timeout )); do
    healthy && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; waited=$(( waited + 5 ))
  done
  if ! healthy; then
    log "$label: server failed to become healthy"
    tail -n 5 "$server_log" >&2
    kill "$pid" 2>/dev/null || true
    return 3
  fi
  log "$label: healthy after ${waited}s"

  python3 - "$tier" "$base_url" "$ctx_tokens" "$max_tokens" "$reps" "$out_dir/decode4k.json" <<'PY'
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
  local rc=$?

  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$pid" 2>/dev/null || true
  return $rc
}

run_config baseline
run_config lever GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1
log "done; results in $out_root"
