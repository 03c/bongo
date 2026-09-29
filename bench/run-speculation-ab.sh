#!/usr/bin/env bash
# Run the suffix/n-gram speculation A/B against the pinned Stage 0 backend.
#
#   ./bench/run-speculation-ab.sh baseline
#   ./bench/run-speculation-ab.sh spec        --spec-type ngram-map-k4v
#   ./bench/run-speculation-ab.sh synth-high  --spec-type ngram-mod --spec-synth-len 5.0
#   ./bench/run-speculation-ab.sh compare     --baseline baseline.json --spec spec.json
#
# It starts its own llama-server with the shipped Vulkan `--n-cpu-moe 16` flags
# plus the requested speculation flags, waits for health, runs
# `bench/measure-speculation.py run`, then stops only the server it started.
# The GPU is exclusive: do not run two of these at once.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "${here}")"
# shellcheck source=bench/gpu-lock.sh
. "$here/gpu-lock.sh"
# The server log is read after the server stops to aggregate the draft
# acceptance, and the run is detached (it queues behind the single-GPU flock and
# outlives the heartbeat that launched it). Keep it under BONGO_SPEC_SCRATCH
# when set, because PAPERCLIP_RUN_SCRATCH_DIR is reaped when the run ends.
scratch="${BONGO_SPEC_SCRATCH:-${PAPERCLIP_RUN_SCRATCH_DIR:-${PAPERCLIP_SCRATCH_DIR:-$(mktemp -d)}}}"
mkdir -p "$scratch"

BONGO_HOME="${BONGO_HOME:-$HOME/.bongo}"
REV="${BONGO_LLAMA_REV:-b11223}"
SERVER_BIN="${BONGO_LLAMA_SERVER:-$BONGO_HOME/llama/$REV/vulkan/llama-server}"
MODEL_DIR="${BONGO_MODEL_DIR:-$BONGO_HOME/models}"
MODEL="${BONGO_MODEL_FILE:-$MODEL_DIR/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf}"
HOST="${BONGO_HOST:-127.0.0.1}"
PORT="${BONGO_PORT:-8080}"
CTX="${BONGO_CTX:-131072}"
N_CPU_MOE="${BONGO_N_CPU_MOE:-16}"
DEVICE="${BONGO_DEVICE:-Vulkan1}"
OUT_DIR="${BONGO_SPEC_OUT:-$repo/bench/results/2026-09-28-speculation}"

label="${1:-}"; shift || true
[ -n "$label" ] || { echo "usage: $0 <label> [--spec-type TYPE] [--spec-synth-len L] [extra measure-speculation flags]" >&2; exit 2; }

if [ "$label" = "compare" ]; then
  exec python3 "$here/measure-speculation.py" compare "$@"
fi

spec_type="none"; spec_synth_len=""; spec_synth_rates=""; extra=()
contexts="${BONGO_SPEC_CONTEXTS:-4096,131072}"
max_tokens="${BONGO_SPEC_MAX_TOKENS:-128}"
repeats="${BONGO_SPEC_REPEATS:-2}"
workloads="${BONGO_SPEC_WORKLOADS:-generic}"

while [ $# -gt 0 ]; do
  case "$1" in
    --spec-type) spec_type="${2:?}"; shift 2;;
    --spec-synth-len) spec_synth_len="${2:?}"; shift 2;;
    --spec-synth-rates) spec_synth_rates="${2:?}"; shift 2;;
    --contexts) contexts="${2:?}"; shift 2;;
    --max-tokens) max_tokens="${2:?}"; shift 2;;
    --repeats) repeats="${2:?}"; shift 2;;
    --workloads) workloads="${2:?}"; shift 2;;
    --) shift; extra+=("$@"); break;;
    *) extra+=("$1"); shift;;
  esac
done

[ -x "$SERVER_BIN" ] || { echo "llama-server not found at $SERVER_BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found at $MODEL" >&2; exit 1; }

# Serialise on the single GPU for the whole measured run (BAS-80). Acquire the
# shared lock before the port/pid guards so contention names the lock holder.
bongo_gpu_lock_acquire "run-speculation-ab $label" || exit 3

if curl -sf -m 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "A server is already listening on http://$HOST:$PORT; refusing to disturb it." >&2
  echo "Stop it (or set BONGO_PORT) before running this A/B." >&2
  exit 3
fi
# The Arc has 32 GiB and the model needs ~30 GiB, so exactly one llama-server
# may run on this box. Refuse if a sibling run holds the device on another port,
# otherwise the second process device-losses and can take the first one with it.
if pgrep -x llama-server >/dev/null 2>&1; then
  echo "A llama-server is already running (another run holds the GPU); refusing to start a second one." >&2
  pgrep -a -x llama-server >&2 || true
  exit 3
fi

flags=(
  --model "$MODEL" --ctx-size "$CTX" --jinja --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0
  --n-gpu-layers 99 --n-cpu-moe "$N_CPU_MOE"
  --host "$HOST" --port "$PORT" --parallel 1 --alias bongo-iq2_xs
  --metrics --device "$DEVICE"
  --spec-type "$spec_type"
)
if [ -n "$spec_synth_len" ]; then flags+=(--spec-synth-len "$spec_synth_len"); fi
if [ -n "$spec_synth_rates" ]; then flags+=(--spec-synth-rates "$spec_synth_rates"); fi

log="$scratch/speculation-$label-server.log"
pidfile="$scratch/speculation-$label-server.pid"

"$SERVER_BIN" "${flags[@]}" >"$log" 2>&1 &
spid=$!
echo "$spid" > "$pidfile"
cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; bongo_gpu_lock_release; }
trap cleanup EXIT

echo "starting $label: spec-type=$spec_type synth-len=${spec_synth_len:-none}"
for i in $(seq 1 200); do
  if ! kill -0 "$spid" 2>/dev/null; then echo "server exited; tail:" >&2; tail -20 "$log" >&2; exit 1; fi
  if curl -sf -m 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then break; fi
  sleep 3
  if [ "$i" -ge 200 ]; then echo "server did not become healthy" >&2; tail -20 "$log" >&2; exit 1; fi
done
echo "server healthy; running measurement"

note="engine=llama.cpp $REV (4da6337767f973e2b4d0797e5b323d77d8565e4a) backend=Vulkan tier=iq2_xs n_cpu_moe=$N_CPU_MOE ctx=$CTX spec_type=$spec_type synth_len=${spec_synth_len:-none} flags=${flags[*]}"
IFS=',' read -ra wl_list <<< "$workloads"
for wl in "${wl_list[@]}"; do
  wl="${wl// /}"
  [ -n "$wl" ] || continue
  echo "measuring workload=$wl"
  python3 "$here/measure-speculation.py" run \
    --base-url "http://$HOST:$PORT/v1" --model bongo-iq2_xs --tier iq2_xs \
    --label "$label" --spec-type "$spec_type" --workload "$wl" \
    --spec-synth "${spec_synth_len:-${spec_synth_rates:-none}}" \
    --contexts "$contexts" --max-tokens "$max_tokens" --repeats "$repeats" \
    --context-limit-guard "$CTX" \
    --server-log "$log" --server-note "$note workload=$wl" \
    --out "$OUT_DIR/$label-$wl.json" "${extra[@]}"
done
echo "note: $note"
echo "saved $OUT_DIR/$label-{$(echo "$workloads" | tr ',' ',')}.json"
