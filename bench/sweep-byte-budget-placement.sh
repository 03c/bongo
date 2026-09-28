#!/usr/bin/env bash
# Stage-1 re-run for the BAS-76 byte-budget `-ot` expert placement.
#
# Replaces the `--n-cpu-moe N` layer-count rule with a cheapest-layer-first byte
# budget expressed as one llama.cpp `--override-tensor` pattern (the complement
# of the resident layer set is offloaded to the CPU).  It launches llama-server
# directly, so it does not depend on bongo.sh and cannot be confused with the
# concurrent prefix-cache work on the same branch.
#
# The Stage 0 Vulkan baseline (`--n-cpu-moe 16`) stays pinned and selectable:
# run this script with `--n-cpu-moe 16` to measure it, or compare against the
# already-recorded `bench/results/2026-09-27-expert-placement/ncmoe-16/`.
#
# Output:
#   bench/results/2026-09-28-byte-budget-placement/<label>/
#     placement.json          the placement spec + coverage
#     server-flags.json       the exact llama-server argv
#     placement-after-load.json  VRAM attributed to the server after load
#     harness.log, matrix.json, matrix.md, raw/
#
# Usage:
#   bench/sweep-byte-budget-placement.sh            # run the byte-budget config
#   bench/sweep-byte-budget-placement.sh --dry-run  # print the plan only
#   bench/sweep-byte-budget-placement.sh --n-cpu-moe 16   # the pinned baseline
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
# shellcheck source=bench/gpu-lock.sh
. "$repo/bench/gpu-lock.sh"

tier="${BONGO_SWEEP_TIER:-iq2_xs}"
budget_gib="${BONGO_BUDGET_GIB:-22.40}"
contexts="${BONGO_SWEEP_CONTEXTS:-4096,131072}"
bongo_home="${BONGO_HOME:-$HOME/.bongo}"
llama_bin="$bongo_home/llama/b11223/vulkan/llama-server"
gguf_dir="$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier"
ctx="${BONGO_CTX:-131072}"
host="127.0.0.1"
port="${BONGO_PORT:-8080}"
device="Vulkan1"
out_root="${BONGO_SWEEP_OUT:-bench/results/2026-09-28-byte-budget-placement}"
expert_bytes="bench/results/2026-09-27-expert-placement/expert-bytes-$tier.json"
analysis="bench/results/2026-09-28-expert-activation/analysis.json"
n_cpu_moe_override=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift;;
    --n-cpu-moe) n_cpu_moe_override="${2:?--n-cpu-moe needs a value}"; shift 2;;
    --budget-gib) budget_gib="${2:?--budget-gib needs a value}"; shift 2;;
    --contexts) contexts="${2:?--contexts needs a value}"; shift 2;;
    *) echo "unknown option '$1'" >&2; exit 2;;
  esac
done

log() { printf '[bb-placement] %s\n' "$*" >&2; }

base_url="http://$host:$port/v1"
pidfile="$bongo_home/run/llama-server-bb.pid"
server_log="$bongo_home/run/llama-server-bb.log"

healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

if (( ! dry_run )) && healthy; then
  log "ERROR: port $port is already serving (another run holds the GPU)."
  log "       Stop it or set BONGO_PORT to a free port; refusing to contaminate a measurement."
  exit 3
fi

[[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }

shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
if (( ${#shards[@]} == 0 )); then log "ERROR: no GGUF shards in $gguf_dir"; exit 2; fi
model="${shards[0]}"

mkdir -p "$out_root"

if [[ -n "$n_cpu_moe_override" ]]; then
  label="ncmoe-$n_cpu_moe_override"
  placement_json=""
  override_tensor=""
  n_cpu_moe="$n_cpu_moe_override"
  log "pinned layer rule: --n-cpu-moe $n_cpu_moe"
else
  label="byte-budget-$(printf '%s' "$budget_gib" | tr '.' '_')"
  placement_json="$out_root/placement-$tier-$budget_gib.json"
  python3 bench/gen-ot-placement.py \
    --expert-bytes "$expert_bytes" \
    --analysis "$analysis" \
    --budget-gib "$budget_gib" \
    --tier "$tier" \
    --out "$placement_json" || exit 2
  override_tensor="$(python3 bench/gen-ot-placement.py --expert-bytes "$expert_bytes" \
    --budget-gib "$budget_gib" --print-arg)"
  n_cpu_moe=0
  log "byte budget: resident complement offloaded to CPU"
  log "  -ot $override_tensor"
fi

out_dir="$out_root/$label"
mkdir -p "$out_dir"
if [[ -n "$placement_json" ]]; then cp "$placement_json" "$out_dir/placement.json"; fi

flags=(--model "$model" --ctx-size "$ctx" --jinja --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0
  --n-gpu-layers 99 --n-cpu-moe "$n_cpu_moe")
if [[ -n "$override_tensor" ]]; then flags+=(--override-tensor "$override_tensor"); fi
flags+=(--host "$host" --port "$port" --parallel 1 --alias "bongo-$tier" --metrics --device "$device")

python3 - "$out_dir/server-flags.json" "${flags[@]}" <<'PY'
import json, sys
out = sys.argv[1]
json.dump({"argv": sys.argv[2:]}, open(out, "w"), indent=2)
print("wrote " + out)
PY

if (( dry_run )); then
  log "dry-run; would run:"
  printf '%q ' "$llama_bin" "${flags[@]}" >&2; printf '\n' >&2
  exit 0
fi

vram_bytes() {
  local pid="$1"
  [[ -n "$pid" ]] || { echo ""; return 0; }
  python3 - "$pid" <<'PY'
import glob, sys
pid = sys.argv[1]
total = 0; found = False
for fd in glob.glob(f"/proc/{pid}/fdinfo/*"):
    try:
        data = open(fd).read()
    except OSError:
        continue
    for line in data.splitlines():
        if line.startswith(("drm-resident-vram0:", "drm-total-vram0:")):
            try:
                kib = float(line.split(":", 1)[1].split()[0])
            except (IndexError, ValueError):
                continue
            total = max(total, kib * 1024); found = True
print(int(total) if found else "")
PY
}

log "starting llama-server ($label); log: $server_log"
# Serialise on the single GPU for the whole measured run (BAS-80).
bongo_gpu_lock_acquire "sweep-byte-budget-placement $label" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
setsid "$llama_bin" "${flags[@]}" > "$server_log" 2>&1 &
echo $! > "$pidfile"
pid="$(cat "$pidfile")"

waited=0
started=0
while (( waited < 600 )); do
  if healthy; then started=1; break; fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 5; waited=$(( waited + 5 ))
done

if (( started == 0 )); then
  cp "$server_log" "$out_dir/server-start.log" 2>/dev/null || true
  why="$(grep -iE 'out of memory|failed to allocate|error|abort|ggml_vk|vk::' "$server_log" | tail -n 3 | tr '\n' ' ' | cut -c1-400)"
  log "server failed to become healthy: ${why:-timeout}"
  kill "$pid" 2>/dev/null || true
  python3 - "$out_dir/matrix.json" "$tier" "$label" "${why:-timeout}" <<'PY'
import json, sys, datetime
out, tier, label, why = sys.argv[1:5]
json.dump({"schema": "bongo.byte-budget-placement.v1", "tier": tier, "label": label,
           "fit": False, "fatal_error": {"stage": "server load", "message": why},
           "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open(out, "w"), indent=2)
PY
  exit 3
fi

log "server healthy after ${waited}s; pid=$pid"
vram="$(vram_bytes "$pid" || true)"
printf '{"label": "%s", "vram_resident_bytes_after_load": %s}\n' "$label" "${vram:-null}" \
  > "$out_dir/placement-after-load.json"
log "VRAM after load: ${vram:-unknown} bytes"

log "warming the page cache with a discarded 4K prefill"
python3 - "$tier" "$base_url" <<'PY' >> "$out_dir/harness.log" 2>&1 || log "warmup failed; continuing"
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

log "running harness at contexts=$contexts"
python3 bench/harness.py \
  --repo-root "$repo" \
  --tier "$tier" \
  --contexts "$contexts" \
  --repeats 1 \
  --max-tokens 128 \
  --needle-context 131072 \
  --hash-mode none \
  --out-dir "$out_dir" \
  >> "$out_dir/harness.log" 2>&1
rc=$?
log "harness exit $rc"

if [[ -n "$placement_json" ]]; then
  log "running measure-prefix-cache.py (held-out agentic turn path)"
  python3 bench/measure-prefix-cache.py --out "$out_dir/prefix-cache" \
    >> "$out_dir/harness.log" 2>&1 || log "prefix-cache measurement failed; continuing"
fi

kill "$pid" 2>/dev/null || true
for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
kill -9 "$pid" 2>/dev/null || true
rm -f "$pidfile"
log "done ($label); results in $out_dir"
exit $rc
