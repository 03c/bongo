#!/usr/bin/env bash
# M3.4b PLE reader engine A/B.
#
# Starts the patched llama-server (Vulkan) on the pinned Stage-0 baseline config
# or the M3.3 byte-budget config, with `--ple-reader off|on`, and runs
# bench/harness.py at 4K + 128K.  Mirrors bench/sweep-byte-budget-placement.sh
# so the numbers are directly comparable, but points at the M3.4b binary.
#
# Usage:
#   bench/ple-reader/run-ab.sh --config baseline --reader off --label baseline-off
#   bench/ple-reader/run-ab.sh --config baseline --reader on  --label baseline-on
#   bench/ple-reader/run-ab.sh --config m33      --reader on  --label m33-on
#
# Env: BONGO_PORT (default 8090), BONGO_DEVICE (default Vulkan1),
#      BONGO_SWEEP_CONTEXTS (default 4096,131072), BONGO_CTX (default 131072),
#      BONGO_HOME, BONGO_PLE_BIN (default $BONGO_HOME/engine/llama.cpp-ple/build-vulkan/bin/llama-server)
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
# shellcheck source=bench/gpu-lock.sh
. "$repo/bench/gpu-lock.sh"

tier="${BONGO_SWEEP_TIER:-iq2_xs}"
budget_gib="${BONGO_BUDGET_GIB:-22.40}"
contexts="${BONGO_SWEEP_CONTEXTS:-4096,131072}"
bongo_home="${BONGO_HOME:-$HOME/.bongo}"
llama_bin="${BONGO_PLE_BIN:-$bongo_home/engine/llama.cpp-ple/build-vulkan/bin/llama-server}"
gguf_dir="$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier"
ctx="${BONGO_CTX:-131072}"
host="127.0.0.1"
port="${BONGO_PORT:-8090}"
device="${BONGO_DEVICE:-Vulkan1}"
out_root="${BONGO_PLE_OUT:-bench/results/2026-09-28-ple-reader-engine}"
expert_bytes="bench/results/2026-09-27-expert-placement/expert-bytes-$tier.json"
analysis="bench/results/2026-09-28-expert-activation/analysis.json"

config=""
reader=""
label=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?}"; shift 2;;
    --reader) reader="${2:?}"; shift 2;;
    --label)  label="${2:?}";  shift 2;;
    *) echo "unknown option '$1'" >&2; exit 2;;
  esac
done

[[ "$config" == "baseline" || "$config" == "m33" ]] || { echo "--config must be baseline or m33" >&2; exit 2; }
[[ "$reader" == "off" || "$reader" == "on" || "$reader" == "auto" ]] || { echo "--reader must be off|on|auto" >&2; exit 2; }
[[ -n "$label" ]] || label="$config-$reader"

log() { printf '[ple-ab] %s\n' "$*" >&2; }
base_url="http://$host:$port/v1"
pidfile="$bongo_home/run/llama-server-ple.pid"
server_log="$bongo_home/run/llama-server-ple.log"

healthy() {
  [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null)" == "200" ]]
}
if healthy; then
  log "ERROR: port $port already serving; set BONGO_PORT or stop the other run."
  exit 3
fi
[[ -x "$llama_bin" ]] || { log "ERROR: no llama-server at $llama_bin"; exit 2; }

shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
(( ${#shards[@]} )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
model="${shards[0]}"

override_tensor=""
n_cpu_moe=0
placement_json=""
if [[ "$config" == "baseline" ]]; then
  n_cpu_moe=16
else
  placement_json="$out_root/placement-$tier-$budget_gib.json"
  python3 bench/gen-ot-placement.py --expert-bytes "$expert_bytes" --analysis "$analysis" \
    --budget-gib "$budget_gib" --tier "$tier" --out "$placement_json" || exit 2
  override_tensor="$(python3 bench/gen-ot-placement.py --expert-bytes "$expert_bytes" \
    --budget-gib "$budget_gib" --print-arg)"
fi

out_dir="$out_root/$label"
mkdir -p "$out_dir"
[[ -n "$placement_json" ]] && cp "$placement_json" "$out_dir/placement.json"

flags=(--model "$model" --ctx-size "$ctx" --jinja --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0
  --n-gpu-layers 99 --n-cpu-moe "$n_cpu_moe")
[[ -n "$override_tensor" ]] && flags+=(--override-tensor "$override_tensor")
flags+=(--host "$host" --port "$port" --parallel 1 --alias "bongo-$tier" --metrics --device "$device"
  --ple-reader "$reader")

python3 - "$out_dir/server-flags.json" "${flags[@]}" <<'PY'
import json, sys
json.dump({"argv": sys.argv[2:]}, open(sys.argv[1], "w"), indent=2)
PY

log "starting llama-server ($label, reader=$reader); log: $server_log"
# Serialise on the single GPU for the whole measured run (BAS-80).
bongo_gpu_lock_acquire "ple-reader/run-ab $label" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
setsid env LD_LIBRARY_PATH="$(dirname "$llama_bin")" "$llama_bin" "${flags[@]}" > "$server_log" 2>&1 &
echo $! > "$pidfile"
pid="$(cat "$pidfile")"

waited=0; started=0
while (( waited < 600 )); do
  healthy && { started=1; break; }
  kill -0 "$pid" 2>/dev/null || break
  sleep 5; waited=$(( waited + 5 ))
done
if (( ! started )); then
  cp "$server_log" "$out_dir/server-start.log" 2>/dev/null || true
  log "server failed to become healthy"; tail -5 "$server_log" >&2
  kill "$pid" 2>/dev/null || true
  exit 3
fi
log "healthy after ${waited}s; pid=$pid"

log "page cache warmup (discarded 4K prefill), matching the BAS-76 sweep"
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

log "harness at contexts=$contexts (reader=$reader)"
python3 bench/harness.py --repo-root "$repo" --tier "$tier" --contexts "$contexts" \
  --repeats 1 --max-tokens 128 --needle-context 131072 --hash-mode none \
  --server-pid "$pid" --out-dir "$out_dir" >> "$out_dir/harness.log" 2>&1
rc=$?
log "harness exit $rc"

python3 - "$out_dir/reader-process.json" "$pid" "$reader" <<'PY' || true
import glob, json, sys
pid, reader = sys.argv[2], sys.argv[3]
o_direct = []
for fd in glob.glob(f"/proc/{pid}/fd/*"):
    try:
        target = json.loads(json.dumps(__import__("os").readlink(fd)))
    except OSError:
        continue
    flags = ""
    try:
        for line in open(f"/proc/{pid}/fdinfo/{__import__('os').path.basename(fd)}"):
            if line.startswith("flags:"):
                flags = line.split(":", 1)[1].strip()
    except OSError:
        pass
    if flags and int(flags, 8) & 0x4000:  # O_DIRECT
        o_direct.append({"fd": __import__("os").path.basename(fd), "target": target, "flags_octal": flags})
json.dump({"reader": reader, "o_direct_fds": o_direct}, open(sys.argv[1], "w"), indent=2)
print("o_direct_fds", o_direct)
PY

kill "$pid" 2>/dev/null || true
for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
kill -9 "$pid" 2>/dev/null || true
rm -f "$pidfile"
log "done ($label); results in $out_dir"
exit $rc
