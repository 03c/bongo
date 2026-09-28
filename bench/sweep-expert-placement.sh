#!/usr/bin/env bash
# Sweep llama.cpp's static expert-placement control on the Arc Pro B70.
#
# Restarts the bongo server with each `--n-cpu-moe` value, runs the benchmark
# harness at 4096 and 131072 context for the selected tier, and writes raw
# results per configuration under:
#
#   bench/results/<date>-expert-placement/ncmoe-<N>/
#
# The script is resumable: a configuration that already has a matrix.json is
# skipped, and a configuration whose server fails to load (VRAM OOM) still gets
# a matrix.json that records the failure.
#
# Usage:
#   ./bench/sweep-expert-placement.sh [comma-separated n-cpu-moe values]
#
# Default sweep: 24,0,12,48 (the configurations expected to fit are measured
# before the CPU-heavy tail, so a partial run still yields >=4 configurations
# once the baseline `--n-cpu-moe 16` result is included).
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

tier="${BONGO_SWEEP_TIER:-iq2_xs}"
configs="${1:-${BONGO_SWEEP_CONFIGS:-24,0,12,48}}"
contexts="${BONGO_SWEEP_CONTEXTS:-4096,131072}"
out_root="${BONGO_SWEEP_OUT:-bench/results/2026-09-27-expert-placement}"

bongo_home="${BONGO_HOME:-$HOME/.bongo}"
pidfile="$bongo_home/run/llama-server.pid"
server_log="$bongo_home/run/llama-server.log"
llama_bin_dir="$bongo_home/llama/b11223/vulkan"
gguf_dir="$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier"
base_url="http://127.0.0.1:8080/v1"

mkdir -p "$out_root"

log() { printf '[sweep] %s\n' "$*" >&2; }

healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

stop_server() {
  local old
  old="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    log "stopping server pid $old"
    kill "$old" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$old" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$old" 2>/dev/null || true
  fi
  pkill -f "$bongo_home/llama/.*/llama-server" 2>/dev/null || true
  rm -f "$pidfile"
  for _ in $(seq 1 20); do healthy || break; sleep 0.5; done
}

# Sample Intel VRAM attributed to the server process via /proc/<pid>/fdinfo.
vram_bytes() {
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || { echo ""; return 0; }
  python3 - "$pid" <<'PY'
import glob, os, sys
pid = sys.argv[1]
total = 0
found = False
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
            total = max(total, kib * 1024)
            found = True
print(int(total) if found else "")
PY
}

# Cumulative bytes the server has read from block devices (SSD) since start.
server_read_bytes() {
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || { echo ""; return 0; }
  awk '/^read_bytes:/{print $2}' "/proc/$pid/io" 2>/dev/null || true
}

# One discarded 4K prefill so the measured 4K run is not the first touch of the
# CPU-resident experts.  Without this, 4K is a cold page-cache number and is not
# comparable to the warm 128K number (the sweep runs 4K first).
warmup_server() {
  python3 - "$tier" <<'PY'
import sys
sys.path.insert(0, "bench")
from bench_lib import Tokenizer, CORPUS, post_json
tier = sys.argv[1]
base = "http://127.0.0.1:8080/v1"
tk = Tokenizer(base, timeout=900)
prompt = tk.size_to(4096, CORPUS, hard_max=4096)
res = post_json(base, "/completions", {"model": f"bongo-{tier}", "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "cache_prompt": False}, 900)
print(f"warmup prefill status={res.status}")
PY
}

write_fit_failure() {
  local n="$1" reason="$2" dir="$out_root/ncmoe-$n"
  python3 - "$dir" "$n" "$tier" "$reason" <<'PY'
import json, sys, datetime
dir_, n, tier, reason = sys.argv[1:5]
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
matrix = {
    "schema": "bongo.expert-placement.v1",
    "generated_at": now,
    "tier": tier,
    "n_cpu_moe": int(n),
    "fit": False,
    "fatal_error": {"stage": "server load", "message": reason},
    "results": [],
    "memory": {},
}
open(f"{dir_}/matrix.json", "w").write(json.dumps(matrix, indent=2))
open(f"{dir_}/matrix.md", "w").write(
    f"# Expert placement — n-cpu-moe={n}\n\n"
    f"- tier: `{tier}`\n- fit: **False**\n\n"
    f"Server failed to load: `{reason}`\n"
)
PY
}

run_config() {
  local n="$1"
  local dir="$out_root/ncmoe-$n"
  mkdir -p "$dir"

  if [[ -f "$dir/matrix.json" ]]; then
    log "n-cpu-moe=$n already measured ($dir/matrix.json); skipping"
    return 0
  fi

  stop_server
  log "starting server: --n-cpu-moe $n"
  setsid ./bongo.sh \
      --tier "$tier" \
      --gguf-dir "$gguf_dir" \
      --llama-bin "$llama_bin_dir" \
      --runtime dir --runtime-dir "$bongo_home/runtime-empty" \
      --backend vulkan --n-cpu-moe "$n" --detach --yes \
      > "$dir/server-start.log" 2>&1 &
  local bs_pid=$!

  local waited=0 started=0
  while (( waited < 300 )); do
    if healthy; then started=1; break; fi
    local cur
    cur="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$cur" ]] && ! kill -0 "$cur" 2>/dev/null; then
      log "server process died during startup (n-cpu-moe=$n)"
      break
    fi
    sleep 3
    waited=$(( waited + 3 ))
  done

  if (( started == 0 )); then
    cp "$server_log" "$dir/server-start.log" 2>/dev/null || true
    local why="server did not become healthy within ${waited}s (see server-start.log; likely VRAM OOM)"
    if [[ -f "$server_log" ]]; then
      why="$(grep -iE 'out of memory|failed to allocate|error|abort|ggml_vk|vk::' "$server_log" | tail -n 3 | tr '\n' ' ' | cut -c1-400)"
      [[ -n "$why" ]] || why="server did not become healthy within ${waited}s"
    fi
    write_fit_failure "$n" "$why"
    kill "$bs_pid" 2>/dev/null || true
    # leave nothing running
    stop_server
    log "recorded fit=false for n-cpu-moe=$n: $why"
    return 0
  fi

  log "server healthy after ${waited}s (n-cpu-moe=$n); sampling VRAM"
  local vram
  vram="$(vram_bytes || true)"
  printf '{"n_cpu_moe": %s, "vram_resident_bytes_after_load": %s}\n' "$n" "${vram:-null}" \
      > "$dir/placement-after-load.json"
  log "VRAM after load: ${vram:-unknown} bytes"

  log "warming page cache with a discarded 4K prefill (n-cpu-moe=$n)"
  warmup_server >> "$dir/harness.log" 2>&1 || log "warmup prefill failed; continuing"
  local io_before
  io_before="$(server_read_bytes || true)"

  log "running harness at contexts=$contexts (n-cpu-moe=$n)"
  python3 bench/harness.py \
      --repo-root "$repo" \
      --tier "$tier" \
      --contexts "$contexts" \
      --repeats 1 \
      --max-tokens 128 \
      --needle-context 131072 \
      --hash-mode none \
      --out-dir "$dir" \
      > "$dir/harness.log" 2>&1
  local rc=$?
  local io_after
  io_after="$(server_read_bytes || true)"
  printf '{"n_cpu_moe": %s, "server_read_bytes_before": %s, "server_read_bytes_after": %s}\n' \
      "$n" "${io_before:-null}" "${io_after:-null}" > "$dir/server-io.json"
  log "harness exit $rc for n-cpu-moe=$n (log: $dir/harness.log)"
  return 0
}

log "sweep tier=$tier contexts=$contexts configs=$configs -> $out_root"
IFS=',' read -r -a vals <<< "$configs"
for n in "${vals[@]}"; do
  run_config "$n"
done
stop_server
log "sweep complete"
