#!/usr/bin/env bash
# Measure llama-server throughput under a reversible cgroup-v2 RAM cap.
#
# One model tier (IQ2_XS), one static expert split, two context lengths.  The
# only thing that changes between runs is `memory.max`: the server is started
# inside a transient `systemd-run --user --scope` unit with `MemoryMax=<N>G`
# and `MemorySwapMax=0`, so the constraint is enforced by the kernel and
# disappears when the scope exits.  No kernel command line change, no reboot,
# no runtime-code change.
#
#   ./bench/run-capacity-sensitivity.sh 16 16     # cap 16 GiB, n-cpu-moe 16
#   ./bench/run-capacity-sensitivity.sh 24 16
#   ./bench/run-capacity-sensitivity.sh 16 24     # CPU expert set > cap -> binds
#
# Results land in:
#   bench/results/<date>-capacity-sensitivity/mem<N>g-ncmoe<M>/
#     matrix.json            from bench/harness.py (throughput + VRAM/RSS)
#     matrix.md
#     samples.jsonl          per-tick /proc + cgroup samples
#     memory.json            peak/aggregate of samples.jsonl
#     server-io.json         cumulative SSD bytes at load/warmup/end
#     cgroup.json            the memory.max actually in force
#     server-start.log, harness.log
#
# Resumable: a run that already has matrix.json is skipped.
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

mem_gib="${1:?usage: run-capacity-sensitivity.sh <memory-max-GiB|none> [n-cpu-moe]}"
n_cpu_moe="${2:-16}"
tier="${BONGO_CAPACITY_TIER:-iq2_xs}"
contexts="${BONGO_CAPACITY_CONTEXTS:-4096,131072}"
out_root="${BONGO_CAPACITY_OUT:-bench/results/2026-09-28-capacity-sensitivity}"
if [[ "$mem_gib" == "none" ]]; then
  run_dir="$out_root/uncapped-ncmoe${n_cpu_moe}"
  unit="bongo-cap-uncapped-${n_cpu_moe}"
else
  run_dir="$out_root/mem${mem_gib}g-ncmoe${n_cpu_moe}"
  unit="bongo-cap-${mem_gib}g-${n_cpu_moe}"
fi

bongo_home="${BONGO_HOME:-$HOME/.bongo}"
pidfile="$bongo_home/run/llama-server.pid"
server_log="$bongo_home/run/llama-server.log"
llama_bin_dir="$bongo_home/llama/b11223/vulkan"
gguf_dir="$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier"
runtime_dir="$bongo_home/runtime-empty"
base_url="http://127.0.0.1:8080/v1"
unit="bongo-cap-${mem_gib}g-${n_cpu_moe}"

log() { printf '[capacity] %s\n' "$*" >&2; }

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
  systemctl --user stop "$unit.scope" 2>/dev/null || true
  rm -f "$pidfile"
  for _ in $(seq 1 20); do healthy || break; sleep 0.5; done
}

pid_io() {
  local pid="$1" key="$2"
  awk -v k="$key" '$1 == k":" {print $2}' "/proc/$pid/io" 2>/dev/null || true
}

# /proc/<pid>/stat majflt (field 12), robust to a comm containing spaces.
pid_majflt() {
  local pid="$1"
  python3 - "$pid" <<'PY' 2>/dev/null || true
import sys
pid = sys.argv[1]
try:
    data = open(f"/proc/{pid}/stat").read()
except OSError:
    sys.exit(0)
tail = data[data.rfind(")") + 1:].split()
# after comm: state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt
print(int(tail[9]))
PY
}

# Aggregate samples.jsonl into a compact memory.json.
aggregate_samples() {
  python3 - "$run_dir/samples.jsonl" "$run_dir/memory.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
recs = []
try:
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
except OSError:
    recs = []
live = [r for r in recs if r.get("pid")]
def peak(path):
    vals = []
    for r in live:
        node = r
        for k in path:
            node = (node or {}).get(k)
        if isinstance(node, (int, float)):
            vals.append(node)
    return max(vals) if vals else None
def last(path):
    for r in reversed(live):
        node = r
        for k in path:
            node = (node or {}).get(k)
        if isinstance(node, (int, float)):
            return node
    return None
def last_str(path):
    for r in reversed(live):
        node = r
        for k in path:
            node = (node or {}).get(k)
        if node is not None:
            return node
    return None
def ev(key):
    vals = []
    for r in live:
        v = ((r.get("cgroup") or {}).get("memory.events") or {}).get(key)
        if isinstance(v, int):
            vals.append(v)
    return max(vals) if vals else None
out = {
    "samples": len(recs),
    "samples_with_pid": len(live),
    "cgroup_path": last_str(["cgroup", "path"]),
    "memory_max": last_str(["cgroup", "memory.max"]),
    "memory_current_peak": peak(["cgroup", "memory.current"]) if False else None,
    "vram_peak_bytes": peak(["vram", "drm-resident-vram0"]),
    "vmrss_peak_kb": peak(["status", "VmRSS_kb"]),
    "vmhwm_kb": peak(["status", "VmHWM_kb"]),
    "majflt_last": last(["stat", "majflt"]),
    "minflt_last": last(["stat", "minflt"]),
    "read_bytes_last": last(["io", "read_bytes"]),
    "read_chars_last": last(["io", "rchar"]),
    "pgmajfault_last": ((live[-1].get("cgroup") or {}).get("memory.stat") or {}).get("pgmajfault") if live else None,
    "memory_events": {k: ev(k) for k in ("max", "oom", "oom_kill", "high", "low")},
}
# memory.current is a string in cgroup v2; peak it numerically.
cur = []
for r in live:
    v = ((r.get("cgroup") or {}).get("memory.current"))
    try:
        cur.append(int(v))
    except (TypeError, ValueError):
        pass
out["memory_current_peak"] = max(cur) if cur else None
open(dst, "w").write(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
PY
}

run_one() {
  mkdir -p "$run_dir"

  if [[ -f "$run_dir/matrix.json" ]]; then
    log "mem=${mem_gib} n-cpu-moe=${n_cpu_moe} already measured; skipping"
    return 0
  fi

  stop_server
  if [[ "$mem_gib" == "none" ]]; then
    log "starting server UNCONSTRAINED (n-cpu-moe=$n_cpu_moe) as the same-protocol control"
    ./bongo.sh \
        --tier "$tier" \
        --gguf-dir "$gguf_dir" \
        --llama-bin "$llama_bin_dir" \
        --runtime dir --runtime-dir "$runtime_dir" \
        --backend vulkan --n-cpu-moe "$n_cpu_moe" --yes \
      > "$run_dir/server-start.log" 2>&1 &
  else
    log "starting server under MemoryMax=${mem_gib}G SwapMax=0 (n-cpu-moe=$n_cpu_moe)"
    systemd-run --user --scope -p "MemoryMax=${mem_gib}G" -p MemorySwapMax=0 \
      --unit "$unit" -- \
      ./bongo.sh \
        --tier "$tier" \
        --gguf-dir "$gguf_dir" \
        --llama-bin "$llama_bin_dir" \
        --runtime dir --runtime-dir "$runtime_dir" \
        --backend vulkan --n-cpu-moe "$n_cpu_moe" --yes \
      > "$run_dir/server-start.log" 2>&1 &
  fi
  local scope_pid=$!

  local waited=0 started=0
  while (( waited < 900 )); do
    if healthy; then started=1; break; fi
    local cur
    cur="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$cur" ]] && ! kill -0 "$cur" 2>/dev/null; then
      log "server process died during startup"
      break
    fi
    sleep 3
    waited=$(( waited + 3 ))
    if (( waited % 30 == 0 )); then log "  ...waiting for load (${waited}s)"; fi
  done

  if (( started == 0 )); then
    cp "$server_log" "$run_dir/server-start.log" 2>/dev/null || true
    log "server failed to become healthy under mem=${mem_gib}G; recording negative result"
    python3 - "$run_dir" "$mem_gib" "$n_cpu_moe" "$tier" <<'PY'
import json, sys, datetime, pathlib
run_dir, mem, n, tier = sys.argv[1:5]
pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
matrix = {
    "schema": "bongo.capacity-sensitivity.v1",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tier": tier, "memory_max_gib": (None if mem == "none" else int(mem)), "n_cpu_moe": int(n),
    "fit": False, "fatal_error": "server did not become healthy under the memory cap",
    "results": [],
}
open(f"{run_dir}/matrix.json", "w").write(json.dumps(matrix, indent=2))
open(f"{run_dir}/matrix.md", "w").write(
    f"# Capacity sensitivity — memory.max={mem}G, n-cpu-moe={n}\n\n"
    "Server did not load/serve under the cap.  See `server-start.log`.\n")
PY
    systemctl --user stop "$unit.scope" 2>/dev/null || true
    stop_server
    return 0
  fi

  local pid
  pid="$(cat "$pidfile")"
  log "server healthy after ${waited}s (pid $pid)"
  local cg
  cg="$(python3 -c "
import sys
for line in open('/proc/$pid/cgroup'):
    p=line.split(':',2)
    if len(p)==3 and p[0]=='0':
        rel=p[2].strip().lstrip('/')
        print(('/sys/fs/cgroup/'+rel) if rel else '/sys/fs/cgroup')
" 2>/dev/null)"
  python3 - "$pid" "$cg" "${mem_gib}G" "$run_dir/cgroup.json" <<'PY'
import json, sys, os
pid, cg, want, dst = sys.argv[1:5]
def r(p):
    try: return open(p).read().strip()
    except OSError: return None
out = {
    "pid": int(pid), "cgroup_path": cg, "requested_memory_max": want,
    "memory.max": r(os.path.join(cg, "memory.max")) if cg else None,
    "memory.current_after_load": r(os.path.join(cg, "memory.current")) if cg else None,
    "memory.swap.max": r(os.path.join(cg, "memory.swap.max")) if cg else None,
    "memory.events_after_load": r(os.path.join(cg, "memory.events")) if cg else None,
}
json.dump(out, open(dst, "w"), indent=2)
print(json.dumps(out))
PY

  # Start the sampler before the warm-up so load-adjacent I/O is captured.
  python3 bench/capacity_sampler.py --pid-file "$pidfile" \
    --out "$run_dir/samples.jsonl" --interval 0.5 \
    > "$run_dir/sampler.log" 2>&1 &
  local sampler_pid=$!

  # Discarded warm-up prefill so the measured 4K run is not a cold-cache
  # number (same protocol as the expert-placement sweep).
  log "warming page cache with a discarded 4K prefill"
  python3 - "$tier" <<'PY' >> "$run_dir/harness.log" 2>&1
import sys
sys.path.insert(0, "bench")
from bench_lib import Tokenizer, CORPUS, post_json
tier = sys.argv[1]
base = "http://127.0.0.1:8080/v1"
tk = Tokenizer(base, timeout=900)
prompt = tk.size_to(4096, CORPUS, hard_max=4096)
res = post_json(base, "/completions", {"model": f"bongo-{tier}", "prompt": prompt,
    "max_tokens": 1, "temperature": 0.0, "cache_prompt": False}, 900)
print(f"warmup prefill status={res.status}")
PY

  local io_rchar_before io_read_before majflt_before
  io_rchar_before="$(pid_io "$pid" rchar)"
  io_read_before="$(pid_io "$pid" read_bytes)"
  majflt_before="$(pid_majflt "$pid")"

  log "running harness contexts=$contexts (mem=${mem_gib}G n-cpu-moe=$n_cpu_moe)"
  python3 bench/harness.py \
      --repo-root "$repo" \
      --tier "$tier" \
      --contexts "$contexts" \
      --repeats 1 \
      --max-tokens 128 \
      --needle-context 131072 \
      --hash-mode none \
      --out-dir "$run_dir" \
      >> "$run_dir/harness.log" 2>&1
  local rc=$?

  local io_rchar_after io_read_after majflt_after
  io_rchar_after="$(pid_io "$pid" rchar)"
  io_read_after="$(pid_io "$pid" read_bytes)"
  majflt_after="$(pid_majflt "$pid")"
  printf '{"mem_gib": %s, "n_cpu_moe": %s, "rchar_before": %s, "rchar_after": %s, "read_bytes_before": %s, "read_bytes_after": %s, "majflt_before": %s, "majflt_after": %s, "harness_exit": %s}\n' \
    "$mem_gib" "$n_cpu_moe" "${io_rchar_before:-null}" "${io_rchar_after:-null}" \
    "${io_read_before:-null}" "${io_read_after:-null}" \
    "${majflt_before:-null}" "${majflt_after:-null}" "$rc" > "$run_dir/server-io.json"

  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  aggregate_samples >> "$run_dir/harness.log" 2>&1 || true
  log "harness exit $rc for mem=${mem_gib}G n-cpu-moe=$n_cpu_moe"
  stop_server
  wait "$scope_pid" 2>/dev/null || true
}

log "capacity sensitivity: tier=$tier mem=${mem_gib}G n-cpu-moe=$n_cpu_moe contexts=$contexts -> $run_dir"
run_one
log "done"
