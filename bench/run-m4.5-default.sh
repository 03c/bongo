#!/usr/bin/env bash
# M4.5 (BAS-163) — verify the *shipped* `bongo.sh` default: `--placement auto`
# with the automatic load fallback to the tier split.
#
# `bongo.sh` default since M4.5 is `--placement auto` (12 CPU expert layers at
# `--ctx <= 131072`, 16 above) with a one-shot fallback to the tier split if the
# server fails to load.  This runner drives `bongo.sh` itself (not llama-server
# directly) so the config, the plan output and the generated JSON are the ones a
# user gets, then checks four stages under one GPU-lock hold (BAS-80):
#
#   ctx128    ./bongo.sh default at 131072: resolves to --n-cpu-moe 12, serves,
#             and measures the 4K decode (the M4.4 16.96 -> 19.59 number).
#   ctx256    ./bongo.sh default at 262144: resolves to --n-cpu-moe 18 (the
#             measured large-context safe split), serves.
#   fallback  ./bongo.sh default at 131072 against a wrapper engine that exits
#             with an OOM when it sees --n-cpu-moe 12: bongo.sh must retry at 16,
#             record the fallback, and serve.
#   turns     the M4.4 turn-regression + needle harness at the auto split (12).
#
# Resumable: a stage whose marker file already exists is skipped.
#
# Usage:
#   bench/run-m4.5-default.sh                 # all stages
#   BONGO_M45_STAGE=ctx128 bench/run-m4.5-default.sh
#   BONGO_M45_DRY_RUN=1 bench/run-m4.5-default.sh
#
# Environment:
#   BONGO_M45_OUT         results root (default bench/results/2026-09-29-m4.5-auto-default)
#   BONGO_M45_STAGE       all | ctx128 | ctx256 | fallback | turns (default all)
#   BONGO_M45_SCRATCH     scratch home root for the bongo.sh runs (default a mktemp dir)
#   BONGO_LLAMA_SERVER    patched engine binary (default the cached M4.2 build)
#   BONGO_GGUF_DIR        tier dir (default $BONGO_HOME/models/.../iq2_xs)
#   BONGO_DEVICE          Vulkan device (default Vulkan1)
#   BONGO_PROFILE_LOAD_TIMEOUT  seconds to wait for model load (default 900)
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
# shellcheck source=bench/gpu-lock.sh
. "$repo/bench/gpu-lock.sh"

bongo_home="${BONGO_HOME:-$HOME/.bongo}"
tier="${BONGO_SWEEP_TIER:-iq2_xs}"
llama_bin="${BONGO_LLAMA_SERVER:-$bongo_home/engine/llama.cpp-pin/build-vulkan/bin/llama-server}"
engine_dir="$(dirname "$llama_bin")"
gguf_dir="${BONGO_GGUF_DIR:-$bongo_home/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/$tier}"
runtime_dir="${BONGO_RUNTIME_DIR:-$bongo_home/runtime}"
host="${BONGO_HOST:-127.0.0.1}"
port="${BONGO_PORT:-8080}"
out="${BONGO_M45_OUT:-bench/results/2026-09-29-m4.5-auto-default}"
stage="${BONGO_M45_STAGE:-all}"
dry_run="${BONGO_M45_DRY_RUN:-0}"
load_timeout="${BONGO_PROFILE_LOAD_TIMEOUT:-900}"
scratch="${BONGO_M45_SCRATCH:-${PAPERCLIP_RUN_SCRATCH_DIR:-${TMPDIR:-/tmp}}/bongo-m45.$$}"
mkdir -p "$scratch"
base_url="http://$host:$port/v1"

log() { printf '[m4.5 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$base_url/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

wait_healthy() { # <pid> <seconds>
  local pid="$1" limit="$2" waited=0
  while (( waited < limit )); do
    healthy && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 5; waited=$(( waited + 5 ))
    (( waited % 60 == 0 )) && log "  ...still loading (${waited}s)"
  done
  return 1
}

stop_server() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$pid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# The wrapper "engine": a llama-server that exits with an OOM when it sees
# --n-cpu-moe 12, and execs the real engine otherwise.  The sibling lib symlinks
# keep bongo.sh's binary_is_m42_patched() check (which greps libggml-vulkan.so)
# green, so the M4.2 levers stay on.
# ---------------------------------------------------------------------------
make_fake_engine() {
  local dir="$out/fallback-engine/bin"
  mkdir -p "$dir"
  local lib
  for lib in "$engine_dir"/lib*.so*; do
    [[ -e "$lib" ]] || continue
    ln -sfn "$lib" "$dir/$(basename "$lib")"
  done
  cat > "$dir/llama-server" <<EOF
#!/usr/bin/env bash
# M4.5 fallback-test wrapper: fail the auto placement once, serve the rest.
args=("\$@")
for ((i = 0; i < \${#args[@]}; i++)); do
  if [[ "\${args[i]}" == "--n-cpu-moe" && "\${args[i + 1]:-}" == "12" ]]; then
    echo "vk::Device::allocateMemory: ErrorOutOfDeviceMemory (forced by the M4.5 fallback test)" >&2
    exit 1
  fi
done
exec "$llama_bin" "\$@"
EOF
  chmod +x "$dir/llama-server"
  printf '%s' "$dir"
}

# run_bongo <label> <ctx> <llama-bin-dir> [extra bongo args...]
# Starts ./bongo.sh --detach with a scratch BONGO_HOME, waits for health, and
# writes bongo-config.json + the server flags into the stage dir.
run_bongo() {
  local label="$1" ctx="$2" bin_dir="$3"; shift 3
  local dir="$out/$label"
  mkdir -p "$dir"
  local home="$scratch/$label"
  mkdir -p "$home"
  local bh="$home/.bongo"
  log "$label: ./bongo.sh --ctx $ctx (llama-bin $bin_dir, BONGO_HOME $bh)"
  BONGO_HOME="$bh" BONGO_GPU_LOCK_HELD=1 \
    "$repo/bongo.sh" --ctx "$ctx" --tier "$tier" --gguf-dir "$gguf_dir" \
      --llama-bin "$bin_dir" --runtime dir --runtime-dir "$runtime_dir" \
      --host "$host" --port "$port" --detach "$@" \
      > "$dir/bongo.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    log "$label: bongo.sh exited $rc"
    tail -n 12 "$dir/bongo.log" >&2
    return $rc
  fi
  cp "$bh/run/bongo-config.json" "$dir/bongo-config.json" 2>/dev/null || true
  cp "$bh/run/bongo-config.env" "$dir/bongo-config.env" 2>/dev/null || true
  cp "$bh/run/llama-server.log" "$dir/llama-server.log" 2>/dev/null || true
  local pid
  pid="$(cat "$bh/run/llama-server.pid" 2>/dev/null || true)"
  if healthy; then
    log "$label: healthy (pid ${pid:-?})"
    printf '%s' "$pid" > "$dir/server.pid"
    return 0
  fi
  if wait_healthy "${pid:-0}" "$load_timeout"; then
    log "$label: healthy after the wait (pid ${pid:-?})"
    printf '%s' "$pid" > "$dir/server.pid"
    return 0
  fi
  log "$label: server did not become healthy"
  tail -n 12 "$dir/llama-server.log" >&2
  stop_server "$pid"
  return 1
}

stop_bongo() { # <label>
  local dir="$out/$1" pid=""
  pid="$(cat "$dir/server.pid" 2>/dev/null || true)"
  stop_server "$pid"
  rm -f "$dir/server.pid"
}

# The 4K decode workload of the M4.4 lever screen: 4096-token prompt, 128
# generated tokens, 3 repeats, cache_prompt=false, after a discarded warm-up.
measure_decode4k() { # <label> <reps> <out.json>
  local label="$1" reps="$2" outfile="$3"
  python3 - "$tier" "$base_url" "$reps" "$outfile" "$label" <<'PY'
import json, sys
sys.path.insert(0, "bench")
from bench_lib import Tokenizer, CORPUS
from harness import streaming_measure
tier, base, reps, outfile, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
tk = Tokenizer(base, timeout=900)
prompt = tk.size_to(4096, CORPUS, hard_max=4096)
streaming_measure(base, f"bongo-{tier}", prompt, 8, 900, cache_prompt=False)
recs = []
for i in range(reps):
    r = streaming_measure(base, f"bongo-{tier}", prompt, 128, 900, cache_prompt=False)
    recs.append(r)
    print(f"run{i}: output_tokens={r.get('output_tokens')} output_tps={r.get('output_tps')} status={r.get('status')}")
json.dump({"label": label, "ctx_tokens": 4096, "max_tokens": 128, "runs": recs}, open(outfile, "w"), indent=2)
PY
}

# Memory sample + one request, recorded as a ctx-fit JSON.
measure_fit() { # <label>
  local dir="$out/$1" pid
  pid="$(cat "$dir/server.pid" 2>/dev/null || true)"
  python3 - "$tier" "$base_url" "$pid" "$dir/fit.json" "$1" <<'PY'
import json, sys, time
sys.path.insert(0, "bench")
from bench_lib import MemorySampler, post_json
tier, base, pid, outfile, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
sampler = MemorySampler(server_pids=[pid], interval=0.25)
sampler.start(); time.sleep(1.5)
res = post_json(base, "/completions", {"model": f"bongo-{tier}", "prompt": "bongo M4.5 default fit",
                                       "max_tokens": 4, "temperature": 0.0, "cache_prompt": False}, 900)
time.sleep(1.5); sampler.stop()
now = time.time(); peak = sampler.window_peak(0, now)
json.dump({
    "schema": "bongo.m4.5-fit.v1", "label": label,
    "request_status": res.status,
    "vram_peak_gib": round(peak["vram_peak_bytes"] / 2**30, 3) if peak["vram_peak_bytes"] else None,
    "vram_method": peak["vram_method"],
    "rss_peak_gib": round(peak["system_ram_process_hwm_bytes"] / 2**30, 3) if peak["system_ram_process_hwm_bytes"] else None,
    "host_ram_available_gib": round((sampler.host_mem().get("MemAvailable") or 0) / 2**20, 3),
}, open(outfile, "w"), indent=2)
print(f"{label}: status={res.status} vram_peak_gib={json.load(open(outfile))['vram_peak_gib']}")
PY
}

check_default_config() { # <label> <expected n_cpu_moe> <fallback 0|1>
  local dir="$out/$1" want="$2" fallback="$3"
  python3 - "$dir/bongo-config.json" "$want" "$fallback" "$1" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
want, fallback, label = int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
p = cfg.get("placement", {})
ok = (p.get("policy") == "auto"
      and p.get("n_cpu_moe") == want
      and p.get("n_cpu_moe_requested") == (want if not fallback else 12)
      and p.get("auto_fallback") is fallback)
print(f"{label} placement: {json.dumps(p)}")
sys.exit(0 if ok else 1)
PY
}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
run_ctx128() {
  local dir="$out/ctx128"
  [[ -f "$dir/decode4k.json" ]] && { log "ctx128 already measured; skipping"; return 0; }
  mkdir -p "$dir"
  run_bongo ctx128 131072 "$engine_dir" || return 1
  check_default_config ctx128 12 0 || { stop_bongo ctx128; return 1; }
  measure_fit ctx128 || { stop_bongo ctx128; return 1; }
  measure_decode4k ctx128 3 "$dir/decode4k.json" || { stop_bongo ctx128; return 1; }
  stop_bongo ctx128
  log "ctx128 done"
}

run_ctx256() {
  local dir="$out/ctx256"
  [[ -f "$dir/fit.json" ]] && { log "ctx256 already measured; skipping"; return 0; }
  mkdir -p "$dir"
  run_bongo ctx256 262144 "$engine_dir" || return 1
  check_default_config ctx256 18 0 || { stop_bongo ctx256; return 1; }
  measure_fit ctx256 || { stop_bongo ctx256; return 1; }
  stop_bongo ctx256
  log "ctx256 done"
}

run_fallback() {
  local dir="$out/fallback"
  [[ -f "$dir/bongo-config.json" ]] && { log "fallback already measured; skipping"; return 0; }
  mkdir -p "$dir"
  local fake
  fake="$(make_fake_engine)"
  if run_bongo fallback 131072 "$fake"; then :; else
    # bongo.sh exits non-zero on failure, but the fallback path is expected to
    # succeed; only flag a real failure.
    log "fallback: bongo.sh did not reach healthy"
    return 1
  fi
  check_default_config fallback 16 1 || { stop_bongo fallback; return 1; }
  measure_fit fallback || { stop_bongo fallback; return 1; }
  if [[ -f "$dir/server.pid" ]]; then
    local bh="$scratch/fallback/.bongo"
    cp "$bh/run/llama-server-auto-12.log" "$dir/llama-server-auto-12.log" 2>/dev/null || true
    grep -q 'ErrorOutOfDeviceMemory' "$dir/llama-server-auto-12.log" 2>/dev/null \
      || log "fallback: warning: the preserved auto log has no OOM line"
  fi
  stop_bongo fallback
  log "fallback done"
}

run_turns() {
  local dir="$out/turns"
  [[ -f "$dir/nc12/profile-host-split.json" ]] && { log "turns already measured; skipping"; return 0; }
  mkdir -p "$dir"
  # The shipped default emits exactly the M4.4 nc12 flags; the M4.4 turn harness
  # accepts --n-cpu-moe 12 through BONGO_M44_TURN_FLAGS (the last flag wins).
  BONGO_M44_TURN_OUT="$dir" \
  BONGO_M44_TURN_FLAGS="--n-cpu-moe 12" \
  BONGO_LLAMA_SERVER="$llama_bin" \
  BONGO_PROFILE_LOAD_TIMEOUT="$load_timeout" \
    bench/run-m4.4-turn-check.sh nc12
}

if [[ "$dry_run" == "1" ]]; then
  log "dry run; results root $out"
  log "engine: $llama_bin"
  log "scenarios: ctx128 (auto=12 + 4K decode), ctx256 (auto=18), fallback (forced OOM 12->16), turns (nc12)"
  exit 0
fi

if (( ! dry_run )); then
  [[ -d "$runtime_dir" ]] || { log "ERROR: runtime dir $runtime_dir not found"; exit 2; }
  [[ -x "$llama_bin" ]] || { log "ERROR: llama-server not found at $llama_bin"; exit 2; }
  shards=("$gguf_dir"/Swift-Qwen3.8-Flash-Next-GSQ-RCO-*.gguf)
  (( ${#shards[@]} > 0 )) || { log "ERROR: no GGUF shards in $gguf_dir"; exit 2; }
  grep -qa -m1 'GGML_VK_HOST_BUFT_PER_DEVICE' "$engine_dir"/libggml-vulkan.so* 2>/dev/null \
    || { log "ERROR: $llama_bin is not the M4.2-patched engine"; exit 2; }
fi

bongo_gpu_lock_acquire "run-m4.5-default ($stage)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
export BONGO_GPU_LOCK_HELD=1
if healthy; then
  log "ERROR: port $port already serving after acquiring the GPU lock; refusing to contaminate."
  exit 3
fi

rc=0
case "$stage" in
  all)  run_ctx128 || rc=$?; run_ctx256 || rc=$?; run_fallback || rc=$?; run_turns || rc=$?;;
  ctx128) run_ctx128 || rc=$?;;
  ctx256) run_ctx256 || rc=$?;;
  fallback) run_fallback || rc=$?;;
  turns) run_turns || rc=$?;;
  *) log "unknown stage '$stage'"; rc=2;;
esac
log "done (rc=$rc); results in $out (scratch $scratch)"
exit $rc
