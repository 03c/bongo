#!/usr/bin/env bash
# M4.1 (BAS-144) host/CPU critical-path measurement session.
#
# One command, one GPU-lock hold, resumable.  It runs, in order:
#
#   1. the host/CPU split probe (bench/profile-host-split.py) for the shipped
#      baseline at 16K -- this is the cost *decomposition*;
#   2. a 16K lever screen: the shipped baseline plus each candidate flag
#      (`--load-mode none`, `--no-op-offload`, `--threads 16`) on the same
#      workload, via bench/run-warm-prefix-profile.sh;
#   3. a short-context needle check for the baseline and the screen winner
#      (bench/run-needle-check.sh);
#   4. the 128K leg for the selected winner (winner first, then the baseline),
#      so the same decomposition and `prompt_ms` exist at long context.
#
# Every step is skippable: a config whose output JSON already exists is skipped
# by the profile / needle runners, so the session can be re-run to resume.
#
# Usage:
#   bench/run-m4.1-host-cpu.sh                    # all stages
#   BONGO_M4_WINNER=no_op_offload bench/run-m4.1-host-cpu.sh
#   BONGO_M4_STAGE=screen bench/run-m4.1-host-cpu.sh
#   BONGO_M4_DRY_RUN=1 bench/run-m4.1-host-cpu.sh
#
# Environment:
#   BONGO_M4_OUT     results root (default bench/results/2026-09-29-host-cpu)
#   BONGO_M4_WINNER  pin the lever promoted to the 128K leg (default: auto-pick)
#   BONGO_M4_STAGE   all | split | screen | needle | longctx (default all)
#   BONGO_M4_LEVERS  space list for the 16K screen
#   BONGO_M4_LOAD_TIMEOUT  per-config server-load wait, seconds (default 300)
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
# shellcheck source=bench/gpu-lock.sh
. "$repo/bench/gpu-lock.sh"

out="${BONGO_M4_OUT:-bench/results/2026-09-29-host-cpu}"
# Context-scoped roots: the profile runner skips a config whose raw file already
# exists, and the 16K and 128K legs share config names (baseline, the winner).
out16="$out/ctx16k"
out128="$out/ctx128k"
stage="${BONGO_M4_STAGE:-all}"
winner="${BONGO_M4_WINNER:-}"
levers="${BONGO_M4_LEVERS:-baseline lm_none no_op_offload threads16}"
load_timeout="${BONGO_M4_LOAD_TIMEOUT:-300}"
dry_run="${BONGO_M4_DRY_RUN:-0}"

log() { printf '[m4.1 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

run_split() {
  local configs="$1" prefixes="$2" deltas="$3" root="$4"
  log "host-split: configs='$configs' prefixes=$prefixes deltas=$deltas root=$root"
  BONGO_HOST_SPLIT=1 BONGO_PROFILE_OUT="$root" \
  BONGO_PROFILE_LOAD_TIMEOUT="$load_timeout" \
  BONGO_PROFILE_CONFIGS="$configs" \
  BONGO_PROFILE_PREFIXES="$prefixes" BONGO_PROFILE_DELTAS="$deltas" \
    bench/run-warm-prefix-profile.sh
}

run_profile() {
  local configs="$1" prefixes="$2" deltas="$3" root="$4"
  log "warm-profile: configs='$configs' prefixes=$prefixes deltas=$deltas root=$root"
  BONGO_PROFILE_OUT="$root" \
  BONGO_PROFILE_LOAD_TIMEOUT="$load_timeout" \
  BONGO_PROFILE_CONFIGS="$configs" \
  BONGO_PROFILE_PREFIXES="$prefixes" BONGO_PROFILE_DELTAS="$deltas" \
    bench/run-warm-prefix-profile.sh
}

run_needle() {
  local configs="$1"
  log "needle: configs='$configs'"
  BONGO_NEEDLE_OUT="$out/needle" BONGO_NEEDLE_CONFIGS="$configs" \
    bench/run-needle-check.sh
}

# Pick the 16K-screen config with the lowest delta-512 `prompt_ms`; report it as
# "name pct" where pct is the reduction vs `baseline`.  Empty output means no
# usable screen data.
pick_winner() {
  python3 - "$out16" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = {}
for path in glob.glob(os.path.join(root, "*", "profile.json")):
    try:
        d = json.load(open(path))
    except Exception:
        continue
    label = d.get("label") or os.path.basename(os.path.dirname(path))
    for p in d.get("points", []):
        for r in p.get("runs", []):
            if r.get("label") == f"grow_p{p['prefix_tokens']}_d512" and r.get("prompt_ms"):
                rows[label] = r["prompt_ms"]
base = rows.get("baseline")
if not base:
    sys.exit(0)
best = min(((v, k) for k, v in rows.items() if k != "baseline"), default=None)
if not best:
    sys.exit(0)
ms, name = best
print(f"{name} {100.0 * (base - ms) / base:.2f}")
PY
}

resolve_winner() {
  if [[ -n "$winner" ]]; then
    return 0
  fi
  local picked
  picked="$(pick_winner || true)"
  if [[ -n "$picked" ]]; then
    winner="${picked%% *}"
    log "16K screen winner: $picked% vs baseline; promoting '$winner'"
  else
    winner="baseline"
    log "16K screen produced no usable winner; treating the baseline as the winner"
  fi
}

if [[ "$dry_run" == "1" ]]; then
  log "dry run; results root $out"
  BONGO_HOST_SPLIT=1 BONGO_PROFILE_OUT="$out16" BONGO_PROFILE_DRY_RUN=1 \
    BONGO_PROFILE_CONFIGS="baseline" BONGO_PROFILE_PREFIXES=16384 BONGO_PROFILE_DELTAS=512 \
    bench/run-warm-prefix-profile.sh
  BONGO_PROFILE_OUT="$out16" BONGO_PROFILE_DRY_RUN=1 \
    BONGO_PROFILE_CONFIGS="$levers" BONGO_PROFILE_PREFIXES=16384 BONGO_PROFILE_DELTAS=512 \
    bench/run-warm-prefix-profile.sh
  BONGO_HOST_SPLIT=1 BONGO_PROFILE_OUT="$out128" BONGO_PROFILE_DRY_RUN=1 \
    BONGO_PROFILE_CONFIGS="no_op_offload_128k baseline" BONGO_PROFILE_PREFIXES=128000 BONGO_PROFILE_DELTAS=512 \
    bench/run-warm-prefix-profile.sh
  exit 0
fi

bongo_gpu_lock_acquire "run-m4.1-host-cpu ($stage)" || exit 3
trap 'bongo_gpu_lock_release' EXIT INT TERM
# Children (the profile runner and the llama-server it starts) must inherit the
# lock, not take it a second time.
export BONGO_GPU_LOCK_HELD=1

rc=0
if [[ "$stage" == "all" || "$stage" == "split" ]]; then
  run_split "baseline" "16384" "512" "$out16" || rc=$?
fi
if [[ "$stage" == "all" || "$stage" == "screen" ]]; then
  run_profile "$levers" "16384" "512" "$out16" || rc=$?
fi
if [[ "$stage" == "all" || "$stage" == "needle" ]]; then
  resolve_winner
  if [[ "$winner" == "baseline" ]]; then
    run_needle "baseline" || rc=$?
  else
    run_needle "baseline $winner" || rc=$?
  fi
fi
if [[ "$stage" == "all" || "$stage" == "longctx" ]]; then
  resolve_winner
  case "$winner" in
    baseline)
      log "winner is the baseline; running only the 128K host-split baseline"
      run_split "baseline" "128000" "512" "$out128" || rc=$?
      ;;
    *)
      # Winner first: the 128K leg is long (~17 min of cold prime per config),
      # so if the session is cut short the measured lever still lands.
      run_split "${winner}_128k baseline" "128000" "512" "$out128" || rc=$?
      ;;
  esac
fi
if [[ "$stage" != "all" && "$stage" != "split" && "$stage" != "screen" && "$stage" != "needle" && "$stage" != "longctx" ]]; then
  log "unknown BONGO_M4_STAGE='$stage' (all|split|screen|needle|longctx)"
  rc=2
fi

log "done (rc=$rc); results in $out"
exit $rc
