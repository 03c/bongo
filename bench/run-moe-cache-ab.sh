#!/usr/bin/env bash
# BAS-139 / BAS-76 Step 2 A/B: the MoE expert-cache engine vs the pinned Stage 0
# `--n-cpu-moe 16` baseline, same session, same box, same harness.
#
#   baseline  the pinned Stage 0 Vulkan binary, --n-cpu-moe 16
#   lru       the BAS-139 engine, --n-cpu-moe 48 (all experts host-pinned) with
#             --moe-expert-cache-profile seeded from the R4 offline profile, then
#             online LRU; counters dumped to moe-cache-stats.json
#
# Both runs go through bench/sweep-byte-budget-placement.sh so the protocol
# (4K + 128K prefill/decode, VRAM, needle, prefix-cache path) is identical, and
# serialise on the shared single-GPU flock (BAS-80).
#
# Usage:
#   bench/run-moe-cache-ab.sh                 # build the profile if missing, run both
#   bench/run-moe-cache-ab.sh --dry-run       # print the two plans only
#   bench/run-moe-cache-ab.sh --only lru      # one config
#
# Environment:
#   BONGO_LRU_BIN   the BAS-139 llama-server (default: the build-llama-vulkan-lru.sh output)
#   BONGO_STAGE0_BIN  the pinned Stage 0 llama-server
#   BONGO_MOE_CACHE_PROFILE  profile file (default: bench/results/2026-09-29-moe-cache-lru/profile-iq2_xs-22.40.txt)
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

tier="${BONGO_SWEEP_TIER:-iq2_xs}"
budget_gib="${BONGO_BUDGET_GIB:-22.40}"
bongo_home="${BONGO_HOME:-$HOME/.bongo}"
stage0_bin="${BONGO_STAGE0_BIN:-$bongo_home/llama/b11223/vulkan/llama-server}"
lru_bin="${BONGO_LRU_BIN:-$bongo_home/engine/llama.cpp-lru/build-lru-vulkan/bin/llama-server}"
out_root="${BONGO_SWEEP_OUT:-bench/results/2026-09-29-moe-cache-lru}"
profile="${BONGO_MOE_CACHE_PROFILE:-$out_root/profile-$tier-22.40.txt}"
expert_bytes="bench/results/2026-09-27-expert-placement/expert-bytes-$tier.json"
only="both"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) only="${2:?--only needs baseline|lru|both}"; shift 2;;
    --dry-run) dry_run=1; shift;;
    --profile) profile="${2:?--profile needs a value}"; shift 2;;
    *) echo "unknown option '$1'" >&2; exit 2;;
  esac
done

log() { printf '[moe-cache-ab] %s\n' "$*" >&2; }

# keep both configs under this issue's result directory (the sweep script reads it)
export BONGO_SWEEP_OUT="$out_root"

if [[ ! -f "$profile" ]]; then
  log "generating the R4 profile at $profile"
  python3 bench/gen-moe-cache-profile.py \
    --raw bench/results/2026-09-28-expert-activation/raw \
    --expert-bytes "$expert_bytes" \
    --budget-gib "$budget_gib" --corpora doc,code,chat,convo --tier "$tier" \
    --out "$profile" --json-out "${profile%.txt}.json" || exit 2
fi

run_baseline() {
  [[ -x "$stage0_bin" ]] || { log "ERROR: Stage 0 binary not found at $stage0_bin"; return 2; }
  local args=(--n-cpu-moe 16 --label-suffix "-stage0")
  (( dry_run )) && args+=(--dry-run)
  log "baseline: pinned Stage 0, --n-cpu-moe 16 ($stage0_bin)"
  BONGO_LLAMA_BIN="$stage0_bin" bench/sweep-byte-budget-placement.sh "${args[@]}"
}

run_lru() {
  [[ -x "$lru_bin" ]] || { log "ERROR: BAS-139 engine not found at $lru_bin (build it with tools/build-llama-vulkan-lru.sh)"; return 2; }
  local args=(--n-cpu-moe 48
              --moe-expert-cache-profile "$profile"
              --moe-expert-cache-inserts "${BONGO_MOE_CACHE_INSERTS:-4}"
              --moe-expert-cache-stats "$out_root/moe-cache-stats.json"
              --label-suffix "-moe-lru")
  (( dry_run )) && args+=(--dry-run)
  log "lru: BAS-139 engine, --n-cpu-moe 48, profile=$profile"
  BONGO_LLAMA_BIN="$lru_bin" bench/sweep-byte-budget-placement.sh "${args[@]}"
}

rc=0
case "$only" in
  baseline) run_baseline || rc=$?;;
  lru)      run_lru || rc=$?;;
  both)     run_baseline || rc=$?; run_lru || rc=$?;;
  *) log "ERROR: --only must be baseline|lru|both"; exit 2;;
esac

log "done (rc=$rc); results in $out_root"
exit $rc
