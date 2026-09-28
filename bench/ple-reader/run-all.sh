#!/usr/bin/env bash
# M3.4b PLE reader engine A/B matrix (BAS-79).
#
# Runs the four configurations sequentially:
#
#   baseline off | baseline on | m33 off | m33 on
#
# Each one calls `bench/ple-reader/run-ab.sh`, which queues on the shared
# single-GPU flock (BAS-80) for the whole measured run.  Results land under
# `bench/results/2026-09-28-ple-reader-engine/<config>-<reader>/`.
#
# Usage:
#   bench/ple-reader/run-all.sh            # all four
#   bench/ple-reader/run-all.sh baseline   # only the baseline pair
#   bench/ple-reader/run-all.sh m33        # only the M3.3 byte-budget pair
#
# Env: BONGO_SWEEP_CONTEXTS (default 4096,131072), BONGO_PORT, BONGO_DEVICE,
#      BONGO_GPU_LOCK_TIMEOUT (default 0 = fail fast if the GPU is busy).
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

which="${1:-all}"
case "$which" in
  all)      configs=(baseline m33);;
  baseline) configs=(baseline);;
  m33)      configs=(m33);;
  *) echo "usage: $0 [all|baseline|m33]" >&2; exit 2;;
esac

rc=0
for config in "${configs[@]}"; do
  for reader in off on; do
    label="$config-$reader"
    echo "=== $label $(date -u +%H:%M:%S) ==="
    bench/ple-reader/run-ab.sh --config "$config" --reader "$reader" --label "$label" || rc=$?
  done
done

echo "=== summary ==="
python3 bench/ple-reader/summarize-ab.py || true
exit "$rc"
