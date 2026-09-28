#!/usr/bin/env bash
# Drive the M3.2b n-gram self-speculation A/B end to end (BAS-131).
#
# Runs the spec-off and spec-on legs at 4K and 128K for each workload class,
# then writes the per-workload compare JSON.  One GPU: run-speculation-ab.sh
# takes the shared flock (BAS-80) for each leg, so the two legs never overlap.
#
#   ./bench/run-speculation-m32b.sh
#
# Env overrides: BONGO_SPEC_WORKLOADS, BONGO_SPEC_CONTEXTS, BONGO_SPEC_REPEATS,
# BONGO_SPEC_OUT (inherited by run-speculation-ab.sh).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "${here}")"
out="${BONGO_SPEC_OUT:-$repo/bench/results/2026-09-28-speculation}"
# The measurement is detached and queued behind the GPU lock, so its server
# logs must survive the heartbeat that launched it.
export BONGO_SPEC_SCRATCH="${BONGO_SPEC_SCRATCH:-$HOME/.bongo/m32b/scratch}"
mkdir -p "$BONGO_SPEC_SCRATCH"
workloads="${BONGO_SPEC_WORKLOADS:-generic,docs}"
contexts="${BONGO_SPEC_CONTEXTS:-4096,131072}"
repeats="${BONGO_SPEC_REPEATS:-2}"
spec_type="${BONGO_SPEC_TYPE:-ngram-map-k4v}"

mkdir -p "$out"
rm -f "$out/.m32b-complete"

echo "== baseline leg =="
BONGO_SPEC_WORKLOADS="$workloads" BONGO_SPEC_CONTEXTS="$contexts" \
  BONGO_SPEC_REPEATS="$repeats" \
  "$here/run-speculation-ab.sh" baseline

echo "== spec leg ($spec_type) =="
BONGO_SPEC_WORKLOADS="$workloads" BONGO_SPEC_CONTEXTS="$contexts" \
  BONGO_SPEC_REPEATS="$repeats" \
  "$here/run-speculation-ab.sh" spec --spec-type "$spec_type"

IFS=',' read -ra wl_list <<< "$workloads"
for wl in "${wl_list[@]}"; do
  wl="${wl// /}"
  [ -n "$wl" ] || continue
  echo "== compare workload=$wl =="
  python3 "$here/measure-speculation.py" compare \
    --baseline "$out/baseline-$wl.json" \
    --spec "$out/spec-$wl.json" \
    --out "$out/compare-$wl.json" || true
done

touch "$out/.m32b-complete"
echo "m32b complete"
