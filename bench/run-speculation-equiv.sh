#!/usr/bin/env bash
# Clean-slate greedy equivalence re-test (BAS-131).
#
# The main A/B issued the 128K equivalence request after a 4K decode, so the
# server reused a KV prefix that the spec leg had built with batched verify
# forwards.  That contaminated the comparison.  This re-test runs the greedy
# equivalence with cache_prompt=false (a full prefill on both legs, no prior
# speculative decode), for one workload class, at 4K and 128K.
#
#   BONGO_GPU_LOCK_TIMEOUT=-1 ./bench/run-speculation-equiv.sh
#
# Env overrides: BONGO_SPEC_EQUIV_WORKLOAD, BONGO_SPEC_EQUIV_CONTEXTS,
# BONGO_SPEC_EQUIV_OUT, BONGO_SPEC_SCRATCH.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "${here}")"
out="${BONGO_SPEC_EQUIV_OUT:-$repo/bench/results/2026-09-28-speculation/equiv}"
workload="${BONGO_SPEC_EQUIV_WORKLOAD:-generic}"
contexts="${BONGO_SPEC_EQUIV_CONTEXTS:-4096,131072}"

export BONGO_SPEC_SCRATCH="${BONGO_SPEC_SCRATCH:-$HOME/.bongo/m32b/scratch}"
mkdir -p "$out" "$BONGO_SPEC_SCRATCH"
rm -f "$out/.equiv-complete"

run_leg() {
  local label="$1"; shift
  BONGO_SPEC_WORKLOADS="$workload" BONGO_SPEC_CONTEXTS="$contexts" BONGO_SPEC_REPEATS=0 \
    BONGO_SPEC_OUT="$out" "$here/run-speculation-ab.sh" "$label" "$@"
}

echo "== equivalence leg: baseline =="
run_leg equiv-baseline
echo "== equivalence leg: spec =="
run_leg equiv-spec --spec-type ngram-map-k4v

echo "== compare workload=$workload (cache-free) =="
python3 "$here/measure-speculation.py" compare \
  --baseline "$out/equiv-baseline-$workload.json" \
  --spec "$out/equiv-spec-$workload.json" \
  --out "$out/equiv-compare-$workload.json" || true

touch "$out/.equiv-complete"
echo "equivalence re-test complete"
