#!/usr/bin/env bash
# GPU-free self-tests for the M3.2 speculation tooling.
#
#   ./bench/speculation-selftest.sh
#
# Proves the reference verify core matches plain greedy, the acceptance policy
# behaves, and the measurement/compare plumbing parses and compares correctly.
# It does not touch the GPU or a model.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${here}/spec_verify_core.py"
python3 "${here}/measure-speculation.py" self-test

echo "speculation self-tests: ALL PASSED"
