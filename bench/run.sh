#!/usr/bin/env bash
# bongo benchmark entry point.
#
# One documented command.  Measures the running OpenAI-compatible endpoint and
# writes bench/results/<YYYY-MM-DD>-baseline/{matrix.json,matrix.md}.
#
#   ./bench/run.sh
#   ./bench/run.sh --tier iq3_xxs --contexts 1024,4096,32768,131072 --repeats 3
#   ./bench/run.sh --out-dir bench/results/2026-09-27-iq3
#
# Environment overrides: BONGO_BASE_URL, BONGO_MODEL, BONGO_TIER, BONGO_CONTEXTS,
# BONGO_REPEATS, BONGO_GGUF_DIR.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${here}/harness.py" --repo-root "$(dirname "${here}")" "$@"
