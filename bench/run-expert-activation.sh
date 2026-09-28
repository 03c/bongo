#!/usr/bin/env bash
# Build and run the MoE router capture for the bongo reference box (BAS-66 / R4).
#
# The reference box ships only prebuilt llama.cpp libraries, no compiler and no
# source tree.  This script therefore:
#
#   1. fetches the llama.h / ggml headers at the exact commit the local build
#      reports (so the by-value `llama_context_params` ABI matches the .so);
#   2. gets a C compiler (zig, installed user-locally through pip);
#   3. compiles bench/tools/route_capture.c and links it against the *existing*
#      libllama.so / libggml*.so of the bongo llama.cpp build;
#   4. runs the prompt corpora through it, writing one `TOPK` TSV per corpus.
#
# The capture runs with n_gpu_layers=0 (CPU).  Router selections are a property
# of the weights, not of the backend, and staying off the GPU avoids the 33 GiB
# expert set versus 32 GiB VRAM ceiling entirely.
#
# Usage:
#   bench/run-expert-activation.sh [options]
#
#   --model PATH        first GGUF shard (required unless --smoke)
#   --llama-bin DIR     llama.cpp build dir (default ~/.bongo/llama/b11223/vulkan)
#   --corpora DIR       prompt dir (default bench/results/2026-09-28-expert-activation/corpora)
#   --out-dir DIR       capture dir (default bench/results/2026-09-28-expert-activation/raw)
#   --decode N          teacher-forced decode steps per corpus (default 0; only
#                       corpora with a matching roster entry are run)
#   --work DIR          build scratch dir (default ${PAPERCLIP_SCRATCH_DIR:-/tmp}/expert-activation-build)
#   --smoke             build, then capture one 40-token prompt and exit
#   --reuse-build       skip header fetch/compile if the binary already exists

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_COMMIT="4da6337767f973e2b4d0797e5b323d77d8565e4a"   # reported by llama-cli --version (build 11223)

MODEL=""
LLAMA_BIN="${BONGO_LLAMA_BIN:-$HOME/.bongo/llama/b11223/vulkan}"
CORPORA="$ROOT/bench/results/2026-09-28-expert-activation/corpora"
OUT_DIR="$ROOT/bench/results/2026-09-28-expert-activation/raw"
DECODE=0
WORK="${PAPERCLIP_SCRATCH_DIR:-/tmp}/expert-activation-build"
SMOKE=0
REUSE=0
N_CTX=8192

while (( $# )); do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --llama-bin) LLAMA_BIN="$2"; shift 2;;
    --corpora) CORPORA="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --decode) DECODE="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --n-ctx) N_CTX="$2"; shift 2;;
    --smoke) SMOKE=1; shift;;
    --reuse-build) REUSE=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

log() { printf '[expert-activation] %s\n' "$*" >&2; }
die() { printf '[expert-activation] ERROR: %s\n' "$*" >&2; exit 1; }

mkdir -p "$WORK" "$OUT_DIR"
HDR="$WORK/hdr"
BIN="$WORK/route_capture"

# --------------------------------------------------------------------------- #
# 1. headers at the pinned llama.cpp commit
# --------------------------------------------------------------------------- #
if (( ! REUSE )) || [[ ! -f "$BIN" ]]; then
  mkdir -p "$HDR"
  log "fetching llama.cpp headers at $LLAMA_COMMIT"
  for f in include/llama.h ggml/include/ggml.h ggml/include/ggml-cpu.h \
           ggml/include/ggml-backend.h ggml/include/ggml-alloc.h \
           ggml/include/ggml-opt.h ggml/include/gguf.h; do
    out="$HDR/$(basename "$f")"
    if [[ ! -s "$out" ]]; then
      curl -fsSL "https://raw.githubusercontent.com/ggml-org/llama.cpp/$LLAMA_COMMIT/$f" -o "$out" \
        || die "could not fetch $f"
    fi
  done

  # 2. a compiler
  ZIG="${ZIG:-}"
  if [[ -z "$ZIG" ]]; then
    if command -v zig >/dev/null 2>&1; then
      ZIG="$(command -v zig)"
    else
      log "installing zig into the user site (pip install --user ziglang)"
      python3 -m pip install --user --quiet ziglang || die "could not install zig"
      ZIG="$(find "$HOME/.local" -type f -name zig 2>/dev/null | head -1)"
    fi
  fi
  [[ -x "$ZIG" ]] || die "no usable zig compiler (set ZIG=...)"

  # 3. build against the existing shared libraries
  log "compiling route_capture.c with $( "$ZIG" version )"
  "$ZIG" cc -O2 -I "$HDR" -o "$BIN" "$ROOT/bench/tools/route_capture.c" \
      -L"$LLAMA_BIN" -lllama -lggml -lggml-base
fi

# ggml_backend_load_all() resolves backend plugins relative to the executable,
# so the capture binary has to live next to libggml-cpu-*.so.
cp "$BIN" "$LLAMA_BIN/route_capture"
RUN_BIN="$LLAMA_BIN/route_capture"

export LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ -d "$HOME/.bongo/run/vulkan-icds" ]] && export VK_ICD_FILENAMES="$HOME/.bongo/run/vulkan-icds"

if (( SMOKE )); then
  [[ -n "$MODEL" ]] || die "--smoke needs --model"
  tmp="$(mktemp -d)"
  printf 'Hello, world. The quick brown fox jumps over the lazy dog.\n' > "$tmp/p.txt"
  "$RUN_BIN" "$MODEL" "$tmp/p.txt" "$tmp/out.tsv" 64
  awk -F'\t' 'NR==1{print "smoke: first layer ne="$3","$4","$5","$6" values="(NF-6)}' "$tmp/out.tsv"
  log "smoke capture ok: $tmp/out.tsv"
  exit 0
fi

[[ -n "$MODEL" ]] || die "--model is required"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"

# --------------------------------------------------------------------------- #
# 4. capture
# --------------------------------------------------------------------------- #
# corpora that get a teacher-forced decode tail (name:steps); keep the prefill
# and decode token sets disjoint so the transfer check is honest
declare -A DECODE_STEPS=( [doc]="$DECODE" [chat]="$DECODE" )

for name in doc code chat convo; do
  src="$CORPORA/$name.txt"
  [[ -f "$src" ]] || { log "skip $name (no prompt)"; continue; }
  steps="${DECODE_STEPS[$name]:-0}"
  if (( DECODE > 0 && steps > 0 )); then
    log "capture $name (decode tail $steps)"
    "$RUN_BIN" "$MODEL" "$src" "$OUT_DIR/$name.tsv" "$N_CTX" "$steps" "$OUT_DIR/${name}_dec.tsv"
  else
    log "capture $name"
    "$RUN_BIN" "$MODEL" "$src" "$OUT_DIR/$name.tsv" "$N_CTX"
  fi
done

log "done; captures in $OUT_DIR"
