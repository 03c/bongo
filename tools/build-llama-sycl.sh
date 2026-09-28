#!/usr/bin/env bash
# build-llama-sycl.sh — build a pinned llama.cpp SYCL tree inside the ggml oneAPI container.
#
# Used by the M3.1 IQ2_XS kernel work (BAS-74). The container
# `ghcr.io/ggml-org/llama.cpp:server-intel` carries the oneAPI 2025.3 DPC++ compiler and
# the Level Zero runtime, so no host compiler install is needed.
#
# usage:
#   tools/build-llama-sycl.sh <llama.cpp-dir> [targets]
# examples:
#   tools/build-llama-sycl.sh "$HOME/.bongo/engine/llama.cpp-pin"
#   tools/build-llama-sycl.sh "$HOME/.bongo/engine/llama.cpp-pin" "llama-bench test-backend-ops"
#
# The source tree is mounted read-write at /work; the build directory `build-sycl/` is created
# inside it. Apply the M3.1 patch first, if wanted:
#   git -C <llama.cpp-dir> apply tools/patches/iq2xs-sycl-integer-mmvq.patch
set -euo pipefail

SRC_DIR="${1:?usage: build-llama-sycl.sh <llama.cpp-dir> [targets]}"
TARGETS="${2:-llama-bench test-backend-ops}"
SRC_DIR="$(cd "$SRC_DIR" && pwd)"
PARENT="$(dirname "$SRC_DIR")"
NAME="$(basename "$SRC_DIR")"
IMAGE="${GGML_SYCL_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-intel}"

docker run --rm --entrypoint sh --user "$(id -u):$(id -g)" \
  -v "$PARENT:/work" -w "/work/$NAME" "$IMAGE" -c "
set -e
if [ ! -d build-sycl ]; then
  cmake -B build-sycl -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
fi
cmake --build build-sycl --config Release -j \"\$(nproc)\" --target $TARGETS
"
