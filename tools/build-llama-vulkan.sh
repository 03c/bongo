#!/usr/bin/env bash
# build-llama-vulkan.sh — build a pinned llama.cpp Vulkan tree in the ggml Vulkan container.
#
# Used by the M3.0b slot-checkpoint work (BAS-86). The patched engine that makes a
# restored slot reusable must be reproducible, while the stock Stage 0 Vulkan
# baseline stays selectable. The container `bongo-llama-build:vulkan` carries
# libvulkan-dev + glslc/spirv-tools, so no host Vulkan SDK install is needed.
#
# usage:
#   tools/build-llama-vulkan.sh <llama.cpp-dir> [targets]
#
# examples:
#   tools/build-llama-vulkan.sh "$HOME/.bongo/engine/llama.cpp-pin" llama-server
#   BONGO_APPLY_PATCH=0 tools/build-llama-vulkan.sh <llama.cpp-dir> llama-server
#
# The source tree is mounted read-write at /work; the build directory
# `build-vulkan/` is created inside it and the server ends up at
# `<llama.cpp-dir>/build-vulkan/bin/llama-server`. Pass that directory to
# `bongo.sh --llama-bin DIR` (or point the bench wrappers at it through
# `BONGO_LLAMA_BIN`).
#
# By default the BAS-86 patch `tools/patches/slot-checkpoints-sidecar.patch` is
# applied when it is not already present. Set `BONGO_APPLY_PATCH=0` for the stock
# baseline. The binaries are linked with an `$ORIGIN` RPATH, so the sibling shared
# libraries are found wherever the directory is copied or mounted on the host.
set -euo pipefail

SRC_DIR="${1:?usage: build-llama-vulkan.sh <llama.cpp-dir> [targets]}"
TARGETS="${2:-llama-server}"
SRC_DIR="$(cd "$SRC_DIR" && pwd)"
PARENT="$(dirname "$SRC_DIR")"
NAME="$(basename "$SRC_DIR")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PATCH="$REPO/tools/patches/slot-checkpoints-sidecar.patch"
M42_PATCH="$REPO/tools/patches/m4.2-vulkan-host-expert-upload.patch"
IMAGE="${BONGO_VULKAN_BUILD_IMAGE:-bongo-llama-build:vulkan}"
APPLY_PATCH="${BONGO_APPLY_PATCH:-1}"
APPLY_M42_PATCH="${BONGO_APPLY_M42_PATCH:-0}"

if [[ "$APPLY_PATCH" == "1" || "$APPLY_PATCH" == "true" ]]; then
  if [[ ! -f "$PATCH" ]]; then
    echo "build-llama-vulkan: patch not found at $PATCH" >&2
    exit 1
  fi
  if git -C "$SRC_DIR" apply --check --reverse "$PATCH" >/dev/null 2>&1; then
    echo "build-llama-vulkan: BAS-86 patch already applied to $SRC_DIR"
  elif git -C "$SRC_DIR" apply "$PATCH"; then
    echo "build-llama-vulkan: applied $(basename "$PATCH") to $SRC_DIR"
  else
    echo "build-llama-vulkan: failed to apply $PATCH; is $SRC_DIR a clean pinned tree?" >&2
    exit 1
  fi
fi

if [[ "$APPLY_M42_PATCH" == "1" || "$APPLY_M42_PATCH" == "true" ]]; then
  if [[ ! -f "$M42_PATCH" ]]; then
    echo "build-llama-vulkan: patch not found at $M42_PATCH" >&2
    exit 1
  fi
  if git -C "$SRC_DIR" apply --check --reverse "$M42_PATCH" >/dev/null 2>&1; then
    echo "build-llama-vulkan: BAS-155 M4.2 patch already applied to $SRC_DIR"
  elif git -C "$SRC_DIR" apply "$M42_PATCH"; then
    echo "build-llama-vulkan: applied $(basename "$M42_PATCH") to $SRC_DIR"
  else
    echo "build-llama-vulkan: failed to apply $M42_PATCH; is $SRC_DIR clean apart from the BAS-86 patch?" >&2
    exit 1
  fi
fi

docker run --rm --entrypoint sh --user "$(id -u):$(id -g)" \
  -v "$PARENT:/work" -w "/work/$NAME" "$IMAGE" -c "
set -e
if [ ! -d build-vulkan ]; then
  cmake -B build-vulkan -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='\$ORIGIN'
fi
cmake --build build-vulkan --config Release -j \"\$(nproc)\" --target $TARGETS
"

BIN="$SRC_DIR/build-vulkan/bin/llama-server"
if [[ -x "$BIN" ]]; then
  echo "build-llama-vulkan: built $BIN"
else
  echo "build-llama-vulkan: build finished but $BIN is missing" >&2
  exit 1
fi
