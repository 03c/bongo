#!/usr/bin/env bash
# build-llama-vulkan-lru.sh — build the BAS-139 MoE expert-cache engine (Vulkan).
#
# A pinned llama.cpp tree with tools/patches/moe-expert-cache.patch applied: a
# GPU-resident LRU cache over the expert weights that --n-cpu-moe / -ot pinned to
# host memory, seeded from the offline profile. This is BAS-76 Step 2.
#
# The container `bongo-llama-build:vulkan` carries libvulkan-dev + glslc/
# spirv-tools, so no host Vulkan SDK install is needed. The source tree is
# mounted read-write at /work; the build directory `build-lru-vulkan/` is created
# inside it and the server ends up at
# `<llama.cpp-dir>/build-lru-vulkan/bin/llama-server`. Point the A/B wrapper at it
# with BONGO_LLAMA_BIN, or a server run with `--moe-expert-cache*`.
#
# usage:
#   tools/build-llama-vulkan-lru.sh <llama.cpp-dir> [targets]
#
#   tools/build-llama-vulkan-lru.sh "$HOME/.bongo/engine/llama.cpp-lru" llama-server
#
# Set BONGO_LRU_APPLY_PATCH=0 to build without (re)applying the patch.
set -euo pipefail

SRC_DIR="${1:?usage: build-llama-vulkan-lru.sh <llama.cpp-dir> [targets]}"
TARGETS="${2:-llama-server}"
SRC_DIR="$(cd "$SRC_DIR" && pwd)"
PARENT="$(dirname "$SRC_DIR")"
NAME="$(basename "$SRC_DIR")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PATCH="$REPO/tools/patches/moe-expert-cache.patch"
IMAGE="${BONGO_VULKAN_BUILD_IMAGE:-bongo-llama-build:vulkan}"
APPLY_PATCH="${BONGO_LRU_APPLY_PATCH:-1}"
BUILD_DIR="${BONGO_LRU_BUILD_DIR:-build-lru-vulkan}"

if [[ "$APPLY_PATCH" == "1" || "$APPLY_PATCH" == "true" ]]; then
  [[ -f "$PATCH" ]] || { echo "build-llama-vulkan-lru: patch not found at $PATCH" >&2; exit 1; }
  if git -C "$SRC_DIR" apply --check --reverse "$PATCH" >/dev/null 2>&1; then
    echo "build-llama-vulkan-lru: BAS-139 patch already applied to $SRC_DIR"
  elif git -C "$SRC_DIR" apply "$PATCH"; then
    echo "build-llama-vulkan-lru: applied $(basename "$PATCH") to $SRC_DIR"
  else
    echo "build-llama-vulkan-lru: failed to apply $PATCH; is $SRC_DIR a clean pinned b11223 tree?" >&2
    exit 1
  fi
fi

docker run --rm --entrypoint sh --user "$(id -u):$(id -g)" \
  -v "$PARENT:/work" -w "/work/$NAME" "$IMAGE" -c "
set -e
if [ ! -d $BUILD_DIR ]; then
  cmake -B $BUILD_DIR -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='\$ORIGIN'
else
  # re-run configure so a changed src/CMakeLists.txt (new source files) is picked up
  cmake -B $BUILD_DIR -G 'Unix Makefiles' >/dev/null
fi
cmake --build $BUILD_DIR --config Release -j \"\$(nproc)\" --target $TARGETS
"

BIN="$SRC_DIR/$BUILD_DIR/bin/llama-server"
if [[ -x "$BIN" ]]; then
  echo "build-llama-vulkan-lru: built $BIN"
else
  echo "build-llama-vulkan-lru: build finished but $BIN is missing" >&2
  exit 1
fi
