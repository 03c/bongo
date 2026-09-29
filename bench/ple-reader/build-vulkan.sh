#!/usr/bin/env bash
# M3.4b: build the pinned llama.cpp (b11223 / 4da633776) with the PLE reader
# patch, Vulkan backend, inside the oneAPI container (the reference box has no
# C/C++ toolchain and no passwordless sudo).
#
# The container image is the ggml-org intel server image plus the Vulkan build
# deps (libvulkan-dev, glslc, glslang, SPIRV headers).  Build it once:
#
#   docker build -t bongo-llama-build:vulkan - <<'EOF'
#   FROM ghcr.io/ggml-org/llama.cpp:server-intel
#   RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#         libvulkan-dev glslang-tools glslc spirv-headers spirv-tools && rm -rf /var/lib/apt/lists/*
#   EOF
#
# Usage:
#   bench/ple-reader/build-vulkan.sh [repo-dir]
#
# It expects a llama.cpp checkout already at the pinned revision in
# $BONGO_HOME/engine/llama.cpp-ple and applies ple-reader.patch.  Pass a fresh
# checkout to reproduce from scratch.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bongo_home="${BONGO_HOME:-$HOME/.bongo}"
engine="${1:-$bongo_home/engine/llama.cpp-ple}"
image="${BONGO_BUILD_IMAGE:-bongo-llama-build:vulkan}"
patch_file="$repo/bench/ple-reader/ple-reader.patch"

[[ -d "$engine" ]] || { echo "no llama.cpp checkout at $engine" >&2; exit 2; }

if ! git -C "$engine" diff --quiet 2>/dev/null; then
  echo "warning: $engine has local changes; assuming the PLE reader patch is already applied" >&2
else
  git -C "$engine" apply --check "$patch_file" \
    && git -C "$engine" apply "$patch_file" \
    && echo "applied $patch_file"
fi

docker run --rm --entrypoint sh --user "$(id -u):$(id -g)" \
  -v "$bongo_home/engine:/work" -w "/work/$(basename "$engine")" "$image" -c '
set -e
if [ ! -d build-vulkan ]; then
  cmake -B build-vulkan -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON
fi
cmake --build build-vulkan --config Release -j "${BUILD_JOBS:-12}" --target llama-server llama-cli
'

echo "built $engine/build-vulkan/bin/llama-server"
