# M3.4b — PLE reader engine integration and A/B

Engine-level follow-up to [BAS-77](/BAS/issues/BAS-77) (the standalone reader).
Owner: Coder ([BAS-79](/BAS/issues/BAS-79)). Parent: [BAS-62](/BAS/issues/BAS-62).

The standalone reader proved the mechanism on the real tensor. This directory
carries the **llama.cpp integration** and the engine harness that measures it
end to end.

## Status

- Reader + selftest **pass** against the patched tree (see **Correctness**).
- Engine integration **confirmed**: `--ple-reader on` logs
  `PLE reader enabled for per_layer_token_embd.weight (320001536 rows x 90 B/row,
  off=361831168, io_depth=16, cache=1000000 rows, window=256)`.
- The 4K/128K A/B over `baseline`/`m33` x `off`/`on` is the measurement in
  flight; raw files land under `bench/results/2026-09-28-ple-reader-engine/`.
- Measurement runs serialise on the shared single-GPU `flock` ([BAS-80](/BAS/issues/BAS-80));
  they queue rather than overlap, because two servers on the Arc B70 corrupt
  every number.

## What was built

An O_DIRECT row reader — a C++ port of BAS-77's `IOPool` + `RowCache` — that
serves `GGML_OP_GET_ROWS` on a `TENSOR_READ_LAZY` PLE tensor from a bounded
read pool and a bounded LRU row cache, instead of one mmap page fault per row.
`--ple-reader off` keeps the existing mmap + lazy path byte-for-byte.

| file | what |
| --- | --- |
| `ple-reader.patch` | the full patch against llama.cpp `4da633776` (`b11223`) |
| `build-vulkan.sh` | container build (no host toolchain) |
| `run-ab.sh` | engine A/B runner (baseline / M3.3 config, `--ple-reader off\|on`) |
| `run-all.sh` | the full 2x2 matrix (`baseline`/`m33` x `off`/`on`) |
| `ple_reader_selftest.cpp` | reader bytes == `pread`, straddling rows + duplicates |
| `summarize-ab.py` | join the four `matrix.json` files into the off/on table, and state the acceptance verdict |

### Integration shape

The patch has four parts:

1. **Reader + registry** — `ggml/include/ggml-ple-reader.h`,
   `ggml/src/ggml-ple-reader.cpp`. Lives in `ggml-base` so llama registers and
   the CPU backend gathers through one registry. One O_DIRECT fd per tensor, a
   fixed worker pool, page-aligned bounce buffers reused across the run, an LRU
   of raw rows, and a bounded window of in-flight reads per gather wave.
2. **GET_ROWS path** — `ggml/src/ggml-cpu/ops.cpp`. When `src0` is quantized,
   2-D, and registered, thread 0 gathers the raw rows through the reader and
   dequantizes them into `dst`; the other threads wait on the ggml barrier.
   Anything else falls through to the untouched mmap implementation.
3. **Plumbing** — `--ple-reader on|off|auto` (default `off`) plus
   `--ple-reader-cache-rows`, `--ple-reader-io-depth`, `--ple-reader-window`.
   `llama_model_loader` registers the reader for each lazy tensor in
   `load_all_data`; `llama_model_free` drops it.
4. **Fallback** — the mmap mapping and the lazy ranges are never removed, so
   `off` is the old path and a reader that cannot open the shard logs a warning
   and keeps mmap.

## Build

```sh
# one-time build image (Vulkan deps on top of the ggml-org intel image)
docker build -t bongo-llama-build:vulkan - <<'EOF'
FROM ghcr.io/ggml-org/llama.cpp:server-intel
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libvulkan-dev glslang-tools glslc spirv-headers spirv-tools && rm -rf /var/lib/apt/lists/*
EOF

bench/ple-reader/build-vulkan.sh
```

## Correctness

The selftest is built against the patched `ggml-base` and passes:

```sh
# inside the build container, from the patched tree
#   docker run --rm --entrypoint sh -v $HOME/.bongo/engine:/work \
#     -w /work/llama.cpp-ple --user $(id -u):$(id -g) bongo-llama-build:vulkan -c '...'
g++ -O2 -std=c++17 -pthread -Iggml/include -Iggml/src -Iggml/src/ggml-cpu \
  ggml/src/ggml-ple-reader.cpp selftest/ple_reader_selftest.cpp \
  -Lbuild-vulkan/bin -lggml-base -o /work/ple_selftest
LD_LIBRARY_PATH=build-vulkan/bin /work/ple_selftest
# PASS: 20000 rows match pread (straddle + duplicates + cache)
```

Engine-side: a server started with `--ple-reader on` opens a **second,
O_DIRECT** fd on the shard that holds the PLE table (recorded by
`run-ab.sh` as `reader-process.json`), and a deterministic completion is
identical with `off` and `on`.

## Measure

```sh
# Stage-0 pinned baseline config
bench/ple-reader/run-ab.sh --config baseline --reader off --label baseline-off
bench/ple-reader/run-ab.sh --config baseline --reader on  --label baseline-on
# M3.3 byte-budget placement (BAS-76)
bench/ple-reader/run-ab.sh --config m33 --reader off --label m33-off
bench/ple-reader/run-ab.sh --config m33 --reader on  --label m33-on
```

Each run writes `matrix.json`, `matrix.md`, `raw/`, `server-flags.json` and
`reader-process.json` under `bench/results/2026-09-28-ple-reader-engine/<label>/`.
`BONGO_DEVICE` (default `Vulkan1`) and `BONGO_PORT` (default `8090`) are
overridable; the runner refuses to start if the port is already serving, and it
queues on the shared single-GPU `flock` ([BAS-80](/BAS/issues/BAS-80)).

Summarise a (possibly partial) set of runs:

```sh
python3 bench/ple-reader/summarize-ab.py \
  --md-out bench/results/2026-09-28-ple-reader-engine/summary.md
# or run the whole matrix in one command:
bench/ple-reader/run-all.sh
```

`summary.md` also carries the verdict the issue is judged on, so the numbers are
not read by eye: `pass` / `fail` / `fail-no-gain` / `incomplete`, the list of metrics that moved
more than 5% against the reader (throughput higher-is-better, TTFT lower-is-better), the prefill
gains, and the RSS growth per pair next to the 26.82 GiB PLE table so "the table is never forced
resident" is checkable. A partial A/B always reports `incomplete`; it never reports `pass`.
The provenance block prints tier, ctx, `--n-cpu-moe`, KV types, flash-attn, device,
`--ple-reader` and the engine commit for each config, read from that run's `server-flags.json`.
