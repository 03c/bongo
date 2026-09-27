# ADR-0002 — baseline engine: pinned llama.cpp with the SYCL backend

- Status: Accepted
- Date: 2026-09-27
- Deciders: CTO
- Related: [ADR-0001](0001-runtime-architecture.md)

## Context

bongo needs a working, benchmarkable engine on the Arc Pro B70 before we build any optimisation layer. The
engine must: read the published GSQ-RCO GGUFs (including `IQ2_XS` / `IQ3_XXS`), understand the `qwen4exp`
architecture, support MoE expert CPU offload, support MTP, expose an OpenAI-compatible server, and run on
Intel's compute stack.

Verified upstream (2026-09-27, llama.cpp master):

- `LLM_ARCH_QWEN4EXP` (`"qwen4exp"`) is present in `src/llama-arch.cpp`, with a `llama_model_qwen4exp`
  constructor in `src/llama-model.cpp`.
- MTP/NextN is supported (`n_layer_nextn`, `LLAMA_CONTEXT_TYPE_MTP`).
- SYCL GPU dequant covers `IQ1_S`..`IQ4_XS`; docs list Arc B-Series as supported.
- `llama-server` exposes `--cpu-moe`, `--n-cpu-moe N`, `-ot`, `--cache-type-k/-v`, and `--spec-draft-*`.

## Decision

Use **llama.cpp built from a pinned commit with `-DGGML_SYCL=ON`** (compiled with Intel `icpx`, run on Level
Zero) as bongo's baseline engine. `bongo.sh` pins the commit and can either fetch a matching prebuilt SYCL
binary or build from source.

Configuration defaults to be established by measurement, not guessed:

- `--n-gpu-layers 99` plus `-ot` rules that keep as many experts as fit on the GPU and the rest on CPU;
- `--cache-type-k q8_0 --cache-type-v q8_0` at >=128K (f16 if quality demands it);
- `--flash-attn on`;
- MTP draft layer enabled via the server's speculative flags;
- `--no-mmap` only if measurement shows mmap thrashing; SSD streaming of the n-gram table is a Stage 1 item.

## Alternatives considered

- **llama.cpp Vulkan.** Simpler dependency set (Mesa ANV), but weaker MoE and i-quant kernels. Kept as a
  fallback; benchmark it on a small model to have a number.
- **IPEX-LLM prebuilt SYCL.** Zero build step, but fork drift and no path to adaptive cache work. Kept as an
  optional first-boot fallback only.
- **Pinning an older "recommended release".** The docs' verified SYCL release targets older oneAPI (2025.1) and
  a B580; our arch (`qwen4exp`) is newer. We pin master at a known commit and record it, rather than an old tag.

## Consequences

- Fast path to a real endpoint and real numbers; no bespoke kernels required for v1.
- We inherit llama.cpp's SYCL gaps (e.g. MoE expert kernels that are still maturing). Measurements must be per
  backend and per tier so we can tell engine cost from hardware cost.
- Pinning is mandatory: llama.cpp moves fast, and an unpinned build makes benchmark results irreproducible.
- The one-command setup now depends on a working oneAPI toolchain on the target OS; test this on a clean Fedora
  44 box as an explicit acceptance step.

## Rollback path

`bongo.sh` writes the exact pinned revision and flags into a generated config. Reverting the pin and the flags
restores the previous working state; the model files and API surface do not change. If SYCL cannot be made to
work on the target, the same wrapper can invoke the Vulkan build with no change to the download or server
contract.
