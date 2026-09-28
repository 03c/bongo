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

## Amendment (2026-09-27, [BAS-56](/BAS/issues/BAS-56))

The default "MTP draft layer enabled via the server's speculative flags" is **withdrawn**. The published GGUF
has no MTP head and llama.cpp `qwen4exp` has no MTP path, so `--spec-draft-*` / `draft-mtp` cannot be used with
this model; speculation, when measured, uses the n-gram/PLE table. The "support MTP" capability requirement is
satisfied by llama.cpp in general but is inactive for `qwen4exp`. The rest of the decision stands. See
[`docs/research/intel-arc-b70.md`](../research/intel-arc-b70.md) §2.1.

## Amendment (2026-09-28, [BAS-72](/BAS/issues/BAS-72) — M3.0 backend A/B)

The default backend is now decided on measurement instead of assumption. Both backends were run with the shipped Stage-0 profile on the same pinned engine build, on an otherwise idle Arc Pro B70 (the single-GPU lock from [BAS-80](/BAS/issues/BAS-80) was held for the whole run):

- engine: llama.cpp `b11223` (`4da6337767f9`)
- tier `iq2_xs`, `n_ctx 131072`, 3 repeats, cold prefill (`--no-cache-prompt`) plus a `--delta 512` cached-turn run
- flags: `--ctx-size 131072 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 16 --cache-prompt`, device `Vulkan1` / `SYCL0`
- raw: `bench/results/2026-09-28-backend-ab/<backend>/raw/`

| metric | Vulkan1 | SYCL0 | SYCL / Vulkan |
| --- | ---: | ---: | ---: |
| 4096 prefill tok/s | 231.31 | 276.59 | 1.20x |
| 131072 prefill tok/s | 133.46 | 109.01 | 0.82x |
| 131072 decode tok/s | 8.00 | 4.69 | 0.59x |
| 131072 cold TTFT ms | 980705 | 1200549 | 1.22x |
| 512-token cached-turn TTFT ms (prefix 31744) | 4165 | 3925 | 0.94x |
| peak VRAM GiB at 131072 | 29.27 | 28.85 | 0.99x |

**SYCL does not become the default; Vulkan stays the default** — the rule (SYCL holds >= 1.3x on 131072 TTFT, cold and cached-turn, and stays within 10% on 131072 decode) was not met:

- vulkan build identity: vulkan build carries the expected backend library (required matches the leg) — pass
- sycl build identity: sycl build carries the expected backend library (required matches the leg) — pass
- 131072 cold TTFT: Vulkan 980705 ms vs SYCL 1200549 ms = 0.82x (required >= 1.3x) — **fail**
- 512-token cached-turn TTFT: Vulkan 4165 ms vs SYCL 3925 ms = 1.06x (required >= 1.3x) — **fail**
- 131072 decode tok/s: Vulkan 8.00 vs SYCL 4.69 tok/s = 0.59x (required >= 0.91x) — **fail**

`bongo.sh`'s default `--backend auto` used to try SYCL first, on the assumption in the original ADR-0002 that SYCL was the baseline. It now tries **Vulkan** first and keeps SYCL as the fallback when no Vulkan device is reported; `--backend sycl` still selects SYCL unconditionally. The Stage-0 `--n-cpu-moe 16` placement baseline is unchanged and still pinned. `docs/bongo-sh.md`'s Backends section was refreshed to match (`e871a5a`).
