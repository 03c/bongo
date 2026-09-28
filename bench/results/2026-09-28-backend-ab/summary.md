# M3.0 backend A/B — SYCL vs Vulkan (agentic profile)

Source: `bench/results/2026-09-28-backend-ab/` — raw files beside this summary, listed per backend below.

Profile (shipped): `--n-cpu-moe 16 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`, IQ2_XS, `n_ctx 131072`, one slot, `bench/harness.py --no-cache-prompt` (cold prefill) x3 plus `measure-prefix-cache.py --delta 512` for the cached turn.

## vulkan — run identity

- engine: llama.cpp `b11223` (`4da6337767f9`), build backend `Vulkan`
- backend library present: vulkan:libggml-vulkan.so
- linked: libggml-base.so.0, libggml.so.0
- binary: `/home/cchild/.bongo/llama/b11223/vulkan/llama-server`
- tier: `iq2_xs`, runtime `/home/cchild/.bongo/runtime`, GPU Intel Corporation Battlemage G31 [Arc Pro B70]
- server flags: `--ctx-size 131072 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 16 --cache-prompt --device Vulkan1`
- n_ctx: 131072, repeats: 3, needle at 131072: **pass**
- peak VRAM at 131072: 29.27 GiB (at 4096: 28.82 GiB)
- raw: `vulkan/raw/iq2_xs-ctx131072-r1.json`, `vulkan/raw/iq2_xs-ctx131072-r2.json`, `vulkan/raw/iq2_xs-ctx131072-r3.json`, `vulkan/raw/iq2_xs-ctx4096-r1.json`, `vulkan/raw/iq2_xs-ctx4096-r2.json`, `vulkan/raw/iq2_xs-ctx4096-r3.json`, `vulkan/raw/needle.json`

## sycl — run identity

- engine: llama.cpp `b11223` (`4da6337767f9`), build backend `SYCL0`
- backend library present: sycl:libggml-sycl.so
- linked: libggml-base.so.0, libggml-cpu.so.0, libggml-sycl.so.0, libggml.so.0, libsycl.so.8
- binary: `/home/cchild/.bongo/llama/b11223/sycl/llama-server`
- tier: `iq2_xs`, runtime `/home/cchild/.bongo/runtime`, GPU Intel Corporation Battlemage G31 [Arc Pro B70]
- server flags: `--ctx-size 131072 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 16 --cache-prompt`
- n_ctx: 131072, repeats: 3, needle at 131072: **pass**
- peak VRAM at 131072: 28.85 GiB (at 4096: 28.85 GiB)
- raw: `sycl/raw/iq2_xs-ctx131072-r1.json`, `sycl/raw/iq2_xs-ctx131072-r2.json`, `sycl/raw/iq2_xs-ctx131072-r3.json`, `sycl/raw/iq2_xs-ctx4096-r1.json`, `sycl/raw/iq2_xs-ctx4096-r2.json`, `sycl/raw/iq2_xs-ctx4096-r3.json`, `sycl/raw/needle.json`

## Result table

| metric | Vulkan | SYCL | SYCL / Vulkan |
| --- | ---: | ---: | ---: |
| 4096 prefill tok/s (higher better) | 231.31 | 276.59 | 1.20x |
| 131072 prefill tok/s (higher better) | 133.46 | 109.01 | 0.82x |
| 131072 decode tok/s (higher better) | 8.00 | 4.69 | 0.59x |
| 131072 cold TTFT ms (lower better) | 980705 | 1200549 | 1.22x |
| 512-token cached-turn TTFT ms, prefix 4096 (lower better) | 3486 | 2547 | 0.73x |
| 512-token cached-turn TTFT ms, prefix 31744 (lower better) | 4165 | 3925 | 0.94x |
| 512-token steady cached turn TTFT ms, prefix 4096 (lower better) | 179 | 359 | 2.00x |
| 512-token steady cached turn TTFT ms, prefix 31744 (lower better) | 282 | 403 | 1.43x |
| peak VRAM GiB at 131072 | 29.27 | 28.85 | 0.99x |

Every ratio is `SYCL / Vulkan`. Above 1 favours SYCL on the tok/s rows and against it on the ms and GiB rows.

## Decision rule

> SYCL becomes the default only if it holds >= 1.3x on 131072 TTFT and stays within 10% on 131072 decode; otherwise Vulkan stays the default.

The ticket's product metric is the agentic turn under prefix reuse, so the 1.3x gate is applied to the 131072 cached-turn TTFT as well as the cold TTFT, and all gates must pass.

| gate | required | measured | result |
| --- | --- | --- | --- |
| vulkan build identity | matches the leg | vulkan build carries the expected backend library | **pass** |
| sycl build identity | matches the leg | sycl build carries the expected backend library | **pass** |
| 131072 cold TTFT | >= 1.3x | Vulkan 980705 ms vs SYCL 1200549 ms = 0.82x | **fail** |
| 512-token cached-turn TTFT | >= 1.3x | Vulkan 4165 ms vs SYCL 3925 ms = 1.06x | **fail** |
| 131072 decode tok/s | >= 0.91x | Vulkan 8.00 vs SYCL 4.69 tok/s = 0.59x | **fail** |

## Decision

**Vulkan stays the default.** The SYCL branch of the decision rule was not met:

- vulkan build identity: vulkan build carries the expected backend library (required matches the leg) — passed
- sycl build identity: sycl build carries the expected backend library (required matches the leg) — passed
- 131072 cold TTFT: Vulkan 980705 ms vs SYCL 1200549 ms = 0.82x (required >= 1.3x) — **failed**
- 512-token cached-turn TTFT: Vulkan 4165 ms vs SYCL 3925 ms = 1.06x (required >= 1.3x) — **failed**
- 131072 decode tok/s: Vulkan 8.00 vs SYCL 4.69 tok/s = 0.59x (required >= 0.91x) — **failed**

