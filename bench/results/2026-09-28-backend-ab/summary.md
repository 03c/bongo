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

## Result table

| metric | Vulkan | SYCL | SYCL / Vulkan |
| --- | ---: | ---: | ---: |
| 4096 prefill tok/s | 231.31 | n/a | n/a |
| 131072 prefill tok/s | 133.46 | n/a | n/a |
| 131072 decode tok/s | 8.00 | n/a | n/a |
| 131072 cold TTFT ms | 980705 | n/a | n/a |
| 512-token cached-turn TTFT ms (prefix 4096) | 3486 | n/a | n/a |
| 512-token cached-turn TTFT ms (prefix 31744) | 4165 | n/a | n/a |
| 512-token steady cached turn TTFT ms (prefix 4096) | 179 | n/a | n/a |
| 512-token steady cached turn TTFT ms (prefix 31744) | 282 | n/a | n/a |
| peak VRAM GiB at 131072 | 29.27 | n/a | n/a |

TTFT ratio is `Vulkan / SYCL` (> 1 means SYCL is faster); tok/s ratios are `SYCL / Vulkan` (> 1 means SYCL is faster).

## Decision rule

> SYCL becomes the default only if it holds >= 1.3x on 131072 TTFT and stays within 10% on 131072 decode; otherwise Vulkan stays the default.

The ticket's product metric is the agentic turn under prefix reuse, so the 1.3x gate is applied to the 131072 cached-turn TTFT as well as the cold TTFT, and all gates must pass.

| gate | required | measured | result |
| --- | --- | --- | --- |
| vulkan build identity | matches the leg | vulkan build carries the expected backend library | **pass** |
| 131072 cold TTFT | >= 1.3x | missing | not measured |
| 512-token cached-turn TTFT | >= 1.3x | no SYCL result | not measured |
| 131072 decode tok/s | >= 0.91x | missing | not measured |

## Decision

**Vulkan stays the default.** No SYCL result was recorded in this run, so the SYCL branch of the decision rule cannot be satisfied. See the run log for why.

