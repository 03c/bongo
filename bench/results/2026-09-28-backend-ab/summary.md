# M3.0 backend A/B — SYCL vs Vulkan (agentic profile)

Source: `bench/results/2026-09-28-backend-ab/` (raw files beside this summary).

## Vulkan (Stage 0 baseline, kept pinned)

- engine: llama.cpp `b11223`
- n_ctx: 131072, 128K needle: **pass**
- peak VRAM: 29.27 GiB

## Result table

| metric | Vulkan | SYCL | SYCL / Vulkan |
| --- | ---: | ---: | ---: |
| 4096 prefill tok/s | 231.31 | n/a | n/a |
| 131072 prefill tok/s | 133.46 | n/a | n/a |
| 131072 decode tok/s | 8.00 | n/a | n/a |
| 131072 TTFT ms (cold prefill) | 980705 | n/a | n/a |
| 512-token cached-turn TTFT ms | 4165 | n/a | n/a |
| peak VRAM GiB | 29.27 | n/a | n/a |

## Decision

**Vulkan stays the default.** No SYCL result was recorded in this run, so the SYCL branch of the decision rule cannot be satisfied. See the run log for why.

