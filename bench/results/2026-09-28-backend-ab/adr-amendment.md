## Amendment (2026-09-28, [BAS-72](/BAS/issues/BAS-72) — M3.0 backend A/B)

The default backend is now decided on measurement instead of assumption. Both backends were run with the shipped Stage-0 profile on the same pinned engine build, on an otherwise idle Arc Pro B70 (the single-GPU lock from [BAS-80](/BAS/issues/BAS-80) was held for the whole run):

- engine: llama.cpp `b11223` (`4da6337767f9`)
- tier `iq2_xs`, `n_ctx 131072`, 3 repeats, cold prefill (`--no-cache-prompt`) plus a `--delta 512` cached-turn run
- flags: `--ctx-size 131072 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 16 --cache-prompt`, device `Vulkan1` / `None`
- raw: `bench/results/2026-09-28-backend-ab/<backend>/raw/`

| metric | Vulkan1 | SYCL0 | SYCL / Vulkan |
| --- | ---: | ---: | ---: |
| 4096 prefill tok/s | 231.31 | n/a | n/a |
| 131072 prefill tok/s | 133.46 | n/a | n/a |
| 131072 decode tok/s | 8.00 | n/a | n/a |
| 131072 cold TTFT ms | 980705 | n/a | n/a |
| 512-token cached-turn TTFT ms (prefix 31744) | 4165 | n/a | n/a |
| peak VRAM GiB at 131072 | 29.27 | n/a | n/a |

**SYCL does not become the default; Vulkan stays the default.** No SYCL measurement completed in this run, so the rule below cannot be satisfied in SYCL's favour.

- vulkan build identity: vulkan build carries the expected backend library (required matches the leg) — pass
- 131072 cold TTFT: missing (required >= 1.3x) — not measured
- 512-token cached-turn TTFT: no SYCL result (required >= 1.3x) — not measured
- 131072 decode tok/s: missing (required >= 0.91x) — not measured

