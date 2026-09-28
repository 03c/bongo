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

`bongo.sh`'s default `--backend auto` used to try SYCL first, on the assumption in the original ADR-0002 that SYCL was the baseline. It now tries **Vulkan** first and keeps SYCL as the fallback when no Vulkan device is reported; `--backend sycl` still selects SYCL unconditionally. The Stage-0 `--n-cpu-moe 16` placement baseline is unchanged and still pinned.

`docs/bongo-sh.md`'s Backends section still describes the old SYCL-first preference and the pre-BAS-72 NEO/GMM abort, both of which this measurement and the `ZEL_LIBRARY_PATH` fix supersede. That file carries uncommitted work from another task, so the refresh is left to whoever lands it.

