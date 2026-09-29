# M4.6 (BAS-167) — CTO independent reproduction of the shipped default

Two fresh `bench/run-m4.5-default.sh ctx128` runs on the shipped `bongo.sh`
default at `--ctx 131072`, taken by the CTO on 2026-09-29 to close
[BAS-62](/BAS/issues/BAS-62) against the re-baselined target (>=19.5 tok/s 4K
decode, median of 3).

- Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`) +
  M4.2 patch, Vulkan1 (Intel Arc Pro B70), tier `iq2_xs`, q8 KV, `--load-mode none`.
- Placement resolved by `--placement auto` at 131072 to `--n-cpu-moe 12`
  (config in each `ctx128/bongo-config.json`).
- Workload: 4096-token prompt, 128 generated tokens, 3 runs, `cache_prompt=false`,
  after the harness warm-up (`bench/run-m4.5-default.sh`, `measure_decode4k`).

| session | run0 | run1 | run2 | median |
| --- | ---: | ---: | ---: | ---: |
| BAS-163 (`2026-09-29-m4.5-auto-default/ctx128`) | 19.035 | 19.575 | 19.556 | **19.556** |
| CTO run1 (`run1/ctx128`) | 19.102 | 19.451 | 19.421 | **19.421** |
| CTO run2 (`run2/ctx128`) | 19.109 | 19.444 | 19.564 | **19.444** |

Pooled 9-run median: **19.444 tok/s**. Run0 is consistently the cold outlier
(~19.0-19.1); the two warm runs land ~19.42-19.56. The shipped default therefore
straddles the 19.5 median gate rather than reliably clearing it.
