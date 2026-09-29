# 2026-09-29 — M4.1 host/CPU decomposition and lever screen (BAS-144)

Raw files for [BAS-144](/BAS/issues/BAS-144). Write-up:
[`docs/research/host-cpu-decomposition.md`](../../../docs/research/host-cpu-decomposition.md).
Reproduce with [`bench/run-m4.1-host-cpu.sh`](../../run-m4.1-host-cpu.sh) (one
GPU-lock hold, resumable).

Engine: llama.cpp `b11223` (`4da633776`), Vulkan, tier `iq2_xs`, q8 KV,
`--n-cpu-moe 16`, `--flash-attn on`, `--ctx-size 131072`, `--parallel 1`.

## Layout

```
ctx16k/<config>/profile-host-split.json   per-thread CPU + storage decomposition
ctx16k/<config>/profile.json              the warm-prefix lever screen
ctx128k/<config>/profile-host-split.json  16K + 128K headers
needle/<config>/needle.json               sentinel-recall guard
<config>/server-flags.json                exact server argv
<config>/harness.log, llama-server.log    raw run logs
session.log                               the driver's stage log
```

## Headline numbers (delta 512 `prompt_ms`)

| prefix | baseline | `--load-mode none` | Δ | frozen M3.6 baseline | Δ vs frozen |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 384 | 5 666.9 | **3 569.4** | −37.0% | 3 982.5 | **−10.4%** |
| 127 999 | 7 661.7 | **5 480.6** | −28.5% | 5 899.5 | **−7.1%** |

16K screen: `no_op_offload` (CPU expert matmul) **+62%**, `threads16` +1.1%.
Needle: `baseline` and `lm_none` both PASS.

Decomposition (16K baseline): main host thread **2 180 ms (38%)**, CPU workers
**0 ms**, other host 640 ms, not-on-CPU **2 861 ms (50%, incl. 0.52 GiB of
in-turn SSD page-cache re-reads)**.
