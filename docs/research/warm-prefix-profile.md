# M3.6 — the warm-prefix (cached-turn) profile and the dominant cost

Work for [BAS-130](/BAS/issues/BAS-130) (M3.6), part of [BAS-62](/BAS/issues/BAS-62).
Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`), Vulkan.
Author: Coder (Paperclip). Date: 2026-09-28.

> **Status: measurement pending.** The reproduction scripts are committed and
> validated against the mock server, but the GPU run is queued behind the
> single-GPU flock held by the BAS-132 long-context measurement (BAS-80). The
> "Measured evidence" section below is from **already-committed** results; the
> warm-turn table for this task and the ablation ranking are produced by
> `bench/run-warm-prefix-profile.sh` and will replace the `[pending]` rows.

## Why this task exists

M3.1's measured result ([BAS-74](/BAS/issues/BAS-74)) is that the integer
IQ2_S/IQ2_XXS/IQ1_M MMVQ path buys only ~1.05x on the cached-turn metric, and
that FP16 expansion is not the dominant cached-turn cost. [ADR-0003](0003-engine-direction.md)
makes that kernel path the central lever, so the remaining milestones were
aimed at a cost nobody had measured. This task measures where a 512-token
warm-prefix delta turn actually goes and names the dominant cost.

The product metric is per-turn TTFT under prefix reuse: a 512-token
continuation over an already-cached prefix. Targets are **<=3 s at 16K** and
**<=5 s at 128K**, with **>=25 tok/s** at 4K decode. After M3.1 the box was at
~3.7 s at 16K and ~14 tok/s at 4K.

## Method

One server config at a time, shipped flags (`--n-cpu-moe 16`, q8 KV,
`flash-attn on`, Vulkan, `--ctx-size 131072`), with **exactly one flag changed**
per ablation. For each config the profiler primes a prefix, then measures the
delta turn against the cached KV:

| case | prompt | cache_prompt | what it is |
| --- | --- | --- | --- |
| `cold` | P | false | the one full prefill that primes the slot |
| `hit` | P | true | full reuse; ~4 new tokens = fixed per-request cost |
| `grow_D` | P+D | true | **the measured delta turn**, D new tokens |
| `decode` | P | true, `max_tokens=32` | decode rate at context P after a hit |

`grow_D - hit` over `D` is the marginal ms/new-token while attending over the
cached prefix; several D values give a slope, not a single point.

The component attribution is a **critical-path probe**: take one component off
the fast GPU onto the slow CPU and measure how much the delta turn changes. A
large positive change names a component that matters; a change near zero names
one hidden behind other work. The configs are:

| config | change | component probed |
| --- | --- | --- |
| `baseline` | shipped (`--n-cpu-moe 16`) | reference |
| `ncmoe24` / `ncmoe8` | +/- 8 MoE layers' experts on CPU | MoE expert compute + placement |
| `fa_off` | `--flash-attn off` | the 12 full-attention layers |
| `attn_cpu` | `-ot attn_{qkv,output,gate,q,k,v}=CPU` | full-attention weights/compute |
| `ssm_cpu` | `-ot ssm_{out,conv1d,alpha,beta}=CPU` | the 36 GatedDeltaNet (recurrent) layers |
| `hc_cpu` | `-ot hc_.*=CPU` | hyper-connection tensors (48 layers) |
| `ub128` / `ub1024` | `--ubatch-size` | prefill batch/chunking |
| `baseline_perf` | same flags + `GGML_VK_PERF_LOGGER=1` | per-op GPU-busy time (Vulkan timestamps) |

The per-op dump is used as **supporting evidence, not the sole method**: the
task asked for ablations because the op names in the Vulkan logger do not carry
tensor names, so a MUL_MAT cannot be attributed to attention vs SSM vs shared
expert by the op alone. `bench/parse-vk-perf.py` classifies the rows
(`MUL_MAT_ID` = MoE experts, `FLASH_ATTN_EXT` = full attention,
`GATED_DELTA_NET`/`SSM_*` = recurrent, `MUL_MAT` = dense) and totals them, so
the GPU-busy share and the CPU/sync residual are visible.

## Reproduction

```sh
# all configs, in order, holding the single GPU (BAS-80)
bench/run-warm-prefix-profile.sh

# one config, or a subset
bench/run-warm-prefix-profile.sh baseline
BONGO_PROFILE_CONFIGS=baseline,ncmoe24,fa_off bench/run-warm-prefix-profile.sh

# analysis
bench/analyze-warm-prefix.py --root bench/results/2026-09-28-warm-prefix-profile

# per-op GPU breakdown from the profiler config's server log
bench/parse-vk-perf.py bench/results/2026-09-28-warm-prefix-profile/baseline_perf/llama-server.log --summary
```

Raw files land in `bench/results/2026-09-28-warm-prefix-profile/<config>/`
(`profile.json`, `server-flags.json`, `llama-server.log`, `harness.log`).

## Measured evidence already committed (before this task's run)

Shipped config, n=16, q8 KV, Vulkan, `bench/results/2026-09-28-prefix-cache*/`:

| prefix | delta 512 prefill | ms / new token | full hit (fixed) | cold prefill ms/token |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 5116 ms | 9.95 | 187 ms | 6.05 (cold-page) |
| 16K | 3867 ms | 7.51 | 238 ms | 5.10 |
| 24K | 3988 ms | 7.74 | 238 ms | 5.28 |
| 31K | 4292 ms | 8.33 | 276 ms | 5.67 |
| 128K | **[pending run]** | — | — | 7.50 |

Placement ablation from `bench/results/2026-09-27-expert-placement/` (cold
prefill, so the number is the whole-prompt per-token cost, not the delta):

| `--n-cpu-moe` | 4K cold prompt tok/s | 128K cold prompt tok/s | 128K cold ms/token |
| ---: | ---: | ---: | ---: |
| 12 | 260.4 (128K fails to fit) | — | — |
| 16 (baseline) | 231.5 | 133.4 | 7.50 |
| 24 | 186.5 | 114.5 | 8.74 |

Moving 8 layers' experts from the GPU to the CPU costs **+1.24 ms/token** at
128K cold. That is a direct measurement that the expert matmul path is a
first-order term and that GPU-resident experts are faster than CPU-resident
ones; the placement lever (+18–27% prefill) is the same effect from the other
direction.

Backend A/B and 4K decode from `bench/results/2026-09-28-backend-ab/`:
Vulkan 4K decode ~17.5 tok/s, 128K decode ~8.0 tok/s; SYCL 4K decode ~6.4
tok/s. The 4K **>=25 tok/s** target is not met.

## Preliminary reading (to be confirmed by the run)

1. **The cached-KV attention term is not dominant at <=31K.** The delta-turn
   cost is 7.5–8.3 ms per *new* token and grows only ~11% when the prefix
   doubles from 16K to 31K. If attention over the cached KV dominated, the cost
   would track the prefix. So the turn is dominated by the compute of the 512
   new tokens, not by reading the cache.
2. **The dominant compute term is the MoE expert prefill matmul.** The expert
   tensors are ~31 GiB of the model's ~35 GiB; the placement ablation shows
   +1.24 ms/token per 8 layers moved to the CPU. M3.1b
   ([`iq2xs-sycl-integer-mmvq.md`](iq2xs-sycl-integer-mmvq.md)) showed the
   prefill path expands the 2-bit i-quant experts to FP16 and runs a oneDNN
   dequant GEMM (~2.5 TFLOPS), and that a chunked MMVQ loop is *slower*.
3. **The decision implication** is therefore that the lever that moves the
   cached-turn target is a **true tiled integer MMQ** (no FP16 expansion) — not
   the single-column MMVQ that M3.1 already measured at ~1.05x, not the PLE
   reader (measured neutral, [ADR-0004](0004-ple-reader-disposition.md)), and
   not speculation (its own A/B, [BAS-75](/BAS/issues/BAS-75)). Placement is
   worth the +18–27% already measured but is VRAM-capped.

These are hypotheses from existing evidence. The run below is what turns them
into a ranked, measured cost table and a proved dominant term.

## Pending from this task's run

- `[pending]` ranked cost table (ms and %) for one 512-token delta turn at 16K
  and at 128K, with engine revision, tier, flags and raw files.
- `[pending]` the ablation that proves the dominant term.
- `[pending]` the per-op GPU-busy breakdown from `baseline_perf`.
- `[pending]` the decision note with the measured effect size of each planned
  lever (kernel / speculation / placement / PLE).
