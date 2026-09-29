# Engine gap analysis — where bongo's 8 tok/s goes, and what closes it

Synthesis of the R1–R7 research spawned by [BAS-62](/BAS/issues/BAS-62). Feeds
[ADR-0003](../adr/0003-engine-direction.md). Written 2026-09-28 by the CTO.

- Baseline: Arc Pro B70, IQ2_XS, llama.cpp `b11223`, **Vulkan** (`--n-cpu-moe 16`), warm:
  128K prompt **133.2 tok/s**, 128K output **8.0 tok/s**, 128K TTFT **983 s**; 4K output 17.7 tok/s.
- Reference: Strata on an RTX 5070 12 GB + 64 GB RAM: **52–65 tok/s at 128K**.

Sources: [strata-architecture](strata-architecture.md) (R1), [ninfer-architecture](ninfer-architecture.md) (R2),
[moe-offload-landscape](moe-offload-landscape.md) (R3), [expert-activation-skew](expert-activation-skew.md) (R4),
[ssd-ngram-shard](ssd-ngram-shard.md) (R5), [sycl-kernel-feasibility](sycl-kernel-feasibility.md) (R6),
[capacity-sensitivity](capacity-sensitivity.md) (R7), [levelzero-submission](levelzero-submission.md)
(R1b/R1c). Raw data under `bench/results/`.

## 1. The headline

**The gap is engine quality, not memory capacity and not a missing op.** Three independent measurements agree:

- **Placement is not the 128K decode lever.** R4: a frequency-ranked hot set covers **98.5%** of expert
  activations at the shipped 22.40 GiB VRAM budget versus 66.0% for the `--n-cpu-moe` layer rule — and 128K
  decode moves **~0%**. R7: the marginal return on GPU-resident experts at 128K above ~17 GiB is
  **+0.016 output tok/s per GiB**. R3 and the Stage-1 sweep agree.
- **RAM is not the lever either.** R7: a reversible 16 GiB cgroup cap matches the uncapped ~30 GiB box within
  1.5% at the shipped split; modelled **32 → 64 GiB RAM = 0 tok/s** at 32 GiB VRAM. RAM only binds on a
  ~12 GiB-VRAM box.
- **What is left is the per-token compute path.** R6: on IQ2_XS both backends **dequantise in prefill** —
  SYCL's integer MMQ is globally disabled in `b11223` (`ggml_sycl_supports_mmq()` → false) and Vulkan has no
  integer MMQ kernel for IQ2_XS. That is exactly the mechanism R1 attributes Strata's speed to.

So bongo's 32 GiB VRAM (vs Strata's 12 GiB) is a real advantage **only on the CPU-expert path** (prefill and
short-context decode). It is largely spent on safety margin and KV headroom at 128K. The path to Strata-class
numbers is the compute path, not a bigger memory budget.

## 2. Mechanism-by-mechanism

Expected effect is the research's own measured or first-order estimate, not a promise.

| # | Mechanism | Evidence | Expected effect on bongo | Cost / portability | Verdict |
|---|---|---|---|---|---|
| 1 | **Quantized-weight MoE/dense kernels (MMQ/MMVQ, weights never dequantised)** | R1 §3 (Strata prefill 1052→1130 on these kernels); R6 §2 (MMQ disabled on SYCL, absent on Vulkan for IQ2_XS) | Single largest term; dequant→FP16 is paid every token | Patch `ggml-sycl` (the decode dp4a MoE path already exists: `mul_mat_vec_q_moe`; enable `ggml_sycl_supports_mmq`); verify numerics | **Build first** |
| 2 | **Speculation: suffix / n-gram drafter + exact verify window** | R1 §4 (1.6–1.8x, drafts are weightless); R3 §6 (third party 43→62 tok/s on this model); MTP unavailable | 1.4–1.8x decode where acceptance holds; near-free on CPU | llama.cpp spec path; adopt Strata's learning policy, not a fixed window | **Build second** |
| 3 | **Backend selection (SYCL vs Vulkan)** | R6 §4/§7: SYCL prefill ~1.4x, decode ~1.6x **slower** on a contended box | Decides the default; may be a per-tier split | Run the warm 4K/128K A/B already specified; no code | **Measure first** |
| 4 | **Placement: byte-budget `-ot`, then dynamic VRAM LRU** | R4 (profile vs layer rule); R3 §6 (online LRU 67–81% vs static ~10% out-of-sample on another model) | +18–27% 128K prefill, +25–37% 4K decode, ~0% 128K decode | Cheapest-layer-first `-ot` is config-only; per-expert layout/LRU is a llama.cpp patch | **Build third** |
| 5 | **PLE/n-gram second shard reader** | R5: 26.82 GiB stays on SSD; 16 parallel 4 KiB reads = **0.26 ms/token** vs a 125 ms token | Keeps the table off the RAM budget; **+2x prefill** with direct reads vs mmap faults (R3) | Direct reads + bounded row cache + prefetch; keep `--lazy-mode` fallback | **Adopt** |
| 6 | **Graph / command-list submission** | R1b/R1c: Level Zero submission = 1.34 µs; 96/token = 0.13 ms. Strata's whole-token graph **cannot port** (host→device doorbell never arrives) | Small here: one captured command list/token saves ~1.76 ms/token at 2,064 nodes | Patch, but only after 1–4 | **Defer** |
| 7 | **SSD-resident expert weights** | R3: best measured = 2.2 tok/s at 79% hit | Negative (−3.5x vs today) | — | **Reject** |
| 8 | **From-scratch Intel-native (SYCL) engine** | R6 §7: no op-coverage gap; decode slower than Vulkan; device-loss hazards | High cost, unproven win | — | **Reject for now** |
| 9 | **More system RAM (32 → 64 GiB)** | R7 §5 | 0 tok/s at 32 GiB VRAM | — | **Reject** |
| 10 | **NInfer design shape** | R2 §7 | Activation-declaration policy, epilogue/projection fusion, KV/state tier, head-agnostic verify core | Host-side/engine design; no CUDA port needed | **Adopt as design input** |

## 3. The target, reframed

The original brief implied "Strata-class 50 tok/s at 128K". The research splits that into three different
problems with different levers:

| Metric | Today | Levers that move it | Levers that do **not** |
|---|---|---|---|
| **128K TTFT (prefill)** — 983 s, the acute product pain | 133 tok/s | MMQ kernels, backend (SYCL prefill +1.4x), PLE direct reads (+2x prefill), placement (+18–27%) | expert profile, RAM size, VRAM size |
| **4K decode** | 17.7 tok/s | MMQ decode (dp4a), speculation, placement (+25–37%) | RAM size |
| **128K decode** | 8.0 tok/s | MMQ kernels and speculation (not yet measured on Arc); long-context attention/KV work | expert placement/profile (~0), RAM size (~0), VRAM size above ~17 GiB |

**This forces a product-priority decision** (see the plan / confirmation): which metric is the target? The
mechanisms bill differently. The engine direction in ADR-0003 attacks TTFT and 4K decode first because those
are where the measured levers are; 128K decode needs the kernel+spec measurement before any commitment.

## 4. The one evidence conflict to resolve locally

R4 (bongo's own model/workload, leave-one-out per domain) finds an **offline frequency profile generalises
well**: held-out coverage 0.881–0.985, and a prefill-built profile serves 97.2% of decode selections.
R3 cites llama.cpp PR #27861 on a different model, where a static top-32 ranking recovers ~10% out-of-sample
(uniform 6.2%) while an **online LRU** recovers 67–81%.

These are not directly contradictory (different model, workload, and definition of "profile"), and R4 is the
stronger evidence for bongo. But the conflict is exactly the kind that kills a residency feature in production.
**Resolution rule for the build:** ship the profile as the *initialisation* for a dynamic VRAM LRU, never as a
frozen policy, and require a local A/B on held-out prompts before relying on it.

## 5. What this means for the task graph

`docs/roadmap.md` M2 concluded "no-go on an adaptive VRAM expert cache" from 128K decode insensitivity. That
conclusion was directionally right for 128K decode but wrong as a general verdict: the same profile is worth
+18–27% prefill and +25–37% 4K decode, and the M2 gate did not test the kernel or speculation levers at all.
M3's gate ("llama.cpp cannot reach the target") is now **open**, because this document is the missing gap
analysis and it points at engine patches, not a new engine. See ADR-0003 for the decision.
