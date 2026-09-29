# ADR-0005 — engine direction, revision 2: the cached-turn bottleneck is host/CPU, not the expert kernel

- Status: **Accepted** (2026-09-29; CTO decision, within the CEO-accepted [ADR-0003](0003-engine-direction.md)
  direction of "patch llama.cpp, do not build a new engine")
- Date: 2026-09-29
- Deciders: CTO
- Supersedes: the "kernel maturity is the largest term" premise of [ADR-0003](0003-engine-direction.md),
  section 2 and its M3.1 expected value. The patch-based approach and every other ADR-0003 decision stand.
- Related: [ADR-0003](0003-engine-direction.md) engine direction, [ADR-0004](0004-ple-reader-disposition.md)
  PLE reader, [warm-prefix profile](../research/warm-prefix-profile.md), [dynamic expert LRU](../research/dynamic-expert-lru.md),
  [M4.1 host/CPU decomposition](../research/host-cpu-decomposition.md)

## Context

ADR-0003 funded the integer MMQ/MMVQ path because "the kernel gap is dequantisation" and that was believed to
be the largest single term for the shipped agentic workload. The M3 build measured that belief, rather than
assuming it, and it did not hold for the turn metric the targets are set on.

Measured on the pinned engine `b11223` (`4da633776`), Vulkan, IQ2_XS, q8 KV:

| result | measurement | source |
| --- | --- | --- |
| M3.1 integer MMVQ (IQ2_S/XXS/IM + Q2_0) | **~1.05x** on the cached turn, not >=1.3x; correct and flag-gated | [BAS-74](/BAS/issues/BAS-74) |
| M3.6 warm-prefix profile, 512-token delta turn at 16K | **80.1% host/CPU + sync**; GPU flash-attn 8.9%, dense 5.2%, **MoE expert matmul 3.4%** | [BAS-130](/BAS/issues/BAS-130) |
| M3.6 proving ablation | `--ubatch-size 128` costs **+129%** (~1.7 s fixed per batch); `--n-cpu-moe 24` costs +18.3% | [BAS-130](/BAS/issues/BAS-130) |
| M3.5b cached delta turn | **12.31 s at 128K** and **10.87 s at 256K** (target <=5 s); <=5 s missed at both | [BAS-132](/BAS/issues/BAS-132) |
| M3.2b n-gram speculation | **negative** on Vulkan (0.92-1.22x); decode-side, does not move prefill TTFT | [BAS-131](/BAS/issues/BAS-131) |
| M3.3 Step 1 byte-budget `-ot` | **+2.91%** 128K prefill / +0.74% decode (coverage +4.25 pp) | [BAS-76](/BAS/issues/BAS-76) |
| M3.4 PLE reader | neutral engine A/B; kept opt-in | [ADR-0004](0004-ple-reader-disposition.md) |

The expert matmul is **3.4% of the turn**. A perfect expert kernel cannot close a 33% miss at 16K or an 18%
miss at 128K. Every planned GPU-side lever is now either measured small (kernel, placement Step 1), decode-only
(speculation), or neutral (PLE).

## Decision

1. **Close the GPU expert-kernel line for the turn metric.** No further MMQ/MMVQ work is funded for the
   cached-turn TTFT target. The landed patch and its `GGML_SYCL_IQUANT_MMVQ_MAX` flag stay shipped and
   selectable; they are not the plan.
2. **Make the fixed per-batch host/CPU critical path the next milestone (M4.1).** The measured ~1.7 s fixed
   cost per batch, the +129% cost of smaller batches, and the +18.3% cost of CPU-resident expert compute name
   the target: fewer CPU/GPU segment transitions, overlap of the CPU-resident expert FFN with GPU work, and
   larger effective resident expert sets. [BAS-139](/BAS/issues/BAS-139) (dynamic VRAM LRU) is the placement
   half of this and stays in flight. **Measured (M4.1):** the CPU backend does no work in this
   configuration (0 ms of worker CPU); the fixed cost is the main-thread host path plus an in-turn page-cache
   re-read of the mmap'd host-resident expert weights, and `--load-mode none` removes the re-read
   ([BAS-144](/BAS/issues/BAS-144)).
3. **Keep the shipped agentic product path as it is.** Prefix reuse, the 256K default, and the checkpoint
   sidecar are real wins and are unaffected:
   - 256K cold prefill: 84.8 prompt tok/s, 128K needle PASS, peak VRAM 30.92 GiB
     (`bench/results/2026-09-28-ctx256-full/`).
   - A restored slot is now **reused**: `cache_n = 31742`, TTFT 557 ms vs a 179.7 s re-prefill
     ([BAS-86](/BAS/issues/BAS-86)).
4. **Record the targets as missed, not met.** At 16K the 512-token turn is 3.98 s (target <=3 s); at 128K it is
   12.31 s (target <=5 s); 4K decode is ~14-18 tok/s (target >=25). The umbrella [BAS-62](/BAS/issues/BAS-62)
   stays open until the turn and decode targets are met or the CEO changes them.

## Alternatives considered

- **(a) More GPU kernel work (tiled integer MMQ, Vulkan MMQ).** Rejected. The term it optimises is 3.4% of the
  measured turn; the M3.1 result already caps the achievable gain near 5%.
- **(b) Buy more VRAM / a bigger GPU, or more RAM.** Already rejected on measurement in ADR-0003 (R7): 32->64
  GiB RAM = 0 tok/s; VRAM stops paying above ~22.4 GiB of GPU experts.
- **(c) Accept the current numbers and close BAS-62.** Rejected. It ships a working agentic product but leaves
  the agreed turn and decode targets unmet, with the dominant term now measured and addressable.
- **(d) Build a from-scratch engine.** Still rejected, for the ADR-0003 reasons (no op-coverage gap, measured
  SYCL deficit, device-loss hazards). The bottleneck is host scheduling, which a rewrite does not by itself fix.

## Consequences

- **Positive.** The next milestone targets a term that is 80% of the turn instead of 3.4%. The plan no longer
  spends on a lever the measurement says cannot pay. Every shipped win stays.
- **Negative / risk.** The host/CPU path is llama.cpp scheduler and server work, not a contained kernel; the
  diff may be larger and less upstream-friendly than a kernel patch. Mitigation: keep M4.1 behind a flag, keep
  the Stage 0 Vulkan baseline pinned, and measure before/after with the committed `warm-prefix-profile` tools.
- **Target honesty.** The `<=3 s` / `<=5 s` / `>=25 tok/s` targets are not met. They are recorded as misses in
  [ADR-0003](0003-engine-direction.md) and [docs/bongo-sh.md](../bongo-sh.md), and on [BAS-132](/BAS/issues/BAS-132).

## Rollback path

All M4.1 work is additive and flag-gated. Reverting the flag or the engine pin restores the current shipped
behaviour. The Stage 0 Vulkan `--ctx 131072 --n-cpu-moe 16` baseline and the 256K default remain selectable. No
production infrastructure, DNS, billing, secrets, or data migration is touched.
