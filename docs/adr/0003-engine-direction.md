# ADR-0003 — engine direction after the R1–R7 research: patch llama.cpp, do not build a new engine

- Status: **Accepted** (2026-09-28; approved by the CEO on [BAS-62](/BAS/issues/BAS-62); target corrected to 256K)
- Date: 2026-09-28
- Deciders: CTO (author); CEO (direction confirmation)
- Supersedes: the Stage 2 gate in [ADR-0001](0001-runtime-architecture.md) ("custom SYCL engine, gated")
- Amends: [ADR-0001](0001-runtime-architecture.md) Stage 1 (adaptive expert cache) and
  [ADR-0002](0002-baseline-engine.md) (SYCL default — now conditional on the measured backend A/B)
- Related: [engine gap analysis](../research/engine-gap-analysis.md), [R1 Strata](../research/strata-architecture.md),
  [R2 NInfer](../research/ninfer-architecture.md), [R3 landscape](../research/moe-offload-landscape.md),
  [R4 skew](../research/expert-activation-skew.md), [R5 PLE](../research/ssd-ngram-shard.md),
  [R6 SYCL](../research/sycl-kernel-feasibility.md), [R7 capacity](../research/capacity-sensitivity.md),
  [R1b/R1c Level Zero](../research/levelzero-submission.md)

## Context

bongo's goal is Strata-class throughput on one Arc Pro B70 (32 GiB VRAM, ~30 GiB RAM) for one model
(Qwen3.8-Flash-Next / Swift 1.5, `qwen4exp`). The Stage 0 baseline is **8.0 tok/s at 128K** (Vulkan,
`--n-cpu-moe 16`). Strata gets **52–65 tok/s at 128K** on an RTX 5070 12 GB + 64 GB RAM.

Seven research tasks (R1–R7) plus two follow-ups (R1b/R1c) measured the reference engines and bongo's own box.
The findings that decide this ADR:

1. **No op-coverage gap exists.** Every op in the `qwen4exp` graph is implemented in both the SYCL and Vulkan
   backends. The gap is kernel maturity and scheduling, not missing kernels (R6).
2. **The kernel gap is dequantisation.** On IQ2_XS both backends dequantise in prefill; SYCL's integer MMQ is
   globally disabled in the pinned `b11223`, and Vulkan has no integer MMQ for IQ2_XS. This is the mechanism
   Strata credits for its speed (R1, R6).
3. **Memory capacity is not the bottleneck.** Expert placement/profile is worth ~0 on 128K decode (R4, R7,
   Stage 1); 32→64 GiB RAM is worth 0 tok/s (R7); SSD expert streaming is a 3.5x regression (R3). The 32 GiB
   VRAM is leveraged up to ~22.4 GiB of experts, then limited by 128K attention/KV.
4. **The port of Strata's graph mechanism is not justified here.** A Level Zero submission costs 1.34 µs
   (96/token = 0.13 ms), and the whole-token graph's host→device doorbell never arrives on this stack
   (R1b/R1c). The CUDA graph workaround targets a Windows/WDDM cost that does not exist on Fedora + Level Zero.
5. **The cheap multipliers are portable.** Suffix/n-gram speculation is weightless and already proven on this
   model by third parties; PLE-on-SSD is near-free when read in parallel (0.26 ms/token); placement is worth
   +18–27% prefill and +25–37% 4K decode (R1, R3, R4, R5).

## Decision

**Do not build a from-scratch engine. Build bongo's optimisation layer as a small, pinned set of patches to
llama.cpp's SYCL backend plus a scheduler/config layer, and sequence the work by measured expected value:**

1. **Measure first (M3.0): the warm SYCL-vs-Vulkan A/B** at 4K and 128K with the shipped flags, three repeats.
   It decides the default backend and the tier policy. R6 gives the decision rule: SYCL default only if it
   holds ≥1.3x on 128K TTFT and stays within ~10% on 128K decode.
2. **Quantized-weight kernels (M3.1):** enable/implement the integer MMQ/MMVQ path for IQ2_XS on SYCL so
   weights are never expanded to FP16. This is the largest single term and the reason a patch is justified.
3. **Speculation (M3.2):** the suffix/n-gram drafter with the exact verify/commit window and Strata's online
   acceptance policy. MTP stays unavailable; this is the only speculation path.
4. **Placement (M3.3):** replace the layer-count rule with a cheapest-layer-first byte-budget `-ot` placement
   (config-only), then a **dynamic VRAM LRU over RAM-pinned experts**, with the R4 profile as *initialisation*
   and a held-out A/B before it is relied on.
5. **PLE reader (M3.4):** direct-file reads + bounded row cache + parallel prefetch for the 26.82 GiB second
   shard; keep mmap + `--lazy-mode` as fallback.
6. **Revisit Stage 2 (a custom engine) only if**, after M3.1–M3.3 are measured, the results miss the target and
   a new ADR names the specific unfixable gap. Command-list capture (R1c) is deferred until then.

Upstreaming generic patches is preferred; bongo keeps a pinned fork and applies local patches where the change
is bongo-specific (per-expert layout, residency policy, PLE reader).

## Amendment (2026-09-28) — workload profile from the plan review

The CEO rejected the first plan revision with the workload definition that the milestones must serve:
**agentic coding** — a small first prompt that grows over a session; **context target 256K** (the model's
262144 native limit); **quantized KV**; and a request for a projected tokens-per-second and
prompt-processing envelope before the build starts. The rejection is recorded on
[BAS-62](/BAS/issues/BAS-62); the measurement is in
[`docs/research/agentic-prefix-cache.md`](../research/agentic-prefix-cache.md).

This changes the target, not the direction:

- **Primary metric is now per-turn TTFT under prefix reuse**, not the cold 128K prefill. Measured: a 512-token
  continuation over a 4K–31K cached prefix costs **3.9–5.1 s**, a full hit **0.19–0.30 s**, while a cold
  re-prefill of 31K costs 180 s and 128K costs 983 s. The harness had sent `cache_prompt: false`, so every
  earlier number was cold; the cached path is the product path. The long-context points were then measured
  ([BAS-132](/BAS/issues/BAS-132), `bench/results/2026-09-28-prefix-cache-longctx/`): the **512-token delta
  turn costs 12.31 s at 128K cached (42.2 delta tok/s) and 10.87 s at 256K cached (48.2 delta tok/s)**, with
  full hits at 0.53/0.66 s. **The `<=5 s` delta-turn target is not met at either long context**; the full
  prefix is reused, so the whole cost is the delta prefill, and the rate falls from 96–133 tok/s at 4K–31K to
  42–48 tok/s at 128K–256K. Closing that gap is the M3.1 (integer MMQ) and M3.2 (speculation) work, not a
  cache-reuse fix.
- **Prefix reuse and KV persistence become first-class engine requirements** (new milestone M3.0a below):
  `--cache-prompt` (on by default), `--slot-save-path` + slot save/restore for cross-idle/cross-restart
  sessions, `--cache-idle-slots`, and a harness mode that measures the cached path.
- **Context target: 256K (measured, not assumed).** Fitting was measured directly (plan-review correction:
  the target is 256K, not 156K): **q8 KV at the shipped `--n-cpu-moe 16` reaches 31.79–31.82 GiB, within
  ~0.03 GiB of the 31.85 GiB device-loss point — unsafe**; **q8 KV at `n=18` sits at 30.35 GiB (safe)** and
  **q4 KV at `n=16` sits at 30.10 GiB (safe)**. **Shipped 256K default: q8 KV with `--n-cpu-moe 18`**; the
  `--ctx 262144 --n-cpu-moe 18` recipe is recorded in
  [`docs/bongo-sh.md`](../bongo-sh.md#long-context-256k--the-shipped-default). It was exercised end-to-end: a
  256K cold prefill at 84.8 prompt tok/s with a 128K needle pass and a 30.92 GiB peak
  (`bench/results/2026-09-28-ctx256-full/`), and the cached delta turn at 10.87 s / 30.86 GiB
  (`bench/results/2026-09-28-prefix-cache-longctx/`). q4 KV at `n=16` is the documented alternative when
  expert residency is worth more than KV precision. The Stage 0 Vulkan baseline (`--ctx 131072
  --n-cpu-moe 16`) stays pinned and selectable. Milestone M3.5.
- **The milestones are re-ordered** so the cheap, measured serving win ships before the kernel work:
  M3.0 backend A/B → **M3.0a prefix-cache serving + persistence** → M3.1 integer MMQ/MMVQ → M3.2 speculation
  → M3.3 placement → M3.4 PLE reader → M3.5 long-context/KV budget.

## Alternatives considered

- **(a) From-scratch bongo-owned SYCL engine (the old Stage 2).** Rejected for now. R6 shows no coverage gap to
  justify it, a measured SYCL decode deficit, and device-loss hazards in the Level Zero path (kernel param
  faults, IGC abort on `OpMemoryBarrier`); R1 shows the ported CUDA kernels are heavily shuffle/`dp4a`/`mma`
  specific. Highest cost, unproven win, and it blocks the one-command deliverable.
- **(b) Bongo-owned patches on a pinned llama.cpp SYCL fork + scheduler layer.** **Chosen.** Smallest diff that
  reaches the identified mechanisms; every change is behind a flag or a pin, so it is reversible.
- **(c) Config-only on stock llama.cpp (`-ot`, `--n-cpu-moe`, KV quant).** Rejected as the ceiling: it cannot
  express per-expert residency, cannot enable the disabled integer MMQ, and cannot add the PLE reader or the
  speculation policy. It remains the fallback and the baseline.
- **(d) SSD-resident expert weights.** Rejected: 2.2 tok/s at a 79% hit rate, a regression (R3).
- **(e) Buy more RAM / a bigger GPU.** Rejected on measurement: 32→64 GiB RAM = 0 tok/s at 32 GiB VRAM (R7).
- **(f) Port Strata's per-layer CUDA graphs.** Rejected: the motive (WDDM launch overhead) does not exist on
  Linux + Level Zero, and the device-wait cannot be driven from the host on this stack (R1b/R1c).

## Consequences

- **Positive.** The work is bounded and each milestone has a measured gate; the baseline stays shippable
  throughout; the largest term (MMQ) is a known, already-partially-implemented backend path.
- **Negative / risks.** (i) A pinned fork carries rebase cost against upstream llama.cpp; mitigate by keeping
  patches small and upstream-first. (ii) The MMQ numerics and the IQ2_XS dp4a correctness must be verified
  against an FP16 oracle before it becomes the default. (iii) The R4-vs-R3 profile conflict must be resolved by
  a local A/B (see gap analysis §4). (iv) The SYCL decode deficit may make the "SYCL default" choice metric
  dependent, so the A/B is a hard prerequisite, not a formality.
- **Product dependency.** The milestones optimise TTFT and 4K decode; 128K decode needs the kernel+spec
  measurement first. The CEO must confirm the target metric (see the plan).

## Rollback path

Every step is additive and pin-reversible:

- The Stage 0 **Vulkan** configuration stays pinned and selectable in `bongo.sh`; reverting the engine pin and
  flags restores the current working baseline (8 tok/s at 128K) with no data migration.
- Each patch lands behind a flag or a checked-out engine revision, and the benchmark harness compares against
  the pinned baseline before the flag becomes the default.
- No production infrastructure, DNS, billing, secrets, or schema is touched by any milestone.
