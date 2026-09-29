# ADR-0006 — BAS-76 Step 2 engine expert-LRU is stopped; Step-1 `-ot` placement is the placement result

- Status: **Accepted** (2026-09-29; CTO decision, on [BAS-147](/BAS/issues/BAS-147))
- Date: 2026-09-29
- Deciders: CTO
- Related: [BAS-139](/BAS/issues/BAS-139) (the engine build), [BAS-76](/BAS/issues/BAS-76) (M3.3 placement),
  [BAS-144](/BAS/issues/BAS-144) (M4.1 host/CPU critical path), [ADR-0005](0005-host-cpu-critical-path.md)
  (host/CPU pivot), [ADR-0003](0003-engine-direction.md) (engine direction),
  [`docs/research/dynamic-expert-lru.md`](../research/dynamic-expert-lru.md),
  [`bench/results/2026-09-29-moe-cache-lru/`](../../bench/results/2026-09-29-moe-cache-lru/moe-cache-ab.md)

## Context

[BAS-76](/BAS/issues/BAS-76) Step 2 was the dynamic VRAM LRU over RAM-pinned MoE experts: keep per-expert
residency over the host-pinned `ffn_{gate,up,down}_exps` weights, initialise the cache from the offline
activation profile, and let a global LRU track drift. [BAS-139](/BAS/issues/BAS-139) built it as a
llama.cpp `qwen4exp` patch (`tools/patches/moe-expert-cache.patch`, engine `b11223`) and ran the engine A/B
at commit `97d678a` (2026-09-29, one session, one shared Arc flock).

A = engine LRU (`--n-cpu-moe 48` + `--moe-expert-cache-profile profile-iq2_xs-22.40.txt`); B = pinned Stage 0
(`--n-cpu-moe 16`). Tier `iq2_xs`, q8 KV, Vulkan.

| metric | A (engine LRU) | B (Stage 0 n=16) | Δ |
| --- | ---: | ---: | ---: |
| 4K decode tok/s | 2.503 | 17.112 | **-85.4%** |
| 128K decode tok/s | 3.041 | 8.044 | **-62.2%** |
| 128K prefill tok/s | 66.771 | 131.816 | -49.4% |
| turn TTFT 4K cached +512 ms | 43 418 | 2 979 | +1 358% |
| needle | pass | pass | — |

The cache **engaged and is correct**: 47 cached layers, hit rate 0.952 (262 723 hits / 13 167 misses), and
30.92 GiB VRAM after load against 6.33 GiB for a plain `--n-cpu-moe 48`, so ~24.6 GiB of cache tensors were
resident. It does not buy throughput. Against the recorded all-experts-on-CPU control (`--n-cpu-moe 48`, no
cache) the cache is flat at 4K (2.503 vs 2.948 tok/s) and ~+34% at 128K (3.041 vs 2.262 tok/s) — against
+480% / +256% for whole-layer residency. Verdict `moe-cache-ab.md`: **FAIL** on every acceptance target.

ADR-0005 ([BAS-62](/BAS/issues/BAS-62)) already measured the MoE expert matmul at **3.4%** of the warm-prefix
512-token delta turn and named the **host/CPU critical path** as the next milestone. The Step-2 design keeps a
CPU `mul_mat_id` **per MoE layer** (it skips cached experts but still crosses the CPU/GPU boundary), so it
*adds* host/CPU work instead of removing the term it was meant to remove. The cache is also **decode-only**
(`n_tokens == 1`), so the primary turn-TTFT target — the 512-token delta turn, a prefill batch — gets nothing
from it, while prefill pays because every expert is host-resident.

## Decision

1. **Stop BAS-76 Step 2 as a throughput lever.** Do not fund the `n=48 ± cache` diagnostic or any further
   LRU optimisation on [BAS-139](/BAS/issues/BAS-139). The advertised failure margin (-85% / -62%) is not a
   config tweak away, and the path cannot serve the primary turn metric.
2. **Ship BAS-76 Step 1 (byte-budget `-ot`) as the placement result.** +2.91% 128K prefill / +0.74% 128K
   decode, +4.25 pp activation coverage at the same 22.40 GiB budget, 0.22 GiB less peak VRAM, no regression,
   both needles pass. This is the shipped *placement* outcome; it does not change `bongo.sh` defaults.
3. **Record the negative engine result** in
   [`docs/research/dynamic-expert-lru.md`](../research/dynamic-expert-lru.md) (owner: [BAS-139](/BAS/issues/BAS-139),
   engineer). The patch stays on `BAS-62-improve-speed-architecture`, **opt-in and inert by default**.
4. **Fold the open diagnostic into M4.1.** The unresolved sub-question — which of the per-layer CPU handoff,
   the quantised cache-tensor `mul_mat_id`, or copy/scheduler overhead dominates — is a subset of
   [BAS-144](/BAS/issues/BAS-144) method step 1 (decompose the host/CPU term into CPU-resident expert FFN,
   host graph build/dispatch, GPU submit/sync). No second GPU campaign for the LRU.

## Alternatives considered

- **(1) Diagnose and optimise the cache.** Rejected. It would spend 1–2 GPU sessions on a **decode-only**
  path that cannot move the turn-TTFT target, to answer a question already funded in
  [BAS-144](/BAS/issues/BAS-144). Removing the per-layer CPU chain for resident-heavy layers is new engine
  work, not a cache fix, and the measured 3.4% expert-matmul share caps its expected value.
- **(2) Re-scope the cache as a small supplement to Step-1 residency.** Rejected. VRAM after load is already
  30.92 GiB against a ~31.85 GiB device-loss point, leaving <1 GiB for a cache — too small to cover a
  meaningful expert set — and the host-resident experts would still slow prefill.
- **(3) Stop Step 2 and keep Step 1.** **Chosen.** Cheapest, matches the ADR-0005 measurement, and keeps the
  whole-layer `-ot` win.

## Consequences

- **Positive.** No further GPU budget on a lever the measurement says cannot pay; the Step-1 placement win is
  retained; M4.1 keeps a single, non-duplicated host/CPU diagnostic thread.
- **Negative / risk.** If per-expert residency later proves the right lever for the host/CPU path, the work
  restarts from the preserved patch and the recorded negative result — it is a new engine design, not a flag.
  The Step-2 negative result is evidence for the M4.1 direction, not a dead end to forget.
- **Batch/needle.** Both A and B passed the needle; the cache changed *where* the matmul ran, not numerics.

## Rollback path

The patch is additive and flag-gated. Re-enabling `--moe-expert-cache-profile` (or reverting this ADR)
restores the experimental path without a rebuild beyond the patched engine pin. No production defaults,
infrastructure, DNS, billing, secrets, or data are touched.
