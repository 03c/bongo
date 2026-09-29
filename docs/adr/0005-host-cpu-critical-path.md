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

## Measured refinements after M4.1 (2026-09-29)

M4.1 ([BAS-144](/BAS/issues/BAS-144)) decomposed the host term and measured one lever, which corrects the
mechanism guess in decision 2:

- **The term is the per-batch host->VRAM upload of host-resident MoE expert weights, on the main thread.**
  At 16K the main host thread is **2180 ms (38.4%)** of the turn, the CPU backend worker threads are **0 ms**,
  and ~0.5 GiB of in-turn page-cache re-reads ride along because the host weights are `mmap`-backed. The
  earlier reading of the `--n-cpu-moe` ablation as "CPU-resident expert FFN compute" was wrong: the weights are
  host-resident, but the scheduler's `MUL_MAT_ID` "copy only used experts" path uploads them and the GPU
  computes. `--no-op-offload` (force the CPU path) is **+62%**, which proves the GPU upload+compute path is the
  cheaper one and the upload is the term to attack.
- **The available lever buys ~10%, not 15%.** `--load-mode none` (anonymous RAM instead of `mmap`) removes the
  in-turn re-read: **16K 3569.4 ms (-10.4% vs the frozen M3.6 baseline), 128K 5480.6 ms (-7.1%)**, storage read
  0.52 -> 0.00 GiB. It misses the 15% gate but is a real, opt-in, revertible win.
- **The placement half failed.** The proposed dynamic VRAM LRU ([BAS-139](/BAS/issues/BAS-139)) measured
  **-49% to -85%** against the Stage 0 baseline; it is stopped and its patch is inert by default
  ([ADR-0006](0006-moe-expert-lru-disposition.md)). Placement is at its VRAM edge: `--n-cpu-moe 8` does not
  load at 128K.
- **Residual.** After `--load-mode none`, the remaining fixed cost is the upload itself (main host 1970 ms at
  16K, 2940 ms at 128K). Nothing available as a server flag removes it. The next step is to overlap the upload
  with GPU compute or to keep uploaded experts resident across batches (M4.2).

### M4.2 result (2026-09-29) — the turn targets are met

M4.2 ([BAS-155](/BAS/issues/BAS-155)) located the upload and fixed its root cause:

- **Call:** `ggml_backend_sched_compute_splits` -> `ggml_backend_tensor_set_async(...)`, the "copy only used
  experts" branch (`ggml/src/ggml-backend.cpp`); **7.35 GB copied over 5721 calls** for the 16K turn.
- **Root cause:** `ggml_backend_vk_host_buffer_type()` hard-coded `vk_instance.devices[0]`. On this box
  Vulkan0 is the AMD iGPU and Vulkan1 is the Arc, so host-resident experts were pinned on the wrong device and
  every copy fell back to a CPU staging memcpy plus a per-copy `ggml_vk_synchronize`.
  `GGML_VK_HOST_BUFT_PER_DEVICE=1` puts the pinned weights on the compute device: the main-thread branch drops
  **1136 -> 45 ms (24x)**. `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` moves the copies to the transfer queue.
- **Measured with both levers on** (`--load-mode none`): **16K 3569.4 -> 2721.8 ms (-23.7%)** and
  **128K 5480.6 -> 4661.0 ms (-14.95%)**. Both are inside the CEO target (`<=3 s` / `<=5 s`). Cold prefill
  falls -29.9% / -21.5%. Needle passes.
- **Decode is unchanged** (17.6 -> 18.1 tok/s, within noise): it is GPU MoE-matmul-bound, not upload-bound, so
  the `>=25 tok/s` target is still open (M4.4).
- Both levers are **environment-gated and default off**; the patch is
  `tools/patches/m4.2-vulkan-host-expert-upload.patch`. Shipping them as the default is M4.3.

**Updated target status:** `<=3 s` at 16K and `<=5 s` at 128K are **met** on the measured configuration; 256K is
met; **decode `>=25 tok/s` is the only target still open** and is a separate, GPU-matmul-bound axis.

## Consequences

- **Positive.** The next milestone targets a term that is 80% of the turn instead of 3.4%. The plan no longer
  spends on a lever the measurement says cannot pay. Every shipped win stays.
- **Negative / risk.** The host/CPU path is llama.cpp scheduler and server work, not a contained kernel; the
  diff may be larger and less upstream-friendly than a kernel patch. Mitigation: keep the lever behind a flag,
  keep the Stage 0 Vulkan baseline pinned, and measure before/after with the committed profiling tools. The
  measured path to the targets is now narrow: `--load-mode none` gives ~10%, and only the upload term is large
  enough to close the rest (M4.2).
- **Target honesty.** After M4.2 the turn targets are **met** (16K 2.72 s, 128K 4.66 s) and 256K is met, but
  only with the M4.2 levers on; they ship by default in M4.3. **Decode `>=25 tok/s` is not met** (~18 tok/s) and
  is a separate axis. The original misses are recorded in [ADR-0003](0003-engine-direction.md) and
  [docs/bongo-sh.md](../bongo-sh.md).

## Rollback path

All M4.1 work is additive and flag-gated. Reverting the flag or the engine pin restores the current shipped
behaviour. The Stage 0 Vulkan `--ctx 131072 --n-cpu-moe 16` baseline and the 256K default remain selectable. No
production infrastructure, DNS, billing, secrets, or data migration is touched.
