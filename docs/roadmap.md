# bongo roadmap

Milestones follow [ADR-0001](adr/0001-runtime-architecture.md). Each milestone is one or more issues; the
parent is [BAS-48](/BAS/issues/BAS-48) (Project setup).

## M0 — Foundation (done)

Research, architecture, and the task graph.

- [x] Target hardware confirmed on the reference box (Arc Pro B70, 32 GB; 30 GiB RAM; Fedora 44; `xe`).
- [x] Model identified and sized (`Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, three tiers).
- [x] Runtime options compared; baseline chosen ([ADR-0002](adr/0002-baseline-engine.md)).
- [x] [BAS-51](/BAS/issues/BAS-51) GGUF tensor inventory + buffer-placement plan — Coder, **done**
  ([doc](research/gguf-inventory.md), [tool](../tools/gguf-inventory.py)).
- [x] [BAS-56](/BAS/issues/BAS-56) speculation story settled: **no usable MTP head** (base weights have one,
  the published GGUF drops it and llama.cpp `qwen4exp` cannot convert or run it). Speculation is re-scoped to
  the n-gram/PLE path. See [research §2.1](research/intel-arc-b70.md) — Coder, **done**.
- [x] [BAS-50](/BAS/issues/BAS-50) `bongo.sh` one-command setup + OpenAI server — Coder, **done**
  ([script](../bongo.sh), [doc](bongo-sh.md)); QA-verified on the reference box. Fixed CX by
  [BAS-58](/BAS/issues/BAS-58) (idempotent user-local runtime, script on `main`, `--uninstall`, `need_cmd`).

## M1 — Baseline runs (Stage 0) — done

A working, reproducible, benchmarked one-command setup.

- [x] [BAS-50](/BAS/issues/BAS-50) one-command setup — Coder, **done**.
- [x] [BAS-52](/BAS/issues/BAS-52) benchmark harness + baseline numbers — Coder, **done**.
- [x] [BAS-54](/BAS/issues/BAS-54) QA end-to-end verification — QA, **done**. Core serving path PASS; the
  two blocking findings (F1 non-idempotent runtime, F2 script not on `main`) were fixed and regression-tested
  in [BAS-58](/BAS/issues/BAS-58). A truly clean-room run (no container/VM, 68 GB download) is UNVERIFIED.

Exit met: `./bongo.sh` reaches a 128K OpenAI-compatible endpoint. Stage 0 baseline,
IQ2_XS on the pinned engine (llama.cpp `b11223`), `--n-cpu-moe 16`, warm, median:

| context | prompt tok/s | output tok/s | TTFT ms |
| ---: | ---: | ---: | ---: |
| 1024 | 234.2 | 19.9 | 4318 |
| 4096 | 231.5 | 17.7 | 17706 |
| 32768 | 174.5 | 11.7 | 187760 |
| 131072 | 133.2 | 8.0 | 982856 |

Peak VRAM 29.27 GiB, peak RSS 14.07 GiB; 128K needle recalled. Raw data:
[`bench/results/2026-09-27-baseline/`](../bench/results/2026-09-27-baseline/).

**Runtime caveat:** the Intel compute stack (Level Zero / SYCL) does not enumerate the B70 (NEO abort), so
these numbers are on the **Vulkan** fallback. [BAS-57](/BAS/issues/BAS-57) tracks restoring SYCL.

## M2 — Expert placement (Stage 1) — resolved, then superseded by R1–R7

Close the gap between static llama.cpp placement and Strata's adaptive expert cache.

- [x] [BAS-53](/BAS/issues/BAS-53) expert-placement spike + Stage 1 go/no-go — Coder, **done**
  ([doc](research/expert-placement.md), [raw](../bench/results/2026-09-27-expert-placement/)).
- **First decision: no-go** on a custom adaptive VRAM expert cache as a *128K decode* optimisation. At 128K,
  decode is flat across the feasible static range (`--n-cpu-moe` 16 vs 24 differ ~1.2%); the static split is
  already at the VRAM edge (n=12 dies at 128K), so a cache cannot add residency.
- [x] Resolution of the M2 exit target: **IQ3_XXS does not fit this hardware** — 75,955,048,960 B
  (70.74 GiB) of tensor data against 32 GB VRAM + 32 GB RAM. Q2_0 (66,538,928,640 B / 61.97 GiB) is the
  nearest higher-quant candidate and is tracked in [BAS-59](/BAS/issues/BAS-59).
- **Superseded in part by [BAS-62](/BAS/issues/BAS-62) R1–R7 (2026-09-28).** The M2 verdict was right for 128K
  decode but too broad. Measured on bongo's own model: a frequency-ranked hot set covers 98.5% of activations
  vs 66.0% for the layer rule (worth +18–27% 128K prefill, +25–37% 4K decode, ~0% 128K decode); the real gap is
  the dequantising kernel path, not residency; RAM capacity is worth 0 tok/s. See
  [gap analysis](research/engine-gap-analysis.md).

## M3 — Engine work (Stage 2 gate now open) — decided by [ADR-0003](adr/0003-engine-direction.md)

The M2 gate asked "can llama.cpp reach the target?". The R1–R7 research is the missing gap analysis and it
answers: the gap is kernel maturity and scheduling, reachable by patching llama.cpp's SYCL backend — **not** a
from-scratch engine. The gate is therefore open for a **patch-based** engine plan, not for a SYCL rewrite.
See [ADR-0003](adr/0003-engine-direction.md). M3 is complete; the measurements are in below.

- [x] M3.0 Backend A/B: warm SYCL vs Vulkan at 4K/128K. **Vulkan stays the default** — SYCL did not clear the
  1.3x 128K gate ([BAS-72](/BAS/issues/BAS-72), [ADR-0002](adr/0002-baseline-engine.md) amended).
- [x] M3.0a Prefix-cache serving for agentic turns. 512-token turn **4.51 s** at 31K, full hit **0.29 s**
  ([BAS-73](/BAS/issues/BAS-73)). Slot save/restore at 256K measured ([BAS-83](/BAS/issues/BAS-83)), and the
  restore-reuse gap is fixed by the checkpoint sidecar ([BAS-86](/BAS/issues/BAS-86)).
- [x] M3.1 Quantized-weight (integer MMVQ) path. Correct and flag-gated, but **~1.05x** on the cached turn —
  below the 1.3x gate ([BAS-74](/BAS/issues/BAS-74)). The expert matmul is 3.4% of the turn (M3.6).
- [x] M3.2 Suffix/n-gram speculation, measured. **Negative on Vulkan**: 0.92-1.22x, no row reaches 1.3x, and
  greedy output is not bit-stable at 128K ([BAS-131](/BAS/issues/BAS-131)). No `bongo.sh` change.
- [x] M3.3 Placement Step 1: byte-budget `-ot` **+2.91%** 128K prefill / +0.74% decode, +4.25 pp coverage
  ([BAS-76](/BAS/issues/BAS-76)). Step 2, the dynamic VRAM LRU over RAM-pinned experts, is [BAS-139](/BAS/issues/BAS-139).
- [x] M3.4 PLE/n-gram second-shard reader: opt-in behind `--ple-reader off`; the engine A/B measured neutral
  ([BAS-79](/BAS/issues/BAS-79)) — [ADR-0004](adr/0004-ple-reader-disposition.md).
- [x] M3.5 Long context: **256K** shipped default q8 KV + `--n-cpu-moe 18`. Real 256K prefill, 128K needle
  PASS, peak 30.92 GiB ([BAS-78](/BAS/issues/BAS-78)); cached delta turn 10.87 s / 30.86 GiB
  ([BAS-132](/BAS/issues/BAS-132)).
- [x] M3.6 Warm-prefix profile: the cached turn is **80.1% host/CPU**, the MoE expert matmul 3.4%
  ([BAS-130](/BAS/issues/BAS-130)).

**Target status.** The turn targets are **met on the shipped default** since M4.3: 512-token turn 2.72 s at
16K (target <=3 s) and 4.67 s at 128K (target <=5 s), with the M4.2-patched engine and the upload
levers on by default; 256K is met. **The 4K decode target is re-baselined to `>=19.5 tok/s` (nominal ~20)**, a
median of 3 runs, per the CEO decision [BAS-164](/BAS/issues/BAS-164): the measured host-side ceiling is
19.59 tok/s (placement) and the GPU-busy floor is 38.7 ms/step. **`>=25 tok/s` is retired as a current
commitment** and becomes the goal of the unfunded GPU milestone [BAS-166](/BAS/issues/BAS-166) (backlog, low).
Umbrella [BAS-62](/BAS/issues/BAS-62) stays blocked by M4.5 ([BAS-163](/BAS/issues/BAS-163)) until the shipped
default reproduces the re-baselined target.

## M4 — Host/CPU critical path (open) — decided by [ADR-0005](adr/0005-host-cpu-critical-path.md)

M3 measured the planned GPU lever as 3.4% of the turn, so M4 targets the dominant term. The mechanism is the
per-batch host->VRAM **upload of host-resident MoE expert weights** on the main thread (2180 ms / 38.4% at 16K;
CPU workers 0%).

- [x] M4.1 ([BAS-144](/BAS/issues/BAS-144)) Decompose the host term and measure a lever. `--load-mode none`
  (anonymous RAM, no `mmap` re-read) gives **-10.4% at 16K / -7.1% at 128K** vs the M3.6 baseline. Opt-in and
  revertible.
- [x] M3.3b ([BAS-139](/BAS/issues/BAS-139)) Dynamic VRAM LRU over RAM-pinned experts: **measured negative**
  (-49% to -85%) and stopped — [ADR-0006](adr/0006-moe-expert-lru-disposition.md). Step-1 `-ot` remains the
  placement result.
- [x] M3.0c ([BAS-145](/BAS/issues/BAS-145)) Checkpoint-sidecar restore reuse confirmed at ~128K (`cache_n`
  = 127998).
- [x] M4.2 ([BAS-155](/BAS/issues/BAS-155)) Located the upload and fixed the root cause: the Vulkan host buffer
  type was pinned to `devices[0]` (the AMD iGPU), so every copy staged through CPU and synchronised. Device-local
  host buffer + transfer queue: **16K -23.7% (2721.8 ms), 128K -14.95% (4661.0 ms)**, needle pass. Env-gated,
  default off.
- [x] M4.3 ([BAS-158](/BAS/issues/BAS-158)) Shipped the M4.2 upload fix as the `bongo.sh` default
  (patched engine + `--load-mode none` + the two `GGML_VK` levers, built/selected reproducibly) with a
  no-rebuild `--engine stage0` opt-out. Shipped default: **16K 2 716 ms (−23.9%), 128K 4 669 ms
  (−14.80%)**, cold prefill within 0.05% of M4.2, needle pass, 256K fit 30.65 GiB. Corrects the RAM
  premise: the default's VmRSS is 2.4 GiB (the ~10 GiB set is the `mmap` opt-out's working set).
  ([doc](research/m4.3-shipped-default.md), [raw](../bench/results/2026-09-29-m4.3-shipped-default/))
- [x] M4.4 ([BAS-159](/BAS/issues/BAS-159)) Decode profiled: dense matmuls 40.3%, flash attention 20.5%, MoE
  expert matmul 10.3%, host ~34%. The MoE term is not dominant, and decode does **no** expert upload (batch 1 is
  below the Vulkan offload threshold of 32, so the host-resident experts run on the CPU). Placement is the only
  lever found: `--n-cpu-moe 12` gives **19.59 tok/s (+15.5%)**; 10/8 OOM at 131072, so that is the ceiling.
  `--placement auto` is landed but opt-in. ([doc](research/m4.4-decode-profile.md),
  [raw](../bench/results/2026-09-29-m4.4-decode/))
- [ ] M4.5 Make `--placement auto` the default **with an automatic OOM fallback** to the tier placement, so the
  decode gain ships without sitting on the VRAM load edge.
- [x] M4.6 ([BAS-167](/BAS/issues/BAS-167)) Record the re-baseline in
  [ADR-0005](adr/0005-host-cpu-critical-path.md) and this roadmap, and close [BAS-62](/BAS/issues/BAS-62) as
  met-with-re-baselined-target once M4.5 lands. Docs carried the re-baseline; the closure waits on M4.5.
- [x] CEO call ([BAS-164](/BAS/issues/BAS-164)): **re-baseline decode to `>=19.5 tok/s` (~20), median of 3, on
  the shipped default**; `>=25 tok/s` is retired as a current commitment and moves to the unfunded GPU milestone
  [BAS-166](/BAS/issues/BAS-166) (backlog, low — not started). The re-baselined target is met once M4.5
  ([BAS-163](/BAS/issues/BAS-163)) ships the `--placement auto` default.
- Deferred: Level Zero command-list capture (R1c).

MTP is **not** on this list: the base model's head is not in the published GGUF and llama.cpp `qwen4exp` cannot
convert or run it ([research §2.1](research/intel-arc-b70.md)).

## Cross-cutting

- **Licence:** confirm the model's terms for scripted/automated download before shipping the setup publicly.
- **Reproducibility:** every result records the llama.cpp commit, model tier, shard hashes, driver, and flags.
- **Rollback:** all changes are config-pin reversible; no data migrations.

## Task graph

```
BAS-48 (Project setup)
 ├─ BAS-50 bongo.sh setup (Coder) ──┬─ BAS-52 benchmark (Coder) ── BAS-53 placement spike (Coder)
 │                                  └─ BAS-54 QA verification (QA) ── BAS-58 CX fixes (Coder)
 ├─ BAS-51 GGUF inventory (Coder)
 ├─ BAS-56 MTP/speculation finding (Coder)
 ├─ BAS-57 restore SYCL device enumeration (Coder)        [open]
 └─ BAS-59 trial Q2_0 at 128K (Coder)                     [open]
```
