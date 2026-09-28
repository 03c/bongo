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
See [ADR-0003](adr/0003-engine-direction.md). Milestones, in measured expected-value order:

- [ ] M3.0 Backend A/B: warm SYCL vs Vulkan at 4K/128K → pick the default (R6 decision rule).
- [ ] M3.1 Quantized-weight (integer MMQ/MMVQ) MoE + dense path for IQ2_XS on SYCL; no FP16 expansion.
- [ ] M3.2 Suffix/n-gram speculation with the exact verify/commit window and an online acceptance policy.
- [ ] M3.3 Placement: cheapest-layer-first byte-budget `-ot`, then a dynamic VRAM LRU over RAM-pinned experts
  (R4 profile as initialisation only, held-out A/B required).
- [ ] M3.4 PLE/n-gram second-shard reader: direct reads, parallel prefetch, bounded row cache.
- Deferred: Level Zero command-list capture (R1c); a from-scratch engine only if M3.1–M3.3 measurably miss the
  target and a new ADR names the unfixable gap.

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
