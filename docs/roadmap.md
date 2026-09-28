# bongo roadmap

Milestones follow [ADR-0001](adr/0001-runtime-architecture.md). Each milestone is one or more issues; the
parent is [BAS-48](/BAS/issues/BAS-48) (Project setup).

## M0 — Foundation (in progress)

Research, architecture, and the task graph.

- [x] Target hardware confirmed on the reference box (Arc Pro B70, 32 GB; 30 GiB RAM; Fedora 44; `xe`).
- [x] Model identified and sized (`Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, three tiers).
- [x] Runtime options compared; baseline chosen ([ADR-0002](adr/0002-baseline-engine.md)).
- [x] [BAS-51](/BAS/issues/BAS-51) GGUF tensor inventory + buffer-placement plan — Coder, **done**
  ([doc](research/gguf-inventory.md), [tool](../tools/gguf-inventory.py)).
- [x] [BAS-56](/BAS/issues/BAS-56) speculation story settled: **no usable MTP head** (base weights have one,
  the published GGUF drops it and llama.cpp `qwen4exp` cannot convert or run it). Speculation is re-scoped to
  the n-gram/PLE path. See [research §2.1](research/intel-arc-b70.md) — Coder, **done**.
- [ ] [BAS-50](/BAS/issues/BAS-50) `bongo.sh` one-command setup + OpenAI server — Coder, **ready**.

## M1 — Baseline runs (Stage 0)

A working, reproducible, benchmarked one-command setup.

- [BAS-50](/BAS/issues/BAS-50) one-command setup (llama.cpp SYCL) — Coder.
- [BAS-52](/BAS/issues/BAS-52) benchmark harness + baseline numbers — Coder, blocked by BAS-50.
- [BAS-54](/BAS/issues/BAS-54) QA end-to-end verification on a clean box — QA, blocked by BAS-50.
- Exit: `./bongo.sh` reaches a 128K OpenAI endpoint; IQ2_XS prompt/output tok/s and VRAM/RAM recorded.

## M2 — Expert placement (Stage 1)

Close the gap between static llama.cpp placement and Strata's adaptive expert cache.

- [BAS-53](/BAS/issues/BAS-53) expert-placement spike + Stage 1 go/no-go — Coder, blocked by BAS-52.
- Follow-ups created from BAS-53's recommendation: adaptive VRAM expert cache, SSD expert streaming,
  n-gram/PLE speculation tuning, n-gram table handling. **MTP tuning is not a follow-up**: the published
  GGUF has no MTP head and llama.cpp `qwen4exp` cannot convert or run one ([research §2.1](research/intel-arc-b70.md)).
- Exit: >= IQ3_XXS fits and runs at 128K; a written Stage 1 gate decision.

## M3 — Optimised engine (Stage 2, gated)

Only if M2 shows llama.cpp cannot reach target throughput. Port Strata's MoE dispatch, i-quant GEMV, fused
attention, and linear-attention mixers to SYCL. **MTP is not on this list**: the base model's head is not in
the published GGUF and llama.cpp `qwen4exp` cannot convert or run it ([research §2.1](research/intel-arc-b70.md));
adding it would be a separate converter + runtime feature, not a SYCL port. Requires a written gap analysis
and a new ADR.

## Cross-cutting

- **Licence:** confirm the model's terms for scripted/automated download before shipping the setup publicly.
- **Reproducibility:** every result records the llama.cpp commit, model tier, shard hashes, driver, and flags.
- **Rollback:** all changes are config-pin reversible; no data migrations.

## Task graph

```
BAS-48 (parent, blocked)
 ├─ BAS-50 bongo.sh setup (Coder) ──┬─ BAS-52 benchmark (Coder) ── BAS-53 placement spike (Coder)
 │                                  └─ BAS-54 QA verification (QA)
 └─ BAS-51 GGUF inventory (Coder)
```
