# ADR-0001 — bongo runtime architecture (staged)

- Status: Accepted
- Date: 2026-09-27
- Deciders: CTO
- Related: [ADR-0002](0002-baseline-engine.md), [`docs/research/intel-arc-b70.md`](../research/intel-arc-b70.md)

## Context

We must run a 125B-parameter MoE model (Swift 1.5 / Qwen3.8-Flash-Next, `qwen4exp`) on a single Intel Arc Pro
B70 with **32 GB VRAM + 32 GB RAM + SSD**, at **>=128K context**, behind an OpenAI-compatible API, from a
**one-command setup**.

The reference implementation, Strata, is a from-scratch CUDA engine that assumes *all 24,576 experts fit in
system RAM* and keeps only the hot ones on the GPU. Our box has half Strata's recommended RAM, so the total
expert capacity (VRAM + RAM ~ 50 GB) is sufficient but the placement problem is harder: some experts must be
computed on the CPU, and the least-hot may need to stream from SSD.

We do not have an Intel GPU kernel codebase, and the GPU's software stack (Level Zero / oneAPI) is not even
installed on the target box yet.

Key facts (see research doc): llama.cpp upstream already supports the `qwen4exp` architecture, i-quants, MTP,
and MoE CPU offload; the SYCL backend supports Arc B-Series; the 48-layer hybrid model makes a 128K KV cache
only ~2-4 GB, so KV is not the binding constraint — expert placement is.

## Decision

Adopt a **three-stage architecture**, and do not start from a blank custom kernel:

1. **Stage 0 — baseline (llama.cpp SYCL, pinned).**
   A `bongo.sh` that provisions the Intel compute runtime, builds/pins llama.cpp with `-DGGML_SYCL=ON`,
   downloads the GGUF tier, and launches `llama-server` with an OpenAI endpoint and >=128K context. This is
   the shippable artifact and the benchmark reference.

2. **Stage 1 — expert-placement layer (bongo-owned).**
   Optimise where experts live and how they move: per-layer GPU/CPU split (`-ot` / `--n-cpu-moe`), KV
   quantisation, MTP speculation, and — the actual differentiator — an **adaptive VRAM expert cache** with
   RAM backing and optional SSD streaming, adapting residency to the conversation. Implemented as a thin
   scheduler/config layer over llama.cpp where possible; a fork or a standalone engine only where necessary.

3. **Stage 2 — custom SYCL engine (gated).**
   Only if Stage 1 measurements show llama.cpp cannot reach the target, port the parts of Strata that matter
   to SYCL (MoE dispatch, i-quant GEMV, fused attention for the 12 full-attention layers, linear-attention
   mixers, MTP). Gate: Stage 1 benchmarks plus a written gap analysis.

## Alternatives considered

- **Blank custom SYCL engine from the start (Strata port).** Highest ceiling, highest cost and risk, and it
  blocks the one-command deliverable for months with no measured baseline. Rejected as a starting point;
  retained as Stage 2.
- **IPEX-LLM prebuilt only.** Fastest to a running server, but we inherit a fork's release cadence and cannot
  implement the adaptive expert cache. Kept as an optional zero-build fallback for first boot.
- **OpenVINO / vLLM XPU.** Neither runs the published i-quant GGUF; both would require re-quantising the model,
  changing quality and scope. Rejected for v1.
- **Vulkan instead of SYCL.** Fewer dependencies, but weaker MoE and i-quant coverage. Kept as a fallback and a
  benchmark comparison, not the default.

## Consequences

- We get a working OpenAI endpoint and a real benchmark early, which de-risks the whole project.
- The one-command setup must install the Intel compute stack. That is a supported-configuration risk we must
  test on a clean box (Fedora 44 first).
- Stage 1 may require a llama.cpp fork; we accept a pinned fork if a config-only layer cannot express adaptive
  cache residency.
- No custom CUDA-to-SYCL port work is authorised until Stage 2's gate is written down and met.

## Rollback path

Stages are additive and independently shippable. If Stage 1 regresses, `bongo.sh` can pin back to the Stage 0
llama.cpp revision and flags; the model download and API surface are unchanged. If Stage 2 is abandoned, the
Stage 1 layer remains the product. No schema or data migration is involved, so rollback is a config pin.

## Amendment (2026-09-27, [BAS-56](/BAS/issues/BAS-56))

MTP is **not** an available capability for this model. The base model and the Swift checkpoint both carry a
1-layer MTP head, but the published GGUF drops it and llama.cpp `qwen4exp` is not wired into the generic MTP
machinery, so it cannot convert or run one. The MTP items in the Decision above (Stage 1 "MTP speculation",
Stage 2 "MTP") are therefore **void** for the current weights; Stage 1 speculation is re-scoped to the
n-gram/PLE path. Everything else in the decision stands. See
[`docs/research/intel-arc-b70.md`](../research/intel-arc-b70.md) §2.1.
