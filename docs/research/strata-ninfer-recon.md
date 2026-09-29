# Recon: how Strata and NInfer beat llama.cpp on one box

**Status:** first-pass reconnaissance for [BAS-62](/BAS/issues/BAS-62), 2026-09-28. Written by the CTO from the
upstream repos. It grounds the research task graph; it is not the build plan (that is the synthesis task) and not
a substitute for the dedicated deep-dives.

**Sources read directly:**
[Niko1221/Strata](https://github.com/Niko1221/Strata) `README.md`, `docs/DETAILS.md`,
`include/strata/core/expert_cache.hpp`, `include/strata/plan/plan.hpp`,
`include/strata/ngram/ple_reader.hpp`, `include/strata/spec/suffix_drafter.hpp`,
`bench/results/2026-09-27-esp/README.md`, `bench/results/2026-09-28-prefill-speed/README.md`;
[Neroued/ninfer](https://github.com/Neroued/ninfer) `README.md`.

## 1. Why this task exists: the measured gap

The Stage 0/1 work concluded "no-go on a custom engine" on the strength of a **static `--n-cpu-moe` sweep run on
the llama.cpp Vulkan fallback**. That sweep showed 128K decode is flat across the feasible placement range
(~7.6-8.0 tok/s). It did **not** test a different engine. The comparison below is the missed variable:

| | Hardware | Engine | 128K decode |
| --- | --- | --- | ---: |
| bongo Stage 0 | Arc Pro B70 32 GB + 32 GB RAM | llama.cpp `b11223`, **Vulkan** | **~8 tok/s** |
| Strata | RTX 5070 12 GB + 64 GB RAM | Strata (custom CUDA) | **52 tok/s** (IQ2_XS), 65 (Q2_0) |

Strata is roughly **6-8x faster on a weaker GPU with less VRAM**. The gap is engine, not hardware dominance:
Strata's own notes say its win comes from MoE kernels that keep weights quantized (MMQ), overlapped expert
streaming, and speculative decoding — not from moving experts around (see §4). That is exactly the class of work
Stage 2 was reserved for and then never gated on. The premise of the M2 no-go is therefore **superseded**, and
Stage 2 must be re-planned on evidence rather than deferred.

## 2. Strata — the tiered design, in detail

Strata is a from-scratch CUDA engine for the same model family (Qwen3.8-Flash-Next 125B MoE). Its own words:
GPU runs the every-token weights plus the most-used experts; **RAM holds all 24,576 experts and the CPU computes
the few the GPU does not have, at the same time as the GPU**; the SSD holds a large lookup table read a few rows
per token.

### 2.1 The memory planner is arithmetic, not a config

`plan/plan.hpp` derives the placement from model geometry and **refuses** rather than overcommitting
(`DoesNotClose`). Fixed costs are: dense weights, embedding, workspace, **recurrent GDN state** (117.7 MB at
fp32 for 36 GDN layers — explicitly not evictable), and INT8 KV + indexer keys. What remains of the VRAM pool is
divided into expert-cache slots of a fixed blob size (1,382,400 B per expert for IQ3-class). The plan reports
slots and throws if zero fit. This is the pattern bongo lacks: llama.cpp takes flags; Strata computes a plan.

### 2.2 The expert cache is a static, frequency-ranked residency table — not per-token streaming

`expert_cache.hpp` is explicit and worth quoting in the plan:

- The problem it targets: the CPU expert pool is the engine's **largest single cost** — measured
  **663.6 MB of expert bytes per token at ~40 GB/s = 16.2 ms**, on a ~53 ms token; the pipeline hides only
  1.055 ms of 19.0 because the residual chain is serial.
- The fix: put the 48 x 512 experts' **most frequently routed** subset in VRAM so the CPU reads only the
  **misses**. `profile.bin` (STRP format, built by `tools/make_profile.py`) is a frequency ranking; at 4,105
  slots the leave-one-out hit rate `h_expert` = **0.6447**, against **0.4864** for a "compulsory-miss"
  arrival-order fill.
- **It never evicts.** Admission is one-time from the profile; eviction policy is called out as a separate
  measured question (`LFU-decay vs LRU`). A per-layer admission variant exists because global arrival-order
  filled the whole cache inside the first two layers (2.97% hit rate; per-layer 8 slots → 21.4%).
- The file states the cache is, today, **not wired into the compute graph** — it is slot storage + residency
  only, and `--expert-cache` defaults to 0. So Strata's headline numbers are **not** from the expert cache; they
  are from the kernels and speculation below. That is an important caveat for our own plan: the cache is a
  promising, measured-later lever, not the current source of Strata's speed.

**This answers the "stream weights to VRAM when needed" assumption:** Strata does **not** demand-page experts per
token. It pins an offline-computed hot set at startup and CPU-computes the rest concurrently. The only per-token
streaming is of the **n-gram/PLE table** (§2.3), which is not a weight.

### 2.3 The second shard (n-gram/PLE table) is SSD-resident by design — confirmed

`ngram/ple_reader.hpp`: the table is **320,001,536 rows x 90 bytes = 26.8 GiB**, and is **never held in RAM**.
Every row is an unbuffered 4 KiB read from the model file. A token needs **16 rows on 16 different pages**, all
determined by the token id, so the only window to hide them is embedding + layer 0. The split API issues the
reads as soon as the token id is known and collects before layer 1. A bounded row cache (~1M rows, ~95 MB)
measures up to ~82% of reads served from cache; 20-34% of rows recur inside one long prompt. Prefill batches a
chunk's rows with page dedup and a bounded in-flight window.

**Answer to the CEO's question: yes — the n-gram/second shard can live on SSD permanently.** Strata already
ships this and bongo's `CONTEXT.md` already treats the table as disk-resident; the open work is the access
pattern, prefetch overlap, and row cache, not feasibility.

### 2.4 Speculation: suffix/ngram drafter, no weights, no GPU

`spec/suffix_drafter.hpp` is a trigram suffix-lookup drafter (WAYS=4, min_match 3, max_match 32, ~20 B per
history token). It proposes the tokens that followed the longest earlier occurrence of the current suffix — near
free, no model. Combined with prompt lookup it gives Strata's advertised **1.6-1.8x** exact-equivalence speed-up
(on copied/edited text acceptance hits ~91%). bongo's MTP path is dead (published GGUF has no MTP head), so this
is the *only* speculation lever available to bongo — and Strata proves it is worth a large multiple.

### 2.5 Kernel and prefill wins (the actual source of the headline numbers)

`bench/results/2026-09-28-prefill-speed/README.md` (engine 0.1.13, Q2_0, 32K prompt): **572 -> 1,290 prompt
tok/s**, broken down as: shared scratch buffers; PLE rows prefetched on a thread; auto chunk size; host experts
copied on helper threads; whole-chunk PLE block; **MMQ kernels (weights stay quantized, int8 tensor cores)
instead of dequantize-to-FP16 + cuBLAS** (1052 -> 1130); and **non-resident experts streamed in a fixed ring
overlapped with the current layer's attention** (1130 -> 1290). The streamed step is bit-identical. Quality was
checked with llama.cpp-style teacher-forcing + KL and needle tests.

The takeaway is that the big multipliers are kernel- and schedule-level (quantized matmul, overlap, prefetch),
which is exactly what a generic llama.cpp Vulkan path leaves on the table.

## 3. NInfer — the single-GPU quality/perf ceiling

NInfer is a from-scratch C++/CUDA engine for one RTX 5090, one resident model, 1-8 active requests, custom
`.ninfer` artifacts. It is **not** a tiered-memory engine: it explicitly lists "no weight offload" as a product
boundary. Its relevance to bongo is the parts that are portable at the design level:

- **A custom artifact format** carrying model config + encoded weights + bindings, so the runtime only
  implements explicitly supported format/shape paths (vs. GGUF's generic tensor soup). Worth weighing for a
  bongo-owned format.
- **MTP speculative decoding with draft windows 1-5** and dedicated companion heads (`--spec mtp`, DFlash,
  DFlash2). bongo cannot use MTP on the published GGUF, but the *verify/kernel structure* is a reference.
- **Device/Host checkpoint and KV tiers** — exact-prefix reuse with a planner that weighs device retention,
  pinned host state/KV, and eviction by restore-vs-reuse cost. Directly relevant to bongo's 128K KV budget.
- **Chunked prefill, CUDA-graph decode, fused kernels** across the same hybrid attention family (GDN/linear +
  full attention), which is the same architectural family as Qwen3.8.
- Published single-request: 8,340 tok/s prefill (7,680 tok, nvfp4) and ~220 tok/s MTP3 decode. That is the
  per-request ceiling on a 5090; bongo's target should be framed as a fraction of this scaled to the Arc B70.

## 4. What this changes

1. **Reopen Stage 2.** The M2 no-go measured llama.cpp placement, not engine quality. Write the gap analysis and
   a superseding ADR (task S1) once the deep-dives land.
2. **Placement is not the main lever; kernels + speculation + overlap are.** Budget the build accordingly:
   quantized-weight MoE matmul, KV/attention for the 12 full-attention layers, linear/GDN state for the other
   36, suffix/ngram speculation, and PLE prefetch — in that order of expected value.
3. **bongo's problem is harder than Strata's, on one axis.** Strata assumes **all experts fit in 64 GB RAM**;
   the reference box has **32 GB RAM + 32 GB VRAM**. So bongo genuinely needs the hot-expert-residency question
   answered (VRAM residency + CPU compute + possibly SSD streaming), *more* than Strata does. The CEO's
   "predict which experts" instinct is right for this box even though Strata's own cache is static.
4. **The 6-8x gap is the target.** Any plan that only re-tunes llama.cpp flags will not close it.

## 5. Open questions routed to research tasks

- **R1 Strata deep-dive:** exact dispatch/scheduling algorithm, kernel list mapped to SYCL portability, the
  profile construction and transferability, the `--expert-cache` wiring gap, and the speculation acceptance
  mechanism (verify window).
- **R2 NInfer deep-dive:** artifact format, MTP/DFlash verify structure, Device/Host KV planner, kernel
  organisation, and what is NVIDIA-specific vs. portable.
- **R3 Landscape:** PowerInfer, KTransformers, AirLLM, FlexGen, DeepSpeed ZeRO-Inference, HyperQwen, Splash,
  and the llama.cpp offload PRs — does anyone stream experts from **SSD** per token, and at what hit rate/latency?
- **R4 Expert activation on the reference box:** measure routing skew on the actual model; can a profile/predictor
  beat the layer rule, and by how much?
- **R5 SSD/PLE shard on the reference box:** measure 16-row-per-token read latency, row-cache hit rate, and
  overlap budget against the bongo harness.
- **R6 Intel/SYCL feasibility:** which of Strata's kernels map to SYCL/Level Zero, what llama.cpp's SYCL backend
  already covers, and the root cause of the B70 device-loss/NEO abort.

The synthesis task (S1) turns the six results into the build plan + ADR.
