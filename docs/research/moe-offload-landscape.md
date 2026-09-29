# MoE-offload / single-model engine landscape — does anyone stream expert weights from SSD?

**Status:** complete, 2026-09-28. Owner: Researcher ([BAS-65](/BAS/issues/BAS-65)). Feeds the Stage-2
synthesis ([BAS-62](/BAS/issues/BAS-62), S1) alongside the Strata and NInfer deep-dives in
[`strata-ninfer-recon.md`](strata-ninfer-recon.md).

**Scope.** Survey the MoE-offload and single-model inference-engine landscape beyond Strata and NInfer and
answer the question the CEO asked directly: *are they streaming weights to VRAM when needed, and can the
second shard / n-gram be on SSD at all times?* Read-only on upstream repos; no vendoring; no multi-GB
downloads. Local measurement is limited to a read-only random-4K probe of the reference box NVMe
([`bench/results/2026-09-28-ssd-random4k-probe/`](../../bench/results/2026-09-28-ssd-random4k-probe/README.md)).

**Method.** Every system below was read from its own repository (README, docs, source) or its paper, fetched
2026-09-28. Review comments and PR bodies are quoted only where they carry measured numbers, and are marked
with their PR state (most of the llama.cpp offload work is **open, unmerged draft** — treat it as evidence,
not as shipped behaviour). Where a number exists, it is cited with its hardware, batch, and context; where it
does not, that is stated rather than estimated.

---

## TL;DR verdict

1. **Yes — people do stream MoE expert *weights* from SSD per token, and it is slow.** Three independent
   implementations exist: llama.cpp's disk-streaming draft [PR #25294](#llamacpp-pr-25294),
   AirLLM's per-layer/per-expert streaming, and FlexGen / DeepSpeed ZeRO-Inference's disk tier. The best
   measured disk-streamed decode on a model-larger-than-RAM is **~2.2 tok/s at a 79 % expert-cache hit rate**
   (llama.cpp #25294, GLM-5.2 on a GB10). That is ~3.5x slower than bongo's current pinned split (7.62 tok/s
   at 128K) and nowhere near the 50+ tok/s target. **Weight streaming from SSD is a feasibility mechanism, not
   a performance mechanism.**
2. **No shipped system relies on a *static, frequency-ranked* hot-expert set.** The only held-out measurement
   of static ranking in the qwen4exp family shows a top-32 "hot" list learned on half a workload covers only
   **~10 %** of the other half (uniform = 6.2 %) — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861).
   A *dynamic* per-layer LRU is what works: **67 % at 64 slots, 81 % at 128 slots**, for +31 % end-to-end
   decode. Strata's `profile.bin` hit rate (0.6447) is a leave-one-out *in-sample* number and the cache is not
   wired into compute, so it is not counter-evidence. This is the single most decision-relevant finding for
   bongo's "predict which experts" instinct: the predictor should be an online LRU, not an offline profile.
3. **Yes — the second shard (n-gram / PLE table) can be on SSD permanently, and every serious qwen4exp
   deployment already does it.** Strata (16 × 4 KiB unbuffered reads/token, ~82 % row-cache hit), the DGX
   Spark recipe (`-ot per_layer_token_embd=CPU -lm mmap`, page-cache served), the 4×3090 vLLM patch (16 rows
   per token inside CUDA graphs), and llama.cpp's `--lazy-mode` all serve a ~26.8 GiB table that is never
   resident. The one caution: Llama.cpp's own mmap-based lazy reads are up to **2x slower than direct reads**
   for prefill ([PR #29030](https://github.com/ggml-org/llama.cpp/pull/29030)), so the reader matters.
4. **The recommendation, ranked: (a) placement — dynamic VRAM LRU over RAM-pinned experts, not an offline
   profile; (b) kernels — Strata's quantized-weight (MMQ) path; (c) speculation — the suffix/ngram drafter;
   (d) SSD I/O — the PLE row reader + `--lazy-mode`. Do not budget for SSD-resident expert weights unless a
   higher tier (IQ3_XXS) is chosen and then does not fit.**

---

## 1. bongo's arithmetic (what the answer has to fit)

From the measured GGUF inventory and placement sweep already in the repo:

| Quantity | Value | Source |
| --- | ---: | --- |
| IQ2_XS tensor bytes | 63.46 GiB | [`gguf-inventory.md`](gguf-inventory.md) §5 |
| — MoE experts (`ffn_*_exps`) | 33.02 GiB | ibid. |
| — n-gram/PLE table | 26.82 GiB | ibid. |
| — non-expert, non-ngram weights | 3.62 GiB | ibid. |
| Arc Pro B70 usable VRAM | 31.92 GiB | [`expert-placement.md`](expert-placement.md) §"A VRAM correction" |
| System RAM | 30 GiB | reference box |
| PLE rows gathered per token | 16 (2 × 8 heads) | [`gguf-inventory.md`](gguf-inventory.md) §3 |
| 128K decode, `--n-cpu-moe 16` (baseline) | 7.62 tok/s | [`expert-placement.md`](expert-placement.md) §Results |
| 128K decode, `--n-cpu-moe 48` (all CPU) | 2.26 tok/s | ibid. |
| 128K decode, all experts on GPU | does not load (33 GiB > 31.92 GiB) | ibid. |

So the relevant questions are: can a **dynamic** expert cache beat the static `n=16` split, and is there any
case for a **third tier (SSD)** rather than only VRAM + RAM.

---

## 2. Comparison table

Legend: **Tiers** = where weights can live. **Experts from SSD/token** = per-token weight reads from disk,
not KV/state. **Quant kernels** = weights stay quantized through the matmul. All measured throughputs are
cited with their hardware and batch.

| System | Tiers | Experts from SSD/token? | Quant-weight kernels | Speculation | Measured (hardware, workload) | Source |
| --- | --- | --- | --- | --- | --- | --- |
| **FlexGen / FlexLLMGen** | GPU, CPU, disk | **Yes** (weights+KV to disk by `--percent`) | Yes (4-bit weights+KV) | No | OPT-175B, 1×T4, 512/32, max batch: **0.69 tok/s** all-disk, 7.32 CPU, 25.26 GPU | [README](https://github.com/FMInference/FlexGen#generation-throughput-tokens), [paper](https://arxiv.org/abs/2303.06865) |
| **DeepSpeed ZeRO-Inference + DeepNVMe** | GPU, CPU, NVMe | **Yes** (weights to CPU or NVMe) | Yes (INT4 W4) | No | LLAMA3-70B 1×A100-80, bsz 96, 512/32: **7 tok/s** (4×Gen4), 17 (4×Gen5), 26 (8×Gen5); NVMe 4×Gen4 **10 GB/s** read, 8×Gen5 48 GB/s | [DeepNVMe 06-2025](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/06-2025/README.md), [08-2024](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/08-2024/README.md), [ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md) |
| **AirLLM** | GPU (1 layer), disk | **Yes, per layer / per expert** | Yes (optional 4/8-bit on-disk shards) | No | Qwen3.8-Flash-Next **5.95 GB VRAM** on RTX 4090 (README); no tok/s published; "bottleneck is mainly at the disk loading"; prefetch = +10 % | [repo](https://github.com/lyogavin/airllm), [`airllm_base.py`](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_base.py), [`airllm_kimi_k3.py`](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_kimi_k3.py), [`airllm_qwen4_exp.py`](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_qwen4_exp.py) |
| **PowerInfer** | GPU (hot), CPU (cold) | No | Yes (sparse, quantized) | Predictor, not token speculation | OPT-175B class, 1×RTX 4090: **13.2 tok/s avg, 29.1 peak**; 18 % below A100 | [repo](https://github.com/SJTU-IPADS/PowerInfer), [paper 2312.12456](https://arxiv.org/abs/2312.12456) |
| **PowerInfer-2** | NPU, CPU, **flash storage** | **Yes** (neuron clusters from flash) | Yes | No | 47B on a smartphone: **11.68 tok/s**; segmented neuron cache + cluster I/O pipeline | [paper 2406.06282](https://arxiv.org/abs/2406.06282) |
| **KTransformers** | GPU, CPU | No | Yes (AMX INT4/INT8 CPU, GPTQ GPU) | No | DeepSeek-R1/V3 671B, 4090 24 GB + 382 GB DRAM: **8.73 tok/s** (32 cores) → 11.26 (dual-socket) → 13.69 (6-expert selection), vs 4.51 llama.cpp | [repo](https://github.com/kvcache-ai/ktransformers), [DeepSeek tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md) |
| **KTransformers expert scheduler** | GPU, CPU | No | Yes | No | Qwen3-Next-80B, 4×4090: 53 → 114 tok/s as GPU expert ratio 0→100 %; frequency/dynamic beat uniform at low ratios | [experts-sched](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md) |
| **IPEX-LLM FlashMoE** | GPU, CPU | No | Yes (GGUF i-quants on SYCL) | No | DeepSeek 671B on 1–8 Arc A770/B580 needs **380 GB CPU RAM**; Qwen3MoE 235B needs **128 GB**; 500 GB disk | [flashmoe_quickstart.md](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/flashmoe_quickstart.md), [repo](https://github.com/intel/ipex-llm) |
| **HyperQwen** | GPU | No | Yes (int8 Marlin) | Yes (DFlash2/MTP) | Qwen3.8-27B, 1×RTX 3090 24 GB: **127 tok/s** single, 1,035 tok/s at 64 concurrent, 150K ctx | [repo](https://github.com/syv-ai/HyperQwen) |
| **Splash (incoai)** | GPU, SSD (KV/GDN) | No (weights); SSD tier for **KV pages + GDN state** | Yes (per-model precompiled kernels) | Yes (draft model) | Qwen3.8 family on Apple silicon: 210 tok/s decode, 2,011 tok/s prefill 32K | [repo](https://github.com/incoai/splash), [SSD-tier PR #3](https://github.com/incoai/splash/pull/3) |
| **llama.cpp `--cpu-moe` / `--n-cpu-moe`** | GPU, CPU | No | Yes (MMQ/MMVQ) | via `--spec-type ngram-mod` | bongo baseline: 7.62 tok/s at 128K; `n=48` CPU-only 2.26 tok/s | [arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp), [`expert-placement.md`](expert-placement.md) |
| **llama.cpp `--lazy-mode` (`-lzm`)** | CPU + **disk** | **Yes, for large lookup tensors** (PLE); rows only | n/a (embedding gather) | n/a | `-lzm on` vs direct reads, Strix Halo: pp512 181 → 401 tok/s, pp8192 274 → 451 | [PR #29030](https://github.com/ggml-org/llama.cpp/pull/29030) |
| **llama.cpp PR #25294 (open)** | GPU, **disk** | **Yes, routed experts, O_DIRECT** | Yes | No | GLM-5.2 (~254 GB) on GB10: **~1.83 tok/s** decode @ 73 % hit (64 slots/~55 GB); **~2.20** @ 79 % (90 slots) | [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) |
| **llama.cpp PR #27861 (open)** | GPU, CPU (LRU into VRAM) | No | Yes | No | Qwen3.8-Flash-Next UD-Q4_K_XL, 2×3090: **18.4 → 24.2 tok/s (+31 %)** with LRU-48/layer (~4.1 GiB); LRU-64 hit 67 %, LRU-128 81 % | [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861) |
| **llama.cpp PR #26414 (open)** | GPU, CPU (mlock) | No | Yes | No | prevents page-cache eviction of mmap experts; `--pin-hot-experts N` | [PR #26414](https://github.com/ggml-org/llama.cpp/pull/26414) |
| **llama.cpp PR #28414 (open)** | GPU, CPU | No | Yes | No | 24B-A3B, 42K prompt: TTFT **15.71 → 13.97 s (−11 %)**, byte-identical; decode flat | [PR #28414](https://github.com/ggml-org/llama.cpp/pull/28414) |
| **Strata** | GPU, CPU, **SSD (PLE)** | No (weights); PLE yes | Yes (**MMQ**) | Yes (suffix/ngram, 1.6–1.8x) | Qwen3.8-Flash-Next IQ2_XS, RTX 5070 12 GB + 64 GB: **52 tok/s** 128K | [`strata-ninfer-recon.md`](strata-ninfer-recon.md), [repo](https://github.com/Niko1221/Strata) |
| **NInfer** | GPU only | No ("no weight offload" by design) | Yes | Yes (MTP/DFlash) | 1×RTX 5090: 8,340 tok/s prefill, ~220 tok/s MTP3 decode | [`strata-ninfer-recon.md`](strata-ninfer-recon.md), [repo](https://github.com/Neroued/ninfer) |
| **0xBakeer DGX-Spark recipe (qwen4exp)** | GPU/unified, **NVMe PLE** | No (weights); PLE via page cache | Yes (GGUF) | Yes (`ngram-mod`) | 180B on 128 GB DGX Spark: **21.05 → 22.40 tok/s** as PLE cache 1.3 %→79 %; 88 tok/s on file rewrite; 128K doc ~56 s | [repo](https://github.com/0xBakeer/qwen38-flash-next-spark), [how-it-works](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/main/docs/how-it-works.md) |
| **tonyd2wild 4×3090 (vLLM)** | VRAM, **NVMe PLE** | No (weights); PLE 16 rows/token | Yes (W4A16 Marlin + FP8 PLE) | Yes (INT4 MTP) | 262K ctx, 31 GB host RAM, RSS ~3 GB: **193 tok/s** count-to-100 w/ MTP | [repo](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090) |
| **R9V / radiance (dual R9700)** | VRAM, RAM, **SSD fallback** | Fallback only; slower | Yes (gfx1201) | Yes (MTP4) | 131K ctx: **89.45 tok/s** median; static maps, host expert copy 40.3 GiB; "below ~160 GB combined … falls back to SSD" | [repo](https://github.com/drwolfen/radiance-vllm-qwen4exp) |
| **qwen4exp-5090 (llama.cpp fork)** | VRAM, RAM | No (GPU-resident expert cache) | Yes | Yes | 1×RTX 5090 + 128 GB RAM: **~84 tok/s @32K, ~62 @262K** on a 74 GB IQ3 | [repo](https://github.com/sergqwer/qwen4exp-5090) |
| **qwen3.8-flash-next-16gb** | VRAM, CPU/**SSD (mmap)** | Experts + PLE on CPU/SSD | Yes | No | 16 GB VRAM + 62 GB RAM, UD-IQ1_S: **~6 tok/s** decode, ~3.5 tok/s prefill | [repo](https://github.com/hocestnonsatis/qwen3.8-flash-next-16gb) |
| **MiaAI-Lab DGX-Spark** | VRAM/unified, **NVMe PLE** | No; PLE mmap | Yes (NVFP4) | Yes (MTP3) | 262K ctx: 48.7 tok/s single, 2,146 tok/s prefill @128K; 10 min 51 s to `/health` (checkpoint read) | [repo](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) |
| **halogen-flash-server (Strix Halo)** | Unified LPDDR5X + **SSD PLE** | No; FP8 n-gram table set aside | Yes | Yes (draft head) | 131K prefill **1,517 tok/s**; 46.0 tok/s decode @32K with draft, 34.1 serial greedy | [repo](https://github.com/peonist-ai/halogen-flash-server) |
| **win-Qwen4exp-rocm (Strix Halo)** | iGPU + RAM + **PLE mmap** | No; PLE pager | Yes (ROCmFP4) | Yes (embedded Qwen MTP) | no published throughput; ships `--tensor-read-lazy on` | [repo](https://github.com/jamesweiym-ops/win-Qwen4exp-rocm-Strix-Halo) |
| **llama.cpp SYCL backend** | VRAM, CPU | No | Yes (fused MoE since 2026.04-05) | via host common | Arc A770/A580, B580 tested; B70 target of bongo | [SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) |
| **OpenVINO GenAI** | CPU, GPU, NPU | No | Yes (INT4/8) | No | KV eviction + prefix caching; **no MoE expert offload or SSD weight streaming documented** | [repo](https://github.com/openvinotoolkit/openvino.genai) |

---

## 3. The SSD-streaming question, answered

### 3.1 Weight streaming from SSD per token: it exists, and it is the wrong tool for throughput

Three independent implementations, all measured or documented:

- **llama.cpp PR #25294** — "llama : stream MoE routed experts from disk" (open, unmerged). Mechanism:
  per-layer device cache of `n_slots` expert slabs; a CPU id-remap op after top-k; demand-load of misses by
  an async worker pool; **O_DIRECT** to bypass the page cache; "Wave-Partitional Prefill" that runs expert
  GEMMs in waves when a ubatch touches more experts than the cache holds. Measured on GB10 (Grace-Blackwell,
  128 GB unified, PCIe 4.0 SSD) with GLM-5.2-UD-Q2_K_XL (~254 GB file, 256 experts):
  **~1.83 tok/s decode at 73 % hit (64 slots ≈ 55 GB cache); ~2.20 tok/s at 79 % (90 slots ≈ 79 GB)**;
  decode latency 430–507 ms/token. Source: [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294).
- **AirLLM** streams a whole decoder module (or, for Kimi K3, an individual expert) from on-disk safetensors
  into the GPU right before it runs and evicts it right after, prefetching the next module on one worker
  thread. For Qwen3.8-Flash-Next specifically, "a whole MoE layer is a few GB of bf16; that is what streams
  today" — the packed 3D expert tensors are not individually hookable. Measured footprint 5.95 GB VRAM, but
  no published tok/s and the README names **disk loading** as the bottleneck; the only speed knob published
  is 4-bit on-disk compression ("up to 3x"). Sources:
  [`airllm_base.py`](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_base.py),
  [`airllm_kimi_k3.py`](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_kimi_k3.py),
  [`airllm_qwen4_exp.py`](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_qwen4_exp.py),
  [README](https://github.com/lyogavin/airllm).
- **FlexGen and DeepSpeed ZeRO-Inference** stream whole tensors/blocks from disk, tuned for throughput with
  very large batches. FlexGen's own table: OPT-175B all-disk **0.69 tok/s** vs 7.32 CPU vs 25.26 GPU
  ([FlexGen README](https://github.com/FMInference/FlexGen#generation-throughput-tokens)). ZeRO-Inference
  on LLAMA3-70B and 4×Gen4 NVMe: **7 tok/s**, rising to 17 (4×Gen5) and 26 (8×Gen5) as NVMe bandwidth rises
  ([DeepNVMe 06-2025](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/06-2025/README.md)).
- **PowerInfer-2** is the one system that treats flash as a first-class tier for *MoE* on a phone: neuron
  clusters + "segmented neuron cache" + a storage engine that pipelines cluster I/O with compute, reaching
  11.68 tok/s for a 47B model on a smartphone ([paper](https://arxiv.org/abs/2406.06282)). The enabling
  mechanism is exactly the one Strata and llama.cpp also use: **a bounded cache in front of the storage, and
  I/O overlapped with compute** — not raw random reads.

**Verdict.** Disk-streamed expert *weights* top out at ~2.2 tok/s in the best published MoE measurement, and
that measurement needed a 55–79 GB cache — more RAM than the whole reference box. For bongo's IQ2_XS target,
the pinned VRAM+RAM split already delivers 7.62 tok/s at 128K, so an SSD weight tier is a regression, not an
optimisation. It only becomes relevant if bongo moves to IQ3_XXS (39.97 GiB of experts) and cannot find
double-digit GiB of RAM for the CPU side.

### 3.2 The second shard (n-gram / PLE table): SSD-resident is the norm, not a risk

This is the answer to "can the second shard be on SSD at all times?" — **yes, and every qwen4exp deployment
already does it.** The table is 26.82 GiB in bongo's IQ2_XS GGUF and 51.2B parameters / ~26.8 GiB in the
other recipes; it is a *lookup*, not a matmul, so per token only ~16 rows are gathered and the rows are
addressed deterministically from the token id. Equivalent mechanisms measured elsewhere:

| System | Mechanism | Measured |
| --- | --- | --- |
| Strata | 320,001,536 × 90 B table, never in RAM; 16 × 4 KiB unbuffered reads/token, issued when the token id is known, collected before layer 1; bounded ~1M-row (~95 MB) cache | ~82 % of reads served from cache; 20–34 % of rows recur within one long prompt ([`ple_reader.hpp`](https://github.com/Niko1221/Strata), [`strata-ninfer-recon.md`](strata-ninfer-recon.md) §2.3) |
| 0xBakeer DGX-Spark | `-ot "per_layer_token_embd=CPU" -lm mmap`; table served by the OS page cache, never VRAM | 1.3 % cached / 13.1 major faults/token → 21.05 tok/s; 79 % / 2.1 faults → 22.40 tok/s; later re-measure: warming makes **no** difference (27.8 tok/s both) ([how-it-works](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/main/docs/how-it-works.md)) |
| tonyd2wild 4×3090 (vLLM) | FP8 n-gram table on NVMe, **16 rows per token read inside CUDA graphs** | 193 tok/s with MTP; 31 GB host RAM, process RSS ~3 GB ([repo](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090)) |
| llama.cpp `--lazy-mode` | "on-demand reading of certain tensors … read the rows from disk on demand instead of keeping them resident (requires mmap)" | mmap lazy vs direct reads on Strix Halo: pp512 181 → 401 tok/s, pp8192 274 → 451 ([arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp), [PR #29030](https://github.com/ggml-org/llama.cpp/pull/29030)) |
| R9V / radiance | derived PLE file 26.82 GiB, extracted from the target GGUF | "falls back to SSD residency for the PLE/expert tiers, which is slower" below ~160 GB combined ([repo](https://github.com/drwolfen/radiance-vllm-qwen4exp)) |
| AirLLM | ~102 GB bf16 table file-mmap'd on the host and gathered on CPU, never in VRAM or anonymous RAM | 5.95 GB VRAM total on RTX 4090 ([`airllm_qwen4_exp.py`](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_qwen4_exp.py)) |

The two instructive negatives: (1) the 0xBakeer author found the table does **not** self-warm to a useful
degree (rows are 3-gram-hash-selected and rarely repeat), and then found that warming did not matter once
graph capture hid the fault cost — so a row cache must be judged on the *held-out* workload, not on a warm-up
ritual; (2) mmap-based lazy reads were up to 2x slower than direct reads for prefill, so "mmap + page cache"
is not automatically the right reader.

### 3.3 Local measurement: raw 4 KiB random-read cost on the reference box

Because the PLE gather is 16 random 4 KiB pages per token, the raw device behaviour bounds the design. Full
method and raw data: [`bench/results/2026-09-28-ssd-random4k-probe/`](../../bench/results/2026-09-28-ssd-random4k-probe/README.md).

| Mode | IOPS | Bandwidth | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 thread, O_DIRECT 4 KiB (QD1) | 1,819 | 7.5 MB/s | 358 µs | 1,296 µs | 1,753 µs |
| 8 threads, O_DIRECT 4 KiB (QD8) | 15,213 | 62.3 MB/s | 450 µs | 1,009 µs | 1,439 µs |
| Sequential O_DIRECT (1M/4M) | — | ~1.0–1.7 GB/s | — | — | — |

At QD1, 16 rows cost ~5.7 ms exposed; at QD8 ~1.05 ms of wall time. Both fit inside a 131 ms decode step
(current 7.62 tok/s at 128K) **only** if the gather is overlapped with embedding + layer 0, which is exactly
what Strata's split API does. Note the ~60x gap between random 4 KiB (~7.5 MB/s) and sequential (~1.3 GB/s):
reading *dense weights* from SSD is bounded by the sequential number, reading *sparse rows* by the random one.

---

## 4. Per-system notes (the non-obvious parts)

### llama.cpp — the offload work that matters to bongo

The shipped controls are `--cpu-moe` / `--n-cpu-moe N` (keep all, or the first N layers', MoE weights on the
CPU), `-ot` (per-tensor buffer overrides), `--load-mode` (`auto`, `none`, `mmap`, `mlock`, `mmap+mlock`,
`dio`), and `--lazy-mode` (`on`/`auto`/`off`, on-demand row reads of large tensors, requires mmap), all in
[`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp). Everything below is an
**open draft PR or issue**, i.e. an indication of what upstream is converging on, not a supported feature.

- **#25294 — stream routed experts from disk** (open). The one true SSD-expert system for llama.cpp;
  measured ~2 to 2.2 tok/s decode on a 254 GB model. Bigger cache → better hit rate (73 % → 79 %) and a
  small decode gain. Uses O_DIRECT because "the page cache cannot help and otherwise thrashes".
- **#27861 — GPU-resident LRU cache for host-offloaded experts** (open). The important measurement: on
  Qwen3.8-Flash-Next UD-Q4_K_XL with 28 expert layers pinned to host by `-ot`, a **static** top-32 hot list
  learned on half the workload covers only ~10 % of the other half (uniform 6.2 %) — "static pinning of
  experts is a dead end" — while a per-layer **dynamic LRU** hits 67 % (64 slots) / 81 % (128 slots) and
  moves decode **18.4 → 24.2 tok/s (+31 %)** at 48 slots/layer (~4.1 GiB VRAM), with uploads throttled and
  asynchronous. Decode-only (`n_tokens == 1`); prefill untouched.
- **#26414 — `--pin-hot-experts N`** (open). Uses `mlock()` to stop the OS evicting mmap'd experts; pairs
  with `--load-mode mmap+pin`. Relevant when the model exceeds RAM and relies on the page cache.
- **#28414 — `--prefetch-experts-slots N`** (open). 1-deep lookahead H2D of host-resident experts on a
  second stream, into rotating staging buffers, for prefill. RTX 5070 Ti, 24B-A3B `-ncmoe 20`, ~42K prompt:
  TTFT 15.71 → 13.97 s (**−11 %**), byte-identical, decode flat. This is the "overlap budget" lever for the
  serial host→device expert copies that [issue #25859](https://github.com/ggml-org/llama.cpp/issues/25859)
  shows leave the GPU idle during offloaded-MoE prefill.
- **#29030 — lazy tensor rows with direct reads** (open). Replaces mmap-backed `-lzm on` with direct reads
  for the qwen4exp/gemma4 PLE-style gather; up to **2x prefill** on Strix Halo.
- **#27742 — qwen4exp model support**, merged 2026-08-27. This is the architecture support bongo already
  relies on.
- Related feature requests: [#27562 JIT expert streaming from storage](https://github.com/ggml-org/llama.cpp/issues/27562),
  [#29130 O_DIRECT for mmap'd experts](https://github.com/ggml-org/llama.cpp/issues/29130),
  [#26448 host RAM via PCIe DMA](https://github.com/ggml-org/llama.cpp/issues/26448),
  [#27584 bandwidth-adaptive CPU–GPU co-execution + global LRU](https://github.com/ggml-org/llama.cpp/issues/27584).

### Strata (recap; full detail in the recon)

Placement is an arithmetic plan, not flags; the expert cache is a **static** frequency table that is
**not wired into compute** (`--expert-cache` defaults 0); the headline 52 tok/s comes from MMQ (weights stay
quantized), overlapped streaming of non-resident experts, and a suffix/ngram drafter. Its PLE reader is the
reference for the second shard. See [`strata-ninfer-recon.md`](strata-ninfer-recon.md) §2 and
[Niko1221/Strata](https://github.com/Niko1221/Strata).

### KTransformers and IPEX-LLM FlashMoE — RAM-pinned, quality-first

Both place experts in system DRAM and use CPU kernels (AMX/AVX for KTransformers; SYCL i-quant GGUF for
FlashMoE). They are the existence proof that a 671B model can serve from 14 GB VRAM — but they need 380 GB
of RAM to do it, and bongo has 30 GiB. Their transferable ideas are the **placement strategies**
(`frequency`, `uniform`, `front-loading`, `random`, plus dynamic update) in
[experts-sched](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md),
and FlashMoE's `numactl --interleave=all` / SNC advice for dual-socket hosts.

### MegaBlocks / DeepSpeed-MoE (dense-kernel reference)

Not an offload system, but the reference for MoE *kernels*: block-sparse expert GEMMs that never drop tokens
([MegaBlocks, arXiv 2211.15841](https://arxiv.org/abs/2211.15841); [DeepSpeed-MoE, arXiv 2201.05596](https://arxiv.org/abs/2201.05596)).
Relevant to the SYCL kernel gap, not to placement.

### HyperQwen / Splash — the "one model, tuned kernels" pattern

Both are model-specific engines: HyperQwen patches vLLM for one Qwen checkpoint on one card class and wins
on KV (KVarN 4/2-bit) and speculation (DFlash2/MTP); Splash compiles per-model kernels and adds an **SSD
tier for KV pages and GDN state** (`--max-cache-disk`, [PR #3](https://github.com/incoai/splash/pull/3),
merged 2026-09-26). Neither streams weights from SSD. Their lesson for bongo: the biggest wins in this model
family come from KV/state management and speculation, not from where experts live.

---

## 5. Tradeoff matrix for bongo

Prices use the measured reference-box numbers. "CPU misses" = experts the GPU does not hold are computed on
the CPU (Strata's model); "SSD tier" = cold experts are read from disk (llama.cpp #25294's model).

| Strategy | VRAM | RAM | SSD | Measured/expected 128K decode | Verdict for bongo |
| --- | --- | --- | --- | --- | --- |
| **A. Pin all experts VRAM+RAM** (current `--n-cpu-moe 16`): 22.40 GiB GPU + 10.62 GiB RAM | 29.26 GiB used of 31.92 | 10.96 GiB | none | **7.62 tok/s** (measured) | Baseline; correct today. All-GPU (`n=0`) needs 33.02 GiB > 31.92 and does not load. |
| **B. RAM-pinned hot set + CPU misses** (Strata) | tunable | needs *all* experts resident | none | Strata 52 tok/s on RTX 5070 + **64 GB** RAM | bongo cannot hold 33 GiB of experts in 30 GiB RAM **and** leave room for PLE page cache; only 10.6 GiB fits. So bongo is strictly harder than Strata here. |
| **C. A + dynamic VRAM LRU over the RAM-resident experts** (llama.cpp #27861) | +4.1 GiB for 48 slots/layer | same 10.6 GiB | none | **+31 %** measured on the same model (18.4→24.2, 2×3090) | The most promising single lever — **but** at `n=16` only ~2.6 GiB of VRAM is free. To fund an LRU, GPU expert residency must drop (e.g. `n=24`), so the gain must beat the ~1 % static decode lost. Needs a local A/B. |
| **D. SSD cold experts** (llama.cpp #25294) | small per-layer slab cache | small | ~55–79 GB of cache for 73–79 % hit | **~2.2 tok/s** at 79 % hit (GLM-5.2, GB10) | Only for a model that does not fit at all (IQ3_XXS experts = 39.97 GiB). Far below A. |
| **E. Higher tier (IQ3_XXS) with E as safety net** | 19.96 GiB GPU (27.71 GiB total) | 20.01 GiB CPU experts → leaves ~3.5–6 GiB PLE page cache | fallback if RAM pressure bites | not measured at 128K | At risk: the recon already flags ~3.5–6 GiB of page cache as the danger zone. Try only after R5 measures the PLE miss cost. |
| **F. PLE on SSD (all strategies)** | never | page cache only | 26.82 GiB always | near-free if overlapped; **+2x prefill** with direct reads | Adopt regardless: `-ot` the PLE tensor to CPU + mmap, or `--lazy-mode`; prefer direct reads over mmap faults. |

---

## 6. Ranked recommendation — what to mine for each subsystem

1. **Placement (highest expected value).** Mine **llama.cpp #27861** (dynamic LRU over host-offloaded
   experts) plus **#26414** (`mlock` the hot set) and **#28414** (lookahead H2D prefetch for prefill).
   The decisive design point is that the residency policy must be **online and temporally local**, not an
   offline frequency profile: the only held-out measurement in this model family says static ranking recovers
   ~10 % vs 6.2 % uniform, while an LRU recovers 67–81 %. Treat Strata's `profile.bin` as a feasible
   *initialisation*, not the policy, and require a held-out hit-rate measurement before wiring it in.
2. **Kernels (the real 6–8x gap).** Mine **Strata's MMQ / quantized-weight MoE matmul** and its shared
   scratch buffers / chunked prefill. Map onto llama.cpp's SYCL backend, which already has "Fused MoE" and
   `MUL_MAT_ID` ([SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)), rather
   than writing a new backend. MegaBlocks/DeepSpeed-MoE are the kernel-design reference.
3. **Speculation (cheapest multiplier).** Mine **Strata's suffix/ngram drafter** (trigram lookup, no weights,
   no GPU, 1.6–1.8x), which is already proven on this exact model by
   **0xBakeer** (88 tok/s on file rewrite) and by llama.cpp's `--spec-type ngram-mod` (43 → 62 tok/s on
   IQ4_XS, [4×3090 README](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090)). bongo's MTP head is
   dropped from the GGUF, so this is the only available speculation path.
4. **SSD I/O (only for the second shard).** Mine **Strata's PLE reader**: issue the 16 reads as soon as the
   token id is known, collect before layer 1, keep a bounded row cache, and prefer direct reads over mmap
   faults for prefill (llama.cpp #29030). Adopt **0xBakeer's** `-ot per_layer_token_embd=CPU -lm mmap`
   pattern, and do **not** warm the table as a boot ritual.
5. **Explicitly reject for now.** SSD-resident expert *weights* (strategy D) as a throughput optimisation;
   PowerInfer's neuron predictor (it needs a retrained ReLU model — the published GGUF is not one); and
   IPEX-LLM FlashMoE / KTransformers sizing, which assume 128–380 GB RAM.

---

## 7. Residual uncertainty and the experiments that would resolve it

These are the claims this survey could **not** settle from published sources, with the measurement that
would close each one. They overlap with R4 (expert activation on the reference box) and R5 (SSD/PLE shard);
those own the full benchmarks, and this survey should not pre-empt them.

| Open question | Why it matters | Experiment to resolve | Expected owner |
| --- | --- | --- | --- |
| Does a dynamic LRU over bongo's 10.62 GiB of RAM-resident experts beat the static `n=16` split on the **Arc B70 / Vulkan or SYCL**, at 128K? | #27861's +31 % is measured on 2×3090 CUDA with UD-Q4_K_XL; bongo has less free VRAM, a different backend and IQ2_XS. The static-vs-dynamic result is model-family-general but not box-general. | Port/rebase the #27861 remap idea or emulate it with `-ot` + a host-side slot table; A/B `n=16` vs `n=24`+LRU at 4K and 128K; record VRAM/RAM peaks. | R4 / engine task |
| What is the **expert routing working-set** on this exact model (IQ2_XS, 512 experts × 48 layers, 10/token)? | Determines the right cache size and whether an LRU can fit in the ~2.6 GiB of VRAM left at `n=16`. Must be held-out, not in-sample. | Instrument `ffn_moe_topk`/router outputs over a mixed held-out workload; report LRU-k hit curves like #27861. | R4 |
| What is the **PLE 16-row read cost and row-cache hit rate** on this box, cold and warm, and does overlaying it on embedding+layer 0 hide it? | The 4K probe bounds the device (p50 358 µs QD1), but mmap vs direct, page-cache hit, and overlap are unmeasured here. | R5's shard benchmark; reuse [`rand4k.py`](../../bench/results/2026-09-28-ssd-random4k-probe/rand4k.py) for the device floor. | R5 |
| Does `--lazy-mode` (or `-ot per_layer_token_embd=CPU` + mmap) actually reduce bongo's RSS/VRAM and change 128K throughput? | The PLE is 26.82 GiB > VRAM, so it must be CPU-side; llama.cpp's `--lazy-mode` is already in the pinned build. | Config-pin A/B at 4K and 128K with `-lzm off/auto/on`, RSS and VRAM peaks, needle. | R5 / engine task |
| Do the merged/open MoE-offload PRs (#25294, #27861, #26414, #28414, #29030) apply cleanly to the bongo-pinned commit and build on SYCL/Vulkan? | None are merged upstream; bongo pins `b11223`. | `git apply --check` each patch against the pin; note conflicts; build only the ones that matter. | engine task |

---

## Appendix — raw data and reproduction

- Random-4K + sequential NVMe probe: [`bench/results/2026-09-28-ssd-random4k-probe/`](../../bench/results/2026-09-28-ssd-random4k-probe/README.md)
  (`rand4k.py`, `raw/rand4k.out`, method and results).
- bongo accounting used above: [`gguf-inventory.md`](gguf-inventory.md) §3–§6,
  [`expert-placement.md`](expert-placement.md) §Results / "A VRAM correction".
- Strata/NInfer prior recon: [`strata-ninfer-recon.md`](strata-ninfer-recon.md).
- Upstream sources are cited inline by URL; nothing upstream is vendored.
