# Research: running Qwen3.8-Flash-Next on an Intel Arc Pro B70

Status: first pass, 2026-09-27. Owner: CTO ([BAS-48](/BAS/issues/BAS-48)).
Everything below is either **verified on the reference machine / upstream source** or marked **(estimate)**.

Primary objective: a one-command setup that downloads a 125B MoE model and serves it over an
OpenAI-compatible endpoint with **>=128K context**, on **32 GB VRAM + 32 GB RAM + SSD**, as fast as the
hardware allows.

---

## 1. Reference machine (verified)

The bongo target box is the same host the agents run on, so we can build and measure on the real GPU.

```
CPU    AMD Ryzen 7 9700X (8C/16T, AVX-512)
RAM    30 GiB total (~23-24 GiB available with the agent stack running)
GPU    Intel Corporation Battlemage G31 [Arc Pro B70]  (PCI 8086:E223, subsystem Sparkle 0105)
        lspci BAR: "Memory at f000000000 (64-bit, prefetchable) [size=32G]"  -> 32 GB VRAM
Driver xe (kernel 7.0.13, Fedora 44 Server); render node /dev/dri/renderD128
Disk   / has ~296 GB free on the model volume (SSD)
Missing compute stack (must be installed): Level Zero, oneAPI/icpx, clinfo, vulkaninfo -- none present yet
```

Implications:

- We do **not** need a separate benchmark machine; child tasks can build, run, and measure here.
- The one-command setup must provision the Intel compute stack itself. Nothing Intel-GPU-specific is installed.

## 2. The model (verified from HF metadata / config)

`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, base `Qwen/Qwen3.8-Flash-Next`.

| Property | Value |
| --- | --- |
| Architecture (`config.architectures`) | `Qwen4ExpForConditionalGeneration`, GGUF arch `qwen4exp` |
| Layers | 48 = 36 `linear_attention` + 12 `full_attention` (every 4th) |
| Hidden size | 2560 |
| Attention | 24 Q heads, **2 KV heads**, head_dim 256, partial rotary (0.25), rope_theta 1e7 |
| Vocabulary | 248,320; max position embeddings **262,144** |
| MoE | **512 experts/layer**, 10 active, `moe_intermediate_size` 640 |
| Total experts | 512 x 48 = **24,576** (matches Strata's figure) |
| Params/expert | 3 x 2560 x 640 = **4.92 M** |
| MTP | 1 NextN layer, hybrid |
| N-gram | `ngram_vocab_size_base` 20,000,000, `heads_per_ngram` 8, `ngram_size` 3 |
| Vision | 27-layer ViT, 0.91 GB `mmproj` BF16 projector (optional) |
| Tensors | 1,224 |

### Quant tiers (verified file sizes, decimal GB)

| Tier | Combined GGUF | Shard 1 | Shard 2 | Expert bytes (approx) | Dev KLD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2_0 (experimental) | 66.55 GB | 39.80 | 26.75 | ~34 GB | 0.424 |
| IQ2_XS (recommended) | 68.15 GB | 39.79 | 28.36 | ~36 GB | 0.341 |
| IQ3_XXS | 75.97 GB | 39.79 | 36.18 | ~43 GB | 0.240 |

Notes:

- Shard 1 is almost identical across tiers (~39.8 GB); the tier difference lands in shard 2. The exact
  tensor-to-shard mapping is an **open question** to resolve from the GGUF tensor inventory before we finalise
  buffer placement (cheap: HTTP range reads of the GGUF header, no full download).
- The model is gated behind the Swift Open License 1.0; check access before scripting the download.
- The model is licensed for the weights only; redistribution of the runtime is separate.

## 3. Memory budget (estimate, to be replaced by measurements)

Per the config, only 12 full-attention layers hold a KV cache; the other 36 layers are linear attention with a
constant-size recurrent state.

**KV cache per token** (K+V, f16): `12 layers x 2 kv_heads x 256 head_dim x 2 bytes x 2 (K,V) = 24 KB/token`

| Context | KV f16 | KV q8_0 | + indexer KV (est.) | + linear state (constant) |
| ---: | ---: | ---: | ---: | ---: |
| 128K | 3.1 GB | ~1.6 GB | +0.4 GB | ~0.15 GB f32 |
| 262K | 6.3 GB | ~3.2 GB | +0.8 GB | ~0.15 GB |

**Working budget for 32 GB VRAM + 30 GiB RAM (estimate):**

| Bucket | VRAM | System RAM |
| --- | ---: | ---: |
| OS / agent stack / process overhead | ~0.5 GB | ~3-4 GB |
| Non-expert weights (attn, linear attn, embed/head, routers, shared experts, MTP) | ~3-5 GB | - |
| N-gram table (~29 GB) | - | disk-resident (OS page cache only) |
| KV cache + linear state @128K, q8 | ~2 GB | ~0.2 GB |
| Compute / graph buffers | ~1-2 GB | - |
| **Available for experts** | **~24 GB** | **~26 GB** |

So the combined expert capacity is roughly **50 GB**, which is *more* than IQ3_XXS's ~43 GB of experts. The
constraint is not total capacity but **placement**: VRAM must hold the hot experts, RAM the rest, and the CPU
must compute whatever is not resident on the GPU. That is exactly the problem Strata solves with an adaptive
expert cache, and it is bongo's core engineering problem.

Unknowns that matter and must be measured:
- Real non-expert weight bytes (from the GGUF inventory).
- Whether the ~29 GB n-gram table can be mmap'd from SSD and read a few rows/token without wrecking latency.
- Whether `xe` + Level Zero exposes enough VRAM for a 4 GB+ single allocation (llama.cpp SYCL note: "Support
  malloc memory on device more than 4GB" landed 2025.11).

## 4. How Strata does it (the thing to beat)

From Strata's README and `docs/DETAILS.md` (NVIDIA, CUDA):

- **GPU**: attention + linear-attention mixers, gated residual weights, routers, shared experts, output head,
  MTP draft layer, KV cache (with KV streaming at >=64K), and an **adaptive expert cache** filling remaining
  VRAM with the most-used experts, adapting during conversation.
- **RAM**: all 24,576 experts pinned; the CPU computes non-cached experts in place, in parallel with the GPU.
- **SSD**: a ~28.8 GB n-gram lookup table, read a few rows/token through the OS page cache.
- **Speculation**: the model's own MTP head drafts up to 3 tokens per pass; plus n-gram/prompt lookup.
- **Prompt**: processed in 2,048-token chunks with experts streamed to VRAM over PCIe.
- Measured on RTX 5070 12 GB / 64 GB RAM: IQ2_XS 78 prompt / 78 output tok/s at 4K; 472 / 52 at 128K.
- Strata explicitly states **32 GB of RAM alone is not enough** for any tier; its design assumes all experts in
  RAM. bongo must instead treat VRAM+RAM as one ~50 GB expert pool.

What bongo can reuse as *ideas*: tiered expert placement, adaptive per-conversation residency, MTP speculation,
KV streaming, SSD-resident lookup tables. What it cannot reuse: the CUDA kernels (all of `src/kernels/cuda`).

## 5. Intel Arc runtime options

| Option | i-quants (IQ2_XS/IQ3_XXS) | MoE CPU offload | MTP | Effort | Notes |
| --- | --- | --- | --- | --- | --- |
| **llama.cpp SYCL** | Yes (IQ1-IQ4 in GPU dequant since 2024) | `--cpu-moe`, `--n-cpu-moe N`, `-ot` | Yes (`n_layer_nextn`, MTP context type) | Low | Primary backend. Intel-maintained, `qwen4exp` supported upstream. oneAPI 2025.3.3 recommended. |
| **llama.cpp Vulkan** | Partial | `-ot` | Yes (same core) | Low | Mesa ANV on `xe`; weaker/fewer kernels than SYCL. Fallback only. |
| **IPEX-LLM (prebuilt llama.cpp SYCL)** | Yes | Yes (fork) | fork-lag | Low | Zero-build path; needs oneAPI runtime libs. Good for a first boot, not for our own optimisations. |
| **OpenVINO GenAI / OVMS** | No (own IR/INT4) | N/A | No | Med | Cannot run the published i-quant GGUF without re-conversion. Rejected for v1. |
| **vLLM XPU** | No (GPTQ/AWQ, datacenter-first) | N/A | partial | High | Not aimed at Arc desktop, no i-quant. Rejected for v1. |
| **Custom SYCL engine (port Strata)** | Yes | full control | Yes | Very high | The end state if llama.cpp is too slow. Months of work; gate on measured gaps. |

### Verified upstream facts

- llama.cpp master has `LLM_ARCH_QWEN4EXP` (`"qwen4exp"`) in `src/llama-arch.cpp`, a `llama_model_qwen4exp`
  constructor, and MTP/NextN support (`n_layer_nextn`, `LLAMA_CONTEXT_TYPE_MTP`). (checked 2026-09-27)
- SYCL backend docs list Arc B-Series as supported (B580 verified) and Fedora among tested Linux distros.
- llama.cpp `llama-server` supports `--cache-type-k/-v` (q8_0, q4_0, ...), `-ot/--override-tensor`,
  `--cpu-moe`, `--n-cpu-moe N`, `--n-cpu-ffn N`, and `--spec-draft-*` flags for a draft/MTP model.
- SYCL 2026.04-05 release notes add "Fused MoE" and K-quant reorder optimisations.

## 6. Recommendation

Ship a **pinned llama.cpp SYCL baseline first**, then invest in an expert-placement layer, and only then
consider a custom engine. Rationale and alternatives are in
[ADR-0001](adr/0001-runtime-architecture.md); the baseline engine choice is [ADR-0002](adr/0002-baseline-engine.md).

## 7. Open questions

1. Exact tensor inventory / per-tensor byte sizes (resolve with GGUF header range reads).
2. Arc Pro B70 memory bandwidth and Xe-core count (product page not retrievable from this host; read from
   `xpu-smi`/Level Zero once installed).
3. Does the published IQ3_XXS fit the ~50 GB expert pool with 128K KV on this box? (measure)
4. SYCL vs Vulkan on Battlemage for this arch and these quant types (measure both on a small model first).
5. N-gram table: mmap vs explicit SSD streaming; does the OS page cache keep up at token rate? (measure)
6. Model license / access conditions for scripted download.
