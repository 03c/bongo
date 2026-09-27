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
| MTP | Base config has a **1-layer hybrid NextN head** (31 `mtp.*` tensors, 4.86 GiB BF16) and the Swift checkpoint keeps it; it is **dropped by the GGUF conversion and not wired into llama.cpp `qwen4exp`** — speculation must use the n-gram/PLE path (§2.1) |
| N-gram | 16 heads over `ngram_vocab_size_base`-scale vocab, `heads_per_ngram` 8, `ngram_size` 3; `per_layer_token_embd.weight` = 26.82 GiB in shard 1 ([gguf-inventory.md](gguf-inventory.md) §3) |
| Vision | 27-layer ViT, 0.91 GB `mmproj` BF16 projector (optional) |
| Tensors | 1,224 |

### Quant tiers (verified file sizes, decimal GB)

| Tier | Combined GGUF | Shard 1 | Shard 2 | Expert bytes (measured) | Dev KLD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2_0 (experimental) | 66.55 GB | 39.80 | 26.75 | 33.97 GB | 0.424 |
| IQ2_XS (recommended) | 68.15 GB | 39.79 | 28.36 | 35.45 GB | 0.341 |
| IQ3_XXS | 75.97 GB | 39.79 | 36.18 | 42.91 GB | 0.240 |

Notes:

- Shard 1 is almost identical across tiers (~39.8 GB) because it holds the **n-gram table** plus layers
  0..12 and all global tensors; the tier difference lands in shard 2. The tensor-to-shard mapping is now
  **resolved** — see [gguf-inventory.md](gguf-inventory.md) §2. In short, the split lands *inside* layer 13
  (IQ2_XS, Q2_0) or layer 12 (IQ3_XXS), and no tensor crosses a shard boundary.
- The **n-gram table is 28,800,138,240 B (26.82 GiB / 28.80 GB)** in every tier, is named
  `per_layer_token_embd.weight`, is `IQ4_NL`, and lives in shard 1. [gguf-inventory.md](gguf-inventory.md)
  §3 has the details; use that figure, not the earlier ~29 GB estimate.
- Expert bytes above are measured from the GGUF tensor table, not estimated. The earlier column said
  `~34/36/43 GB`; the real values are `33.97/35.45/42.91 GB`.
- **No MTP / NextN tensor is present in any of the three GGUFs.** All 1,224 tensors are `blk.0`-`blk.47`
  plus globals, and no name matches `nextn|mtp|draft|eagle`, and there is no `qwen4exp.nextn_predict_layers`
  key. This is not a packaging accident: the base model *and* the Swift checkpoint do carry an MTP head,
  but llama.cpp's `qwen4exp` conversion and runtime do not support one. Speculation is therefore scoped to
  the n-gram/PLE path. The full evidence, the re-conversion cost, and the runtime limits are in **§2.1**;
  see also [gguf-inventory.md](gguf-inventory.md) §4/§7.
- The model is gated behind the Swift Open License 1.0 on the Hub page, but the repo currently answers
  anonymous range reads; access conditions for scripted download still need checking.

### 2.1 The MTP / NextN head — present in the weights, absent from the GGUF and the runtime

Short answer to "does this model have an MTP head we can use?": **the weights have one, bongo cannot use
it.** The base model and the Swift checkpoint both ship a 1-layer NextN/MTP head, but the published GGUF
conversion drops it and the llama.cpp `qwen4exp` implementation has no MTP path. Speculation on bongo is
re-scoped to the **n-gram/PLE** path (`per_layer_token_embd.weight`, §3), which is already in the GGUF and
already lazy-read by llama.cpp.

**Weight claim (verified 2026-09-27 from Hub metadata and safetensors headers, no full download):**

| Source | Finding |
| --- | --- |
| `Qwen/Qwen3.8-Flash-Next/config.json` | `mtp_num_hidden_layers: 1`; `mtp: {hybrid: true, layer_types: ["full_attention"], num_hidden_layers: 1, mtp_use_hidden_state_from_layer: null, rope_theta: 1e7}`; `mtp_use_dedicated_embeddings: false`. `architectures` stays `Qwen4ExpForConditionalGeneration` — the MTP head is a config block, not a separate architecture. |
| `Qwen/.../README.md` | "MTP: 1 layer, trained with multi-steps"; parameter table: "125B with 6B activated, plus 51B n-gram embedding and 4B MTP". |
| `ukisai/Swift-Qwen3.8-Flash-Next/config.json` | Byte-identical to the Qwen base config (MTP block included). |
| `ukisai/Swift-Qwen3.8-Flash-Next/model.safetensors.index.json` | 1,658 tensors, of which **31 are named `mtp.*`**. Its `weight_map` and `metadata` are identical to `Qwen/Qwen3.8-Flash-Next`'s index. |
| `ukisai/Swift-Qwen3.8-Flash-Next/README.md` | "The checkpoint includes the base model's one-layer MTP head." It also shows vLLM (`--speculative-config '{"method":"mtp",...}'`) and SGLang (`--speculative-algorithm NEXTN ...`) usage — i.e. the head is real and usable *in those engines*. |
| MTP tensor sizes (safetensors headers, HTTP range reads) | 31 BF16 tensors, **5,214,301,696 B = 4.86 GiB**, dominated by the MoE experts (`gate_up_proj` 3.13 GiB, `down_proj` 1.56 GiB). |

The public `ukisai/Swift-1.5-Qwen3.8-Flash-Next` repo is gated (HTTP 401 for anonymous reads); the BF16
model the GGUF card names as `base_model` is the public `ukisai/Swift-Qwen3.8-Flash-Next`.

**The published GGUFs drop it (verified by re-reading the headers):**

- 1,224 tensors, all `blk.0`-`blk.47` plus globals; no name matches `nextn|mtp|draft|eagle`.
- `qwen4exp.block_count = 48`; there is **no `qwen4exp.nextn_predict_layers` key** in the metadata.
- `recipe-summary.json` and `evaluation/report.json` make no MTP/NextN claim.

**Runtime claim (llama.cpp master, checked 2026-09-27):**

- llama.cpp *does* support MTP in general: `n_layer_nextn`, `LLAMA_CONTEXT_TYPE_MTP`, the
  `blk.%d.nextn.*` tensor names, and a `draft-mtp` speculative implementation that even auto-discovers an
  MTP sidecar next to a draft repo.
- **`qwen4exp` is not wired into any of it:**
  - `conversion/qwen4exp.py` sets `supports_mtp_export = False` and `no_mtp = True`, commented "the MTP
    block is a separate draft head; vLLM drops it too". The shared `_QwenMtpMixin` (`conversion/qwen.py`)
    that *does* remap `mtp.*` to the standard `blk.<n>.nextn.*` naming is inherited by Qwen3-Next and
    Qwen3.5/3.6, **not** by `Qwen4ExpTextModel`.
  - `gguf-py/gguf/constants.py`'s `MODEL_ARCH.QWEN4EXP` tensor list contains **no** `NEXTN_*` entries.
  - `src/models/qwen4exp.cpp` has **zero** references to `mtp`, `nextn`, `draft`, `ctx_type` or
    `n_layer_nextn`: the constructor creates no nextn tensors and the graph builder has no MTP branch.
  - `common/speculative.cpp`'s `common_speculative_impl_draft_mtp` lists its supported variants
    ("step35", "qwen35 / qwen35moe") — not `qwen4exp`.

**Consequence, and what a re-conversion would cost.** `--mtp` / `--no-nextn` are explicitly rejected for
`qwen4exp`, so upstream llama.cpp cannot even produce an MTP GGUF for this model, and a hand-built one
would not run: the arch has no NextN tensor table, tensor creation, or MTP graph. Using the head would
require *both* (a) converter work (add `NEXTN_*` to the `QWEN4EXP` tensor list and adapt the
`_QwenMtpMixin` remap to the qwen4exp checkpoint naming) and (b) runtime work (nextn tensor creation and
an MTP graph/context branch in `src/models/qwen4exp.cpp`). That is an engine feature, not a configuration
flag, so it is **out of scope for M0/M1 and for the Stage 1 expert-placement layer**; it belongs in a
Stage 2 decision. The weight cost if it were built: 4.86 GiB in BF16, or roughly ~0.7 GiB at an
IQ2_XS-class quantisation (one layer of expert bytes — compare [gguf-inventory.md](gguf-inventory.md)
§6.3), plus the draft KV.

## 3. Memory budget (estimate, to be replaced by measurements)

Per the config, only 12 full-attention layers hold a KV cache; the other 36 layers are linear attention with a
constant-size recurrent state.

**KV cache per token** (K+V, f16): `12 layers x 2 kv_heads x 256 head_dim x 2 bytes x 2 (K,V) = 24 KB/token`

| Context | KV f16 | KV q8_0 | + indexer KV (est.) | + linear state (constant) |
| ---: | ---: | ---: | ---: | ---: |
| 128K | 3.1 GB | ~1.6 GB | +0.4 GB | ~0.15 GB f32 |
| 262K | 6.3 GB | ~3.2 GB | +0.8 GB | ~0.15 GB |

**Working budget for 32 GB VRAM + 30 GiB RAM (non-expert weight figures are now measured):**

| Bucket | VRAM | System RAM |
| --- | ---: | ---: |
| OS / agent stack / process overhead | ~0.5 GiB | ~3-4 GiB |
| Driver + graph/compute buffers | ~2.0 GiB | - |
| Non-expert weights (attn, linear attn, embed/head, routers, shared experts, hyper-connections) | 3.62 GiB (IQ2_XS) | - |
| N-gram table (26.82 GiB) | - | disk-resident (OS page cache only) |
| KV cache + linear state @128K, q8 | ~1.8 GiB | ~0.2 GiB |
| **Available for experts** | **~22.4 GiB** | **~24 GiB** |

So the combined expert capacity is roughly **46 GiB**, which is *more* than IQ3_XXS's 39.97 GiB of experts.
The constraint is not total capacity but **placement**: VRAM must hold the hot experts, RAM the rest, and the
CPU must compute whatever is not resident on the GPU, *while the n-gram table keeps working out of a page
cache that has to share the same 30 GiB*. The full `-ot` / `--n-cpu-moe` rule and the resulting VRAM/RAM
split are in [gguf-inventory.md](gguf-inventory.md) §6.

**No MTP draft layer is in this budget.** The published GGUFs have no MTP head and llama.cpp `qwen4exp`
cannot convert or run one (§2.1), so the "non-expert weights" bucket is trunk weights only. If the MTP head
ever becomes usable it would add one layer of GPU weights (4.86 GiB in BF16; ~0.7 GiB at IQ2_XS-class
quantisation) plus its draft KV, and the §2.1 re-conversion/runtime work would have to land first.

Unknowns that matter and must be measured:
- Driver/graph buffer use and the real KV/indexer/state footprint (decides `--n-cpu-moe N`).
- Whether the 26.82 GiB n-gram table can be mmap'd from SSD and read 8-16 rows/token without wrecking
  latency; a random 90-byte row costs a whole 4 KiB page, so page traffic is the metric to watch.
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
  Strata controls its own weight conversion and engine, so it can keep the MTP head. bongo cannot: the
  published GGUF drops the head and llama.cpp `qwen4exp` has no MTP path (§2.1).
- **Prompt**: processed in 2,048-token chunks with experts streamed to VRAM over PCIe.
- Measured on RTX 5070 12 GB / 64 GB RAM: IQ2_XS 78 prompt / 78 output tok/s at 4K; 472 / 52 at 128K.
- Strata explicitly states **32 GB of RAM alone is not enough** for any tier; its design assumes all experts in
  RAM. bongo must instead treat VRAM+RAM as one ~50 GB expert pool.

What bongo can reuse as *ideas*: tiered expert placement, adaptive per-conversation residency, KV
streaming, SSD-resident lookup tables, and self-speculation as a *goal* — but sourced from the n-gram/PLE
path, not from an MTP head (§2.1). What it cannot reuse: the CUDA kernels (all of `src/kernels/cuda`).

## 5. Intel Arc runtime options

| Option | i-quants (IQ2_XS/IQ3_XXS) | MoE CPU offload | MTP | Effort | Notes |
| --- | --- | --- | --- | --- | --- |
| **llama.cpp SYCL** | Yes (IQ1-IQ4 in GPU dequant since 2024) | `--cpu-moe`, `--n-cpu-moe N`, `-ot` | **No for `qwen4exp`** (§2.1) | Low | Primary backend. Intel-maintained, `qwen4exp` supported upstream. oneAPI 2025.3.3 recommended. |
| **llama.cpp Vulkan** | Partial | `-ot` | **No for `qwen4exp`** | Low | Mesa ANV on `xe`; weaker/fewer kernels than SYCL. Fallback only. |
| **IPEX-LLM (prebuilt llama.cpp SYCL)** | Yes | Yes (fork) | fork-lag | Low | Zero-build path; needs oneAPI runtime libs. Good for a first boot, not for our own optimisations. |
| **OpenVINO GenAI / OVMS** | No (own IR/INT4) | N/A | No | Med | Cannot run the published i-quant GGUF without re-conversion. Rejected for v1. |
| **vLLM XPU** | No (GPTQ/AWQ, datacenter-first) | N/A | partial | High | Not aimed at Arc desktop, no i-quant. Rejected for v1. |
| **Custom SYCL engine (port Strata)** | Yes | full control | Only by porting conversion + graph (§2.1) | Very high | The end state if llama.cpp is too slow. Months of work; gate on measured gaps. |

### Verified upstream facts

- llama.cpp master has `LLM_ARCH_QWEN4EXP` (`"qwen4exp"`) in `src/llama-arch.cpp` and a
  `llama_model_qwen4exp` constructor. (checked 2026-09-27)
- llama.cpp master has generic MTP/NextN support (`n_layer_nextn`, `LLAMA_CONTEXT_TYPE_MTP`, the
  `draft-mtp` speculative type), but **`qwen4exp` is not wired into it**: `conversion/qwen4exp.py` sets
  `supports_mtp_export = False`/`no_mtp = True`, the `QWEN4EXP` tensor list has no `NEXTN_*` entries, and
  `src/models/qwen4exp.cpp` creates no nextn tensors and has no MTP graph. See §2.1. (checked 2026-09-27)
- **`qwen4exp` already solves the n-gram-table placement**: `src/models/qwen4exp.cpp` creates
  `per_layer_token_embd.weight` with `TENSOR_READ_LAZY`, and `llama-model-loader.cpp` resolves lazy tensors
  to the CPU buffer type and reads their rows from the file on demand. `--lazy-mode` (`auto` by default,
  only for tensors > 4 GiB) needs **mmap**; `--no-mmap` or `--lazy-mode off` forces the 28.80 GB table
  resident. `auto` degrades to `off` if any device reports no mmap support. See
  [gguf-inventory.md](gguf-inventory.md) §3.
- SYCL backend docs list Arc B-Series as supported (B580 verified) and Fedora among tested Linux distros.
- llama.cpp `llama-server` supports `--cache-type-k/-v` (q8_0, q4_0, ...), `-ot/--override-tensor`,
  `--cpu-moe`, `--n-cpu-moe N`, `--n-cpu-ffn N`, and `--spec-draft-*` flags for a draft/MTP model. The
  MTP (`draft-mtp`) path only serves the archs explicitly wired for it — not `qwen4exp` (§2.1).
- SYCL 2026.04-05 release notes add "Fused MoE" and K-quant reorder optimisations.

## 6. Recommendation

Ship a **pinned llama.cpp SYCL baseline first**, then invest in an expert-placement layer, and only then
consider a custom engine. Rationale and alternatives are in
[ADR-0001](adr/0001-runtime-architecture.md); the baseline engine choice is [ADR-0002](adr/0002-baseline-engine.md).

## 7. Open questions

1. ~~Exact tensor inventory / per-tensor byte sizes~~ — **resolved** in
   [gguf-inventory.md](gguf-inventory.md) (1,224 tensors, all three tiers, verified against the authors'
   allocation file and recovery capsules).
2. Arc Pro B70 memory bandwidth and Xe-core count (product page not retrievable from this host; read from
   `xpu-smi`/Level Zero once installed).
3. Does the published IQ3_XXS fit the expert pool with 128K KV on this box? (measure: it fits in the
   combined buffer, but leaves only ~3.5-6 GiB of RAM for the n-gram page cache —
   [gguf-inventory.md](gguf-inventory.md) §6.2/§7)
4. SYCL vs Vulkan on Battlemage for this arch and these quant types (measure both on a small model first).
5. N-gram table: mmap vs explicit SSD streaming; does the OS page cache keep up at token rate? (measure)
6. Model license / access conditions for scripted download (the repo answers anonymous range reads today).
7. ~~Is the MTP/NextN head really absent from the GGUF, and does a later release add it?~~ — **resolved
   (§2.1):** the base model and the Swift checkpoint both carry a 1-layer MTP head; the published GGUF
   drops it, and llama.cpp `qwen4exp` cannot convert or run one. Speculation is re-scoped to the
   n-gram/PLE path.
