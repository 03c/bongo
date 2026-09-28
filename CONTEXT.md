# CONTEXT — bongo

Project vocabulary. Keep this short; add a term when it first appears in code or docs.

## Product

- **bongo** — the project: an Intel Arc runtime + one-command setup for large MoE models.
- **Strata** — the sibling NVIDIA runtime (`github.com/Niko1221/Strata`). Reference implementation for the
  tiered-memory design and the speculation work. bongo reuses its *ideas*, not its CUDA code. Strata keeps
  the model's MTP head because it owns its conversion and engine; bongo cannot (see **MTP** below).
- **fringeplan.com** — Bassett's unrelated first product. Out of scope here.

## Hardware

- **Arc Pro B70** — Intel Battlemage G31 discrete GPU, 32 GB VRAM, `xe` kernel driver. Targets PCI
  `8086:E223`. The reference machine has one, alongside a Ryzen 7 9700X and 30 GiB RAM.
- **B-series** — Intel Arc Battlemage generation (B50 / B60 / B70). The per-token code path is largely shared
  with Alchemist (A-series), so A-series regression testing is useful but not authoritative.
- **VRAM / RAM / SSD tiers** — the three memory levels the runtime schedules weights and experts across.
  Strata's terms; keep them.

## Model

- **Qwen3.8-Flash-Next** — the upstream 125B-parameter MoE model (`Qwen/Qwen3.8-Flash-Next`).
  Architecture string in GGUF: `qwen4exp`.
- **Swift 1.5** — UkisAI's fine-tune of Qwen3.8-Flash-Next that emits fewer thinking tokens at ~equal accuracy.
  Same architecture. The bongo target model.
- **GSQ-RCO** — the quantization method (ISTA-DASLab) behind the published GGUF tiers.
- **Tier** — one published quantization: `Q2_0`, `IQ2_XS`, `IQ3_XXS`. Higher tier = more bytes and better KLD.
- **Expert** — one MoE feed-forward branch. The model has `512 experts/layer x 48 layers = 24,576` experts,
  `10` active per token per layer. Experts are ~4.9 M parameters each and dominate model size.
- **Expert cache** — the VRAM region that holds the most-used experts; the rest live in RAM (and, if needed,
  stream from SSD). Strata's key trick and bongo's central optimisation.
- **MTP** — multi-token prediction / speculative decoding. The base model and the Swift checkpoint both ship
  a 1-layer MTP head, **but the published GGUF drops it and llama.cpp `qwen4exp` cannot convert or run it**.
  bongo's speculation path is therefore the n-gram/PLE table, not an MTP head
  ([research §2.1](docs/research/intel-arc-b70.md)).
- **KV cache** — attention key/value state. Only 12 of 48 layers are full attention; the other 36 are linear
  attention with a constant-size state, so 128K context is cheap (~2-4 GB), not the bottleneck.
- **N-gram table** — a very large lookup table in the model (~29 GB) that is read a few rows per token and can
  stay on SSD. Treat it as a disk-resident tensor, not RAM-resident.

## Runtime

- **SYCL / oneAPI** — Intel's cross-vendor compute stack; the primary bongo GPU backend (built with `icpx`,
  run through Level Zero).
- **Level Zero** — low-level Intel GPU driver interface; the SYCL runtime sits on top of it.
- **Vulkan (ANV)** — the simpler alternative GPU backend via Mesa; weaker MoE/i-quant coverage than SYCL.
- **IPEX-LLM** — Intel's prebuilt llama.cpp/SYCL distribution. Considered as a zero-build fallback, not the
  long-term engine.
- **llama.cpp** — the baseline engine. Already knows `qwen4exp`, i-quants, generic MTP (not wired for
  `qwen4exp`), and MoE CPU offload.
- **Baseline vs engine** — *baseline* = pinned llama.cpp SYCL configuration that works and is benchmarked;
  *engine* = the bongo-owned optimisation layer (adaptive expert cache, SSD streaming, tuned kernels).
