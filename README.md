# bongo

**A local LLM runtime for the Intel Arc Pro B70 (Battlemage, 32 GB).**

bongo is the Intel-Arc counterpart to [Strata](https://github.com/Niko1221/Strata): one command pulls a
125-billion-parameter MoE model, serves it behind an OpenAI-compatible endpoint, and squeezes the model
across GPU VRAM, system RAM, and SSD so it runs on a single desktop-class machine.

Target machine (the reference box):

| | |
| --- | --- |
| GPU | Intel Arc Pro B70 ("Battlemage G31"), **32 GB** VRAM, `xe` driver |
| RAM | **32 GB** system RAM |
| Disk | SSD, ~80 GB free for the model |
| OS | Fedora 44 (primary), Ubuntu 24.04 (secondary) |
| Context | **128K tokens minimum**, up to 262K if memory allows |

Target model: [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(Qwen3.8-Flash-Next 125B MoE, Swift 1.5 fine-tune). Start at **IQ2_XS**; reach for **IQ3_XXS** if it fits.

## Status

Early. This repository currently contains the research, the architecture decision, and the task plan.
The one-command setup is being built under [BAS-48](/BAS/issues/BAS-48).

## Documentation

- [`docs/research/intel-arc-b70.md`](docs/research/intel-arc-b70.md) — hardware, model, memory budget, runtime options.
- [`docs/research/gguf-inventory.md`](docs/research/gguf-inventory.md) — exact tensor inventory (1,224 tensors, all three tiers) and the 32 GB VRAM / 30 GiB RAM buffer-placement plan.
- [`tools/gguf-inventory.py`](tools/gguf-inventory.py) — the header-range-read inventory tool (no weight download).
- [`docs/adr/0001-runtime-architecture.md`](docs/adr/0001-runtime-architecture.md) — the staged runtime decision.
- [`docs/adr/0002-baseline-engine.md`](docs/adr/0002-baseline-engine.md) — why llama.cpp SYCL is the baseline.
- [`docs/roadmap.md`](docs/roadmap.md) — milestones and the task graph.
- [`CONTEXT.md`](CONTEXT.md) — project vocabulary.

## Intended one-command UX

```sh
git clone https://github.com/03c/bongo && cd bongo && ./bongo.sh
```

`bongo.sh` will:

1. detect the Arc GPU and OS, and install the Intel compute runtime (Level Zero / oneAPI);
2. build or fetch a pinned llama.cpp SYCL binary;
3. download the chosen GGUF tier with resume;
4. start an OpenAI-compatible server at `http://127.0.0.1:8080/v1` with a >=128K context.

## License

To be decided (see [BAS-48](/BAS/issues/BAS-48)). Model weights carry their own licenses (Swift Open License 1.0
and the Qwen Community License 1.0).
