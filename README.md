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

**Working and verified.** `./bongo.sh` takes a clean checkout to a live OpenAI-compatible
endpoint at a **131072-token context** on the reference box. The shipped default is the
pinned llama.cpp **Vulkan** build plus the M4.2 host-expert upload patch, auto expert
placement (`--n-cpu-moe 12` at <= 131072), and `--load-mode none`.

Speed is measured and regression-tracked. Versus the Stage 0 baseline the cached
agentic turn is **-31.7% at 16K** and **-19.6% at 128K**, and **4K decode is ~19.5 tok/s**
(target `>=19.0` median). The original `>=25 tok/s` decode ambition is retired to the
unfunded GPU milestone [BAS-166](/BAS/issues/BAS-166) (backlog). Start with
[`docs/final-overview.md`](docs/final-overview.md) for the goal, the approach, the
numbers, and how to run it.

Known gaps: the SYCL path is not the default, no clean-room install is verified, and
higher-quant tiers are not shipped. See the overview's "What is not done".

## Documentation

- [`docs/final-overview.md`](docs/final-overview.md) — **the goal, the approach, the speed, and how to run it.**
- [`docs/roadmap.md`](docs/roadmap.md) — the milestone ledger and target status.
- [`docs/research/intel-arc-b70.md`](docs/research/intel-arc-b70.md) — hardware, model, memory budget, runtime options.
- [`docs/research/gguf-inventory.md`](docs/research/gguf-inventory.md) — exact tensor inventory (1,224 tensors, all three tiers) and the 32 GB VRAM / 30 GiB RAM buffer-placement plan.
- [`docs/research/expert-placement.md`](docs/research/expert-placement.md) — the `--n-cpu-moe` sweep on the B70, the VRAM feasibility edge, and the Stage 1 adaptive-cache go/no-go.
- [`docs/research/moe4all-arc-b70-eval.md`](docs/research/moe4all-arc-b70-eval.md) — first-pass evaluation of MoE4All/INFR on the Arc B70 (quant speeds, the Intel host-DMA hang, and the Qwen3.8 MTP sidecar pairing) — [BAS-179](/BAS/issues/BAS-179).
- [`bench/results/2026-09-28-q2_0/RECOMMENDATION.md`](bench/results/2026-09-28-q2_0/RECOMMENDATION.md) — Q2_0 at 128K (smallest fitting `--n-cpu-moe` = 11) and why IQ2_XS stays the default.
- [`tools/gguf-inventory.py`](tools/gguf-inventory.py) — the header-range-read inventory tool (no weight download).
- [`docs/adr/0001-runtime-architecture.md`](docs/adr/0001-runtime-architecture.md) — the staged runtime decision.
- [`docs/adr/0002-baseline-engine.md`](docs/adr/0002-baseline-engine.md) — the baseline engine, amended: Vulkan is the default, SYCL is opt-in.
- [`docs/adr/0003-engine-direction.md`](docs/adr/0003-engine-direction.md) — patch llama.cpp, do not build a new engine.
- [`docs/adr/0005-host-cpu-critical-path.md`](docs/adr/0005-host-cpu-critical-path.md) — the M4 host/CPU critical path and the decode re-baseline.
- [`docs/bongo-sh.md`](docs/bongo-sh.md) — the one-command setup: options, provisioning, backends, tiers, config.
- [`docs/runbooks/active-run-watchdog-recovery.md`](docs/runbooks/active-run-watchdog-recovery.md) — platform ops: clear a board-owned `active_run_watchdog` recovery hold without re-filing work.
- [`CONTEXT.md`](CONTEXT.md) — project vocabulary.

## Intended one-command UX

```sh
 git clone https://github.com/03c/bongo && cd bongo && ./bongo.sh
```

`bongo.sh`:

1. detects the Arc GPU and OS, and installs the Intel compute runtime (Level Zero / oneAPI);
2. fetches the pinned llama.cpp build (**Vulkan** default, with the M4.2 host-expert upload patch);
3. downloads the chosen GGUF tier with resume;
4. starts an OpenAI-compatible server at `http://127.0.0.1:8080/v1` with a >= 128K context.

See [`docs/bongo-sh.md`](docs/bongo-sh.md) for options and the generated config.

## License

To be decided (see [BAS-48](/BAS/issues/BAS-48)). Model weights carry their own licenses (Swift Open License 1.0
and the Qwen Community License 1.0).
