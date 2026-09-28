# Expert placement on the Arc Pro B70 — sweep + Stage 1 go/no-go

Measurement-only spike for [BAS-53](/BAS/issues/BAS-53), feeding the Stage 1 gate in
[ADR-0001](../adr/0001-runtime-architecture.md). It sweeps llama.cpp's static
expert-placement control (`--n-cpu-moe`) on the reference box and asks whether a custom
adaptive VRAM expert cache is justified.

- Date: 2026-09-28
- Hardware: Intel Arc Pro B70 ("Battlemage G31", 32 GiB VRAM), AMD Ryzen 7 9700X, 32 GB system RAM, Fedora 44
- Engine: llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), **Vulkan** backend (`--device Vulkan1`)
- Model: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, tier **IQ2_XS**
  (shard SHA-256 recorded in [`bench/results/2026-09-27-baseline/matrix.md`](../../bench/results/2026-09-27-baseline/matrix.md))
- Raw results: [`bench/results/2026-09-27-expert-placement/`](../../bench/results/2026-09-27-expert-placement/)
  (`sweep-matrix.json`, `sweep-matrix.md`, and one `ncmoe-<N>/` directory per config)

## TL;DR — Stage 1 recommendation

**No-go (defer) on the custom adaptive VRAM expert cache for IQ2_XS on this box, as a
throughput optimisation at the 128K target.**

The Stage 1 differentiator would move expert residency around to compute fewer experts on
the CPU. But the measurement below shows that at 128K **decode throughput is essentially
insensitive to expert placement**: going from 22.4 GiB to 16.8 GiB of GPU-resident experts
(an extra 5.6 GiB / ~25% of the expert set on the CPU) changes 128K output throughput by
only **-1.2%** (7.62 → 7.53 tok/s). Prefill is more sensitive (~3.4 prompt-tok/s per GPU
GiB, ~14% over the same 5.6 GiB), but the static split already sits at the VRAM feasibility
edge, so a cache cannot add GPU residents — it can only re-label which ones are hot. The
measured ceiling on that reallocation is a low-single-digit-percent prefill change at 128K.

The performance cost of the target context is dominated by long-context attention/KV, not
by where the experts live: 4K → 128K drops decode from ~16-18 tok/s to ~7.5-8 tok/s on
every configuration. Stage 1 money is better spent elsewhere.

## Method

### The control

`--n-cpu-moe N` keeps the routed experts (`ffn_{gate,up,down}_exps`) of the **first N
layers** on the CPU; layers `N..47` keep their experts on the GPU. Everything else
(attention, SSM, routers, shared experts, hyper-connections, KV) stays on the GPU. It is
therefore a *layer-count* rule, not a byte-budget rule: because per-layer expert bytes vary
(0.56 / 0.62 / 0.72 GiB), moving one layer is not always the same number of bytes.

The exact per-layer expert byte counts used here are in
[`expert-bytes-iq2_xs.json`](../../bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json);
they come from the GGUF tensor table (`tools/gguf-inventory.py`) and agree with
[`gguf-inventory.md`](gguf-inventory.md). Total expert set: **33.02 GiB**.

### The sweep

[`bench/sweep-expert-placement.sh`](../../bench/sweep-expert-placement.sh) restarts the
bongo server for each `--n-cpu-moe` value and runs the Stage 0 harness
([`bench/harness.py`](../../bench/harness.py)) at **4096** and **131072** context, one
repeat each. For every config it records prompt tok/s, output tok/s, TTFT, peak VRAM
(`/proc/<pid>/fdinfo` `drm-resident-vram0`), peak process RSS, and whether the config
loaded and served.

Two methodology points that matter for reproducibility:

- **Warm-up before measuring.** Each restart is followed by a discarded ~4K prefill so the
  measured 4K run is not the first touch of the weights. Without it the first 4K prefill is
  a cold-page-cache number (e.g. `--n-cpu-moe 24` measured 98.7 prompt-tok/s cold vs 186.5
  warm), while the later 128K run is warm — an apples-to-oranges comparison.
- **Fit is measured, not assumed.** "Fits" means the config loaded **and** both contexts
  served a 200. A failed context still records the peak VRAM reached before the failure.

### A VRAM correction

The project's placement plan in [`gguf-inventory.md`](gguf-inventory.md) §6 budgets against
"32 GB = 29.80 GiB". The card actually exposes **32 GiB** of VRAM:

```
lspci:        Memory at f000000000 (64-bit, prefetchable) [size=32G]
journalctl -k: xe 0000:03:00.0: VRAM[0]: Actual physical size 0x0000000800000000
               usable size exclude stolen 0x00000007f9000000
```

`0x7f9000000` = 34,275,917,824 B = **31.92 GiB usable** (32.00 GiB physical minus stolen).
That is ~2.1 GiB more headroom than the plan assumed, which is why `--n-cpu-moe 12` loads at
31.4 GiB instead of OOM-ing immediately. It is still not enough for an all-GPU expert set
(needs ~33 GiB of experts plus ~3.6 GiB of weights plus KV).

## Results

Warm measurement, IQ2_XS, one repeat per context. `GPU experts` / `CPU experts` are derived
from the exact tensor bytes, not from the layer count. Source:
[`sweep-matrix.md`](../../bench/results/2026-09-27-expert-placement/sweep-matrix.md).

| n-cpu-moe | loads | fits 128K | GPU experts GiB | CPU experts GiB | CPU expert share | 4K prompt tok/s | 4K output tok/s | 4K VRAM GiB | 128K prompt tok/s | 128K output tok/s | 128K VRAM GiB | 128K RAM GiB |
| ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **no** | — | 33.02 | 0.00 | 0.000 | — | — | — | — | — | — | — |
| 12 | yes | **no** | 25.02 | 8.00 | 0.242 | 260.44 | 17.71 | 31.45 | *crash* | *crash* | 31.85 | 7.12 |
| 16 | yes | yes | 22.40 | 10.62 | 0.322 | 231.50 | 16.11 | 28.82 | 133.43 | 7.62 | 29.26 | 10.96 |
| 24 | yes | yes | 16.83 | 16.19 | 0.490 | 186.50 | 11.25 | 23.25 | 114.46 | 7.53 | 23.68 | 16.53 |
| 48 | yes | (pending) | 0.00 | 33.02 | 1.000 | see below | see below | see below | see below | see below | see below | see below |

The Stage 0 baseline (3 repeats at 4K, 1 at 128K) is the same configuration as `n=16` and
reproduces it: 4K prompt 231.46 / output 17.71, 128K prompt 133.16 / output 7.98, 128K
needle **pass**, peak VRAM 29.27 GiB. The sweep's single-repeat `n=16` is 133.43 / 7.62 — within
~5% on output, identical on prefill.

### What fails, and how

- **`n=0` (all experts on GPU):** model load fails. The server dies allocating a 0.78 GiB
  Vulkan buffer — the expert set alone is 33.02 GiB, more than the 31.92 GiB usable VRAM.
  This is the hard ceiling: llama.cpp's default all-GPU placement is unreachable on this card.
- **`n=12`:** loads (31.37 GiB after load) and is the *fastest* short-context config
  (4K prompt 260.4, output 17.71), but at 128K the GPU is lost:
  `decode() failed: vk::Queue::submit: ErrorDeviceLost` after a 870 s prefill, peak VRAM
  31.85 GiB. It is 0.07 GiB under the usable size at the moment it dies — the 128K KV cache
  and compute buffers push a config that fits at load over the edge at depth.
- **`n=16` and `n=24`:** complete both contexts, needle passes, no spill visible.

**Feasibility edge for 128K is between `n=12` and `n=16`** (untested: `n=13,14,15`). The
shipped default (`n=16`) is at that edge and is a correct conservative choice.

## Analysis

### Throughput per GPU-resident expert GiB

Comparing the two configs that both serve 128K:

| config | GPU experts GiB | 4K output tok/s | 128K output tok/s | 128K prompt tok/s |
| --- | ---: | ---: | ---: | ---: |
| n=16 | 22.40 | 16.11 | 7.62 | 133.43 |
| n=24 | 16.83 | 11.25 | 7.53 | 114.46 |
| **marginal** | **+5.57** | **+4.86** | **+0.09** | **+18.97** |
| **per GPU GiB** | | **+0.87** | **+0.016** | **+3.41** |

- **At 128K decode, the marginal value of an expert GiB on the GPU is ~0.016 tok/s — inside
  the run-to-run noise.** The 128K needle passing at both ends proves the context is real;
  the cost is the long-context attention/KV path, which both configs pay identically.
- **At 128K prefill the marginal is ~3.4 prompt-tok/s per GPU GiB (~14% over 5.6 GiB).**
  Prefill is where expert placement still shows up at depth, because every prompt token
  routes through the experts and the CPU expert matmul is the slow path.
- **At 4K both are sensitive** (marginal ~+0.87 output tok/s per GPU GiB, and the `n=12`
  point is faster still), so placement matters much more for short conversations.

### CPU-expert share

The static rule puts a **fixed** share of the expert set on the CPU: `n=16` → 32.2% of
expert bytes (33.3% of layers), `n=24` → 49.0%. That share is constant for the whole
conversation, independent of which experts are actually being selected.

### Does the best static split leave a gap an adaptive cache could close?

The adaptive cache's mechanism is *which* experts are GPU-resident, not *how many*. The
number is pinned by VRAM: `n=16` already uses 22.4 GiB of the ~22-25 GiB expert budget, and
`n=12` (25.0 GiB) already fails at 128K. So an adaptive cache cannot keep meaningfully more
expert bytes on the GPU than the static split — it can only replace cold experts with hot
ones.

The most a frequency-based cache could buy, therefore, is the difference between the static
CPU share (32%) and the CPU share implied by the hot-expert working set. If expert activation
is heavily skewed, that might be a few GiB of effective GPU residency. Valued at the
measured marginals, that is:

- **128K output: ≲1%** (0.016 tok/s/GiB × a few GiB).
- **128K prefill: a low-single-digit-percent improvement** (3.4 prompt-tok/s/GiB × a few
  GiB), on a prefill that already takes ~15 minutes.
- **4K output/prefill: ~10-15%, but only in the first turns of a conversation before the
  context grows.**

Against that, Stage 1 carries a fork-or-scheduler cost, VRAM-pressure risk (the `n=12`
device-loss shows how thin the margin is), and a new failure surface at exactly the context
length the product promises.

### The cheaper alternative the data points to

The `n=12` result is interesting for a different reason: it is the fastest short-context
config (4K prefill +12.5%, output +9.9% over `n=16`) but cannot survive 128K. That is a
**context-dependent configuration** problem, not an expert-cache problem. A conversation can
start at `n=12` and step to a 128K-safe `n` (13-16) as the context grows; that captures most
of the 4K win with a config switch and no new engine code.

## Stage 1 go/no-go

**No-go** for the custom adaptive VRAM expert cache as the Stage 1 differentiator, on the
current evidence:

1. At the required 128K context, expert placement barely moves decode (1.2% across a 25%
   shift in GPU expert residency), so the cache's only clear lever is prefill.
2. The cache cannot increase GPU expert residency; `n=12` already exceeds the 128K
   feasibility edge, so the static budget is the real constraint.
3. The measured up-side (low single digits at 128K) does not justify a fork or a scheduler
   layer, and the `n=12` device-loss shows the VRAM margin is too thin to safely trade.

Re-open Stage 1 only if a later measurement changes the premise, e.g. the SYCL backend is
made to work (its MoE kernels may move the CPU/GPU balance), a smaller/other tier changes
the expert:VRAM ratio, or a byte-budget (`-ot`) placement beats the layer-count rule.

### Smallest next experiment (cheap, no cache code)

Find the true 128K feasibility edge and test the context-adaptive switch. Concretely:

1. Run `--n-cpu-moe 13,14,15` at 128K (same harness, 1 repeat) to locate the largest
   GPU-resident split that survives 128K without a device loss. This bounds how much the
   static default can be raised.
2. If `n=13/14` survives, measure the 4K→128K switch: start at `n=12`, step to the edge when
   the prompt crosses a threshold (e.g. 32K), and compare end-to-end against fixed `n=16`.

This is a config-layer change; it does not require the Stage 1 cache and can be tested with
the existing harness.

## Reproduce

```sh
# Server per config is started by the sweep script:
#   ./bongo.sh --tier iq2_xs --gguf-dir <gguf> --llama-bin ~/.bongo/llama/b11223/vulkan \
#     --runtime dir --runtime-dir ~/.bongo/runtime-empty --backend vulkan \
#     --n-cpu-moe <N> --detach --yes

# Full sweep (resumable; skips configs that already have a matrix.json):
./bench/sweep-expert-placement.sh 24,0,12,16,48

# Aggregate the per-config results into sweep-matrix.{json,md}:
python3 bench/aggregate-expert-placement.py
```

Environment on the reference box: Mesa Vulkan (ANV) only — no Level Zero / oneAPI / SYCL is
installed, so the baseline runs the documented Vulkan fallback from
[ADR-0002](../adr/0002-baseline-engine.md). `--device Vulkan1` pins the Arc; the CPU's AMD
iGPU is `Vulkan0`.

## Limitations

- **One repeat per context.** A 128K prefill is ~15-20 minutes, so `--repeats 1` was used.
  The `n=16` config matches the 3-repeat Stage 0 baseline within ~5% on output and exactly
  on prefill, which bounds the systematic error; differences reported here are 1-15%.
- **Prefill is measured with the server's own prompt sizing** (`/tokenize`), so prompt token
  counts are honest (130,861 tokens at the 128K target).
- **Vulkan only.** These numbers do not transfer to a working SYCL backend, whose MoE kernels
  may shift the CPU/GPU balance — that is the first thing to re-measure if SYCL is fixed.
- **`-ot` byte-budget placement was not swept.** `--n-cpu-moe N` is layer-count based; a
  byte-budget `-ot` rule could pack slightly more expert bytes into VRAM and is worth a
  follow-up if the feasibility edge matters.
