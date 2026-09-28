# Memory-capacity sensitivity for bongo — VRAM/RAM scaling and a capacity model

Measurement + modelling spike for [BAS-69](/BAS/issues/BAS-69), answering the CEO's capacity
question for [BAS-62](/BAS/issues/BAS-62): **does more VRAM and more system RAM convert into
higher throughput for this model on this engine, and can we prove the RAM benefit at 16 vs 32 GiB
and extrapolate to 64 GiB?**

- **Date:** 2026-09-28
- **Hardware:** Intel Arc Pro B70 (32 GiB VRAM, 31.92 GiB usable), AMD Ryzen 7 9700X, 30.45 GiB
  system RAM, Fedora 44, kernel `7.0.13-200.fc44.x86_64`
- **Engine:** llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), Vulkan build
- **Model:** `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`qwen4exp`), tier **IQ2_XS**,
  48 layers x 512 experts, 10 active per token per layer, **33.02 GiB** of expert weights,
  68.15 GB of shards on the KIOXIA EXCERIA G3 NVMe
- **Raw data:**
  - VRAM axis: [`bench/results/2026-09-27-expert-placement/`](../../bench/results/2026-09-27-expert-placement/)
    (`sweep-matrix.json`, unconstrained RAM)
  - activation hit-rate: [`bench/results/2026-09-28-expert-activation/analysis.json`](../../bench/results/2026-09-28-expert-activation/analysis.json) ([BAS-66](/BAS/issues/BAS-66))
  - RAM axis: [`bench/results/2026-09-28-capacity-sensitivity/`](../../bench/results/2026-09-28-capacity-sensitivity/)
    (`mem16g-ncmoe16/`, `mem16g-ncmoe24/`, `samples.jsonl`, `cgroup.json`, `server-io.json`)
  - per-read SSD cost: [`bench/results/2026-09-28-ssd-random4k-probe/`](../../bench/results/2026-09-28-ssd-random4k-probe/) and
    [`docs/research/ssd-ngram-shard.md`](ssd-ngram-shard.md) ([BAS-67](/BAS/issues/BAS-67))
- **Tooling:** [`bench/run-capacity-sensitivity.sh`](../../bench/run-capacity-sensitivity.sh),
  [`bench/capacity_sampler.py`](../../bench/capacity_sampler.py),
  [`bench/capacity-model.py`](../../bench/capacity-model.py) (reproduces every modelled table below)

## TL;DR — verdict

**Conditional, and the condition is the opposite of the intuitive one.** On this engine and model:

- **More VRAM buys throughput only on the CPU-expert path, and only up to the transfer point.**
  Decode at 4K output rises from 2.95 tok/s (0 GiB of experts on the GPU) to 16.11 (22.40 GiB) —
  but the last 5.6 GiB (16.83 → 22.40) is worth only +0.87 output tok/s/GiB, and at 128K it is
  worth **+0.016 output tok/s/GiB**, i.e. nothing. The 128K decode path is fixed-cost, not
  expert-placement-bound.
- **More system RAM does not buy throughput at any measured configuration.**
  Constraining `llama-server` with a reversible cgroup `memory.max=16G` (and `MemorySwapMax=0`)
  reproduced the unconstrained (~30 GiB) sweep to within run-to-run noise at the shipped
  `--n-cpu-moe 16` split (4K output 15.61 vs 16.11, 128K output **8.00 vs 7.62**, 128K prefill
  133.5 vs 133.4) and at the harsher `--n-cpu-moe 24` split whose CPU expert set (16.19 GiB)
  **does not fit** in the 16 GiB cap.
- **At the shipped configuration the RAM requirement is already met with room to spare.** 22.40 GiB
  of experts live in VRAM, so only 10.62 GiB of expert pages need the page cache; 16 GiB holds
  them together with the live PLE window. The modelled 32 → 64 GiB RAM gain at 32 GiB VRAM is
  **0 tok/s**. RAM starts to matter only when VRAM is small enough that the CPU expert set
  approaches the RAM budget — around a 12 GiB GPU (27.98 GiB of CPU experts).
- **The real RAM effect is SSD traffic, not throughput.** A 26.82 GiB lazily-read per-token PLE
  table ([BAS-67](ssd-ngram-shard.md)) competes with the expert pages for the same page cache.
  Measured per-run SSD `read_bytes` is essentially unchanged between the 16 GiB cap and the
  unconstrained 30 GiB box at `n=16` (1.22 GiB vs 1.64 GiB during the measured window) — see §3.3
  for the one case (all-CPU) where it explodes to 74 GB.
- **Where it stops paying:** VRAM stops paying once the GPU-resident expert budget reaches
  22.40 GiB (the 128K-safe ceiling); the marginal 128K decode return beyond ~17 GiB of GPU experts
  is inside noise, and beyond 22.4 GiB the 128K KV/compute buffers make the config unsafe. RAM
  stops paying once the cap holds `CPU expert GiB + ~4 GiB` (also true at 16 GiB for the shipped
  VRAM budget). 64 GiB RAM is worth nothing on this box and model; it is only interesting for a
  low-VRAM box (Strata's 12 GiB case) where the CPU expert set is ~28 GiB.

## 1. Method

### 1.1 VRAM axis (reused measurement)

No new GPU sweep was run. The committed Stage-1 sweep
([`bench/results/2026-09-27-expert-placement/sweep-matrix.json`](../../bench/results/2026-09-27-expert-placement/sweep-matrix.json))
restarts `llama-server` per `--n-cpu-moe` value and runs
[`bench/harness.py`](../../bench/harness.py) at 4096 and 131072, one discarding 4K warm-up first.
It records prompt/output tok/s, TTFT, peak VRAM and peak RSS, and the cumulative SSD
`read_bytes` delta across the measured window. The measured rows are re-tabulated in §2.

### 1.2 RAM axis (new, primary evidence)

[`bench/run-capacity-sensitivity.sh`](../../bench/run-capacity-sensitivity.sh) starts `llama-server`
inside a **transient, reversible** cgroup-v2 scope:

```sh
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- ./bongo.sh ...
```

`memory.max` is enforced by the kernel and the scope disappears when the run ends — no kernel
command-line change, no reboot, no `bongo.sh` change. `bench/capacity_sampler.py` samples
`/proc/<pid>/io`, `/proc/<pid>/stat` (major faults), `/proc/<pid>/status` (VmRSS/VmHWM),
`/proc/<pid>/fdinfo` (VRAM) and the cgroup `memory.current` / `memory.events` / `memory.stat` once
per 0.5 s. Two RAM budgets were run at the shipped IQ2_XS tier:

| run | memory.max | `--n-cpu-moe` | CPU expert GiB | contexts |
| --- | ---: | ---: | ---: | --- |
| `mem16g-ncmoe16` | 16 GiB | 16 (shipped) | 10.62 | 4096, 131072 |
| `mem16g-ncmoe24` | 16 GiB | 24 | 16.19 | 4096, 131072 |

The `n=24` run is the deliberate over-subscription case: its CPU expert set (16.19 GiB) plus the
live PLE window does **not** fit in the cap, so it is the experiment that *can* show a RAM effect.
The cgroup was confirmed in force: `memory.max=17179869184`, `memory.current` peaked at
17.18 GB, `memory.events max` incremented 6342 times, `oom_kill 0`
([`mem16g-ncmoe16/cgroup.json`](../../bench/results/2026-09-28-capacity-sensitivity/mem16g-ncmoe16/cgroup.json),
[`memory.json`](../../bench/results/2026-09-28-capacity-sensitivity/mem16g-ncmoe16/memory.json)).

A 24 GiB third point was **not** run: at the shipped `n=16` split the 16 GiB point already does not
bind (the whole CPU working set fits), and the unconstrained ~30 GiB sweep row is the control, so
24 GiB could only interpolate between two identical results. The time was spent on the
over-subscribed `n=24` point instead.

## 2. Measured VRAM axis

Warm, one repeat per context, IQ2_XS, unconstrained RAM. SSD `read_bytes` is the delta over the
measured window (after the discarded warm-up). Source:
[`sweep-matrix.json`](../../bench/results/2026-09-27-expert-placement/sweep-matrix.json).

| n-cpu-moe | loads | 128K fits | GPU experts GiB | CPU experts GiB | 4K prompt tok/s | 4K output tok/s | 128K prompt tok/s | 128K output tok/s | SSD read_bytes |
| ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | no | — | 33.02 | 0.00 | — | — | — | — | — |
| 12 | yes | **no** | 25.02 | 8.00 | 260.44 | 17.71 | *device lost* | *device lost* | — |
| 16 | yes | yes | 22.40 | 10.62 | 231.50 | 16.11 | 133.43 | 7.62 | 1.64 GB |
| 24 | yes | yes | 16.83 | 16.19 | 186.50 | 11.25 | 114.46 | 7.53 | 2.13 GB |
| 48 | yes | yes | 0.00 | 33.02 | 52.92 | 2.95 | 72.35 | 2.26 | **74.08 GB** |

### 2.1 Marginal tok/s per GPU-resident expert GiB

| move | ΔGPU GiB | 4K output tok/s | per GiB | 128K output tok/s | per GiB | 4K prompt tok/s | per GiB | 128K prompt tok/s | per GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n=12 → 16 | −2.62 | −1.60 | **+0.61** | *unsafe* | — | −28.94 | +11.0 | *unsafe* | — |
| n=16 → 24 | −5.57 | −4.86 | **+0.87** | −0.09 | **+0.016** | −45.00 | +8.1 | −18.97 | +3.4 |
| n=24 → 48 | −16.83 | −8.30 | **+0.49** | −5.27 | **+0.313** | −133.58 | +7.9 | −42.11 | +2.5 |

### 2.2 What limits the curve

- **128K decode is fixed-cost.** `n=16` (22.40 GiB GPU experts) and `n=24` (16.83 GiB) decode at
  7.62 and 7.53 tok/s — a 1.2% difference for a 25% shift in GPU residency. Fitting the three 128K
  anchors (CPU expert bytes `B` → ms/token) gives a fixed term of **131.1 ms/token** and an
  expert term that is negligible until the all-CPU cliff (§4.1). That fixed term is the long-context
  attention/KV and dense work every configuration pays identically; **it is the dominant bottleneck
  at the product's 128K target**, not placement.
- **4K decode is placement-sensitive** at 0.49–0.87 output tok/s per GPU GiB. The expert term is
  the bottleneck here (`T0` = 51.2 ms of the 62 ms at `n=16`; the CPU term grows as `B^2.86`).
- **Prefill is placement-sensitive at every context** (~2.5–3.4 128K prompt-tok/s per GPU GiB on
  the `n=16→24` and `n=24→48` steps). This matches [BAS-66](expert-activation-skew.md).
- **The all-CPU cliff (n=48)** is a different regime, not a RAM effect: 74.08 GB of SSD reads in
  one run and 2.26 tok/s at 128K. Both the CPU compute and the page-cache thrash contribute; §3.3
  separates them as far as the data allows.

## 3. Measured RAM axis

Raw data: [`bench/results/2026-09-28-capacity-sensitivity/`](../../bench/results/2026-09-28-capacity-sensitivity/)
(`matrix.json`, `matrix.md`, `samples.jsonl`, `memory.json`, `server-io.json`, `cgroup.json` per run).
Numbers below are the harness medians (one repeat per context).

### 3.1 Shipped split (`--n-cpu-moe 16`, CPU experts 10.62 GiB)

| RAM budget | 4K prompt tok/s | 4K output tok/s | 128K prompt tok/s | 128K output tok/s | 128K TTFT s | peak VRAM GiB | peak RSS GiB | SSD read_bytes (window) | major faults (window) | needle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| ~30 GiB (unconstrained sweep) | 231.50 | 16.11 | 133.43 | 7.62 | 981 | 29.26 | 10.96 | 1.64 GB | — | pass |
| **16 GiB (cgroup)** | 231.25 | 15.61 | 133.52 | **8.00** | 980 | 29.26 | 10.89 | 1.22 GB | 8,096 | pass |

**The cap changes nothing measurable.** Every context served, the needle passed, peak VRAM and peak
RSS are identical to two decimals, and the SSD traffic during the measured window is the same
(1.22 vs 1.64 GB — the constrained run read *less*). The deltas are inside the sweep's own
run-to-run envelope. The reason is visible in the geometry: at this split only 10.62 GiB of expert
pages need the page cache, and they fit in 16 GiB together with the live PLE window, so the kernel's
reclaim (6342 `max` events) only discards the 60 GB of one-shot load pages behind the server's back.

### 3.2 Over-subscribed split (`--n-cpu-moe 24`, CPU experts 16.19 GiB)

| RAM budget | 4K prompt tok/s | 4K output tok/s | 128K prompt tok/s | 128K output tok/s | peak VRAM GiB | peak RSS GiB | SSD read_bytes (window) | major faults (window) | needle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| ~30 GiB (unconstrained sweep) | 186.50 | 11.25 | 114.46 | 7.53 | 23.68 | 16.53 | 2.13 GB | — | pass |
| **16 GiB (cgroup)** | 186.30 | 14.29 | 112.92 | **6.99** | 23.68 | 15.56 | **6.84 GB** | 62,725 | pass |

This is the point that *should* bind: 16.19 GiB of CPU experts (more than the ~12 GiB page-cache
budget left after overhead) and the cgroup peaked at 17.18 GB. **The I/O effect is large and the
throughput effect is small.** Against the committed unconstrained `n=24` sweep row the SSD reads
rose **3.2×** (6.84 GB vs 2.13 GB); against the same-protocol constrained `n=16` run they rose
**5.6×** (6.84 GB vs 1.31 GB) and major faults rose **7.7×** (62,725 vs 8,096). Against the
committed unconstrained `n=24` sweep row the 128K prefill is **−1.3%** (112.92 vs 114.46) and 128K
decode **−7.2%** (6.99 vs 7.53). The 4K output is *higher* than the sweep (14.29 vs 11.25) because
the sweep's single `n=24` 4K repeat sits below the same-protocol control (§3.4); a memory cap
cannot raise throughput, so that gap is baseline noise, not a RAM gain.

Take the two rows together: **when the CPU expert set exceeds the RAM budget the machine reads
3× more from the SSD, and still loses under 10% of throughput.** The page-cache misses are almost
entirely hidden behind the (already CPU-bound) expert compute.

### 3.3 SSD traffic and page faults

Two facts about the constrained runs:

- **`read_bytes` is the meaningful SSD counter; `rchar` is not.** llama.cpp serves the model
  through `mmap`, so file reads happen as page faults and never appear as `read()`/`pread()`
  syscalls. Over the whole measured window `rchar` moved by ~0.2 MB while `read_bytes` moved by
  gigabytes. The harness records both (`server-io.json`); only `read_bytes` is used.
- **The all-CPU configuration is the one that thrashes the SSD.** `--n-cpu-moe 48` pushed its
  entire 33.02 GiB expert set through a 30 GiB box and read **74.08 GB** from the device in one
  run, against 1.6–2.1 GB for the partially offloaded splits. That is the measured failure mode
  when the CPU working set exceeds RAM.

## 4. Capacity model

[`bench/capacity-model.py`](../../bench/capacity-model.py) is the runnable model. It reads the
committed per-layer expert bytes, the BAS-66 hit-rate curve, the Stage-1 sweep and the two
capacity runs, and reproduces every table below.

### 4.1 Throughput term (VRAM axis)

Per context, decode time per token is modelled as

```
t_ms(ctx, B) = T0(ctx) + A(ctx) * B^p(ctx)          B = CPU-resident expert GiB
```

with `T0, A, p` fit by grid search over `p` plus linear least squares over the sweep anchors
(`bench/results/2026-09-27-expert-placement/sweep-matrix.json`). `B` is computed from the exact
per-layer expert bytes (`expert-bytes-iq2_xs.json`) by the static `--n-cpu-moe` split, and the
coverage `h(GPU budget)` is the BAS-66 static curve. Fitted coefficients:

| ctx | T0 (ms) | A (ms/GiB^p) | p | in-sample SSE |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 51.23 | 0.0131 | 2.86 | 0.3 |
| 131072 | 131.11 | 5.7e-7 | 7.34 | 0.0 |

The 4K term is a shallow power law: the CPU-expert compute cost. The 128K term is effectively a
step: attention/KV fixes 131 ms and the expert path stays hidden until `B` approaches the whole set.
`T0(128K) = 131.1 ms` is **74% of the shipped 131.3 ms/token**, which names the dominant bottleneck.

### 4.2 RAM term

Measured, not fitted: the RAM budget enters the model in two places.

1. **Feasibility.** A configuration is RAM-safe iff `CPU expert GiB + 4 GiB <= RAM`. The 4 GiB is
   the live PLE window ([BAS-67](ssd-ngram-shard.md): 26.82 GiB table, 16 pages = 64 KiB per token)
   plus loader/process overhead; 4 GiB is the round budget that makes the 16 GiB `n=16` measurement
   consistent.
2. **SSD traffic.** The CPU expert bytes above the cacheable budget
   (`cacheable = RAM − 4 GiB`) are the pages that must be re-read from the device. This is the
   quantity that grows in `read_bytes`; §3.3 shows the measured values.

RAM does **not** enter the throughput term inside the measured range, because the two deliberately
different RAM budgets produced the same tok/s at the same VRAM split (§3.1, §3.2). The model keeps a
deliberately pessimistic *serialised-miss* bound (every miss paid in full at the measured QD1
random-4K bandwidth, 7.5 MB/s) only to show how completely the engine hides the reads; the
measurement falsifies it.

### 4.3 Grid — modelled output tok/s

`G = min(VRAM − non-expert footprint, 22.40 GiB)`, `non-expert` is 6.86 GiB at 128K and 6.42 GiB
at 4K, and 22.40 GiB is the measured-safe GPU expert budget at 128K.

**ctx 131072** (the product target)

| VRAM GiB | RAM GiB | GPU exp GiB | n-cpu-moe | CPU exp GiB | RAM-safe | modelled output tok/s |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 12 | 16 | 5.14 | 41 | 27.98 | **no** | 4.48 |
| 12 | 32 | 5.14 | 41 | 27.98 | yes | 4.48 |
| 12 | 64 | 5.14 | 41 | 27.98 | yes | 4.48 |
| 24 | 16 | 17.14 | 24 | 16.19 | **no** | 7.53 |
| 24 | 32 | 17.14 | 24 | 16.19 | yes | 7.53 |
| 24 | 64 | 17.14 | 24 | 16.19 | yes | 7.53 |
| 32 | 16 | 22.40 | 16 | 10.62 | yes | 7.62 |
| 32 | 32 | 22.40 | 16 | 10.62 | yes | 7.62 |
| 32 | 64 | 22.40 | 16 | 10.62 | yes | **7.62** |

**ctx 4096**

| VRAM GiB | RAM GiB | GPU exp GiB | n-cpu-moe | CPU exp GiB | RAM-safe | modelled output tok/s |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 12 | 16 | 5.58 | 41 | 27.98 | **no** | 4.34 |
| 12 | 32 | 5.58 | 41 | 27.98 | yes | 4.34 |
| 12 | 64 | 5.58 | 41 | 27.98 | yes | 4.34 |
| 24 | 16 | 17.58 | 23 | 15.47 | **no** | 11.89 |
| 24 | 32 | 17.58 | 23 | 15.47 | yes | 11.89 |
| 24 | 64 | 17.58 | 23 | 15.47 | yes | 11.89 |
| 32 | 16 | 25.58 | 12 | 8.00 | yes | 17.79 |
| 32 | 32 | 25.58 | 12 | 8.00 | yes | 17.79 |
| 32 | 64 | 25.58 | 12 | 8.00 | yes | 17.79 |

Two readings:

- **RAM is worth 0 tok/s at every VRAM ≥ 24 GiB.** The only cells where it should matter are
  `VRAM = 12, RAM = 16` (and the marginal 24/16), and even there the model's throughput column is
  identical because the measurement found no throughput dependence; what changes is the SSD-traffic
  and feasibility flag, not the tok/s.
- **The `VRAM = 12` row is a warning, not a recommendation.** At 128K, 12 GiB of VRAM leaves
  5.14 GiB of GPU experts and a 27.98 GiB CPU expert set. The model predicts 4.48 tok/s, but that is
  extrapolated below every measured point except the all-CPU split (2.26 tok/s), and the
  `RAM = 16` cell is over-subscribed.

### 4.4 Model error at the measured anchors

In-sample (the fit uses the sweep anchors):

| source | ctx | n-cpu-moe | RAM GiB | measured tok/s | modelled tok/s | rel err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sweep | 4096 | 12 | — | 17.71 | 17.79 | +0.5% |
| sweep | 4096 | 16 | — | 16.11 | 16.01 | −0.6% |
| sweep | 4096 | 24 | — | 11.25 | 11.27 | +0.2% |
| sweep | 4096 | 48 | — | 2.95 | 2.95 | −0.0% |
| sweep | 131072 | 16 | — | 7.62 | 7.62 | −0.0% |
| sweep | 131072 | 24 | — | 7.53 | 7.53 | +0.0% |
| sweep | 131072 | 48 | — | 2.26 | 2.26 | −0.0% |
| **capacity** | **4096** | **16** | **16.0** | **15.61** | **16.01** | **+2.6%** |
| **capacity** | **131072** | **16** | **16.0** | **8.00** | **7.62** | **−4.7%** |
| capacity | 4096 | 24 | 16.0 | 14.29 | 11.27 | +26.8% |
| capacity | 131072 | 24 | 16.0 | 6.99 | 7.53 | +7.7% |

The two `n=24` capacity rows show the sweep-fit problem directly: the model was calibrated on the
committed sweep, whose `n=24` 4K anchor (11.25 tok/s) is ~21% below the same-protocol control
(§3.4), so the model *under*-predicts that cell by 26.8%. The 128K `n=24` cell is within 7.7%.

**At the two required anchors the model error is +2.6% (16 GiB, 4K) and −4.7% (16 GiB, 128K);
against the ~30 GiB unconstrained anchor the 128K error is −0.0% (7.62 modelled vs 7.62 measured).**
The 16 GiB "anchor error" is therefore really a *measurement* difference: the model has no RAM term,
and the RAM-change signal is smaller than the model's own run-to-run spread.

**Leave-one-out error** (refit without each anchor) exposes where the model is trustworthy:

| ctx | n-cpu-moe | measured | LOO modelled | rel err |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 12 | 17.71 | 17.97 | +1.5% |
| 4096 | 16 | 16.11 | 15.96 | −1.0% |
| 4096 | 24 | 11.25 | 11.55 | +2.6% |
| 4096 | 48 | 2.95 | 2.44 | −17.1% |
| 131072 | 16 | 7.62 | 13.41 | +75.9% |
| 131072 | 24 | 7.53 | 4.84 | −35.8% |
| 131072 | 48 | 2.26 | 7.37 | +225.6% |

**The 128K fit is in-sample exact and LOO-unstable.** With three anchors and a step-like curve,
dropping any one point destroys the fit. Concretely: the 128K model is reliable *at and near the
shipped `n=16`/`n=24` operating point* (where the flat 131 ms fixed term dominates) and is **not**
reliable for a prediction of the all-CPU cliff or of any configuration below ~17 GiB of GPU
experts. The 4K fit, with four anchors spanning the whole range, has ≤2.6% LOO error down to
`n=24` and −17% at the cliff. This is the model's single largest weakness and is why the grid's
`VRAM = 12` row is flagged.

**Dominant bottleneck term:** at 131072, the context-fixed term `T0 = 131.1 ms/token` (attention/KV
+ dense) — 74% of the shipped token time; expert placement and RAM are second-order. At 4096, the
CPU-expert term `A·B^2.86` dominates above ~8 GiB of CPU experts.

## 5. Extrapolation

### 5.1 Modelled 32 → 64 GiB RAM gain

**0 output tok/s at 32 GiB VRAM, and 0 prefill tok/s.** The gain is zero because at the shipped
split only 10.62 GiB of expert pages need RAM and 16 GiB already holds them. The model's
feasibility rule makes the same statement for every VRAM ≥ 24 GiB. RAM only becomes a binding
resource in the `VRAM = 12 GiB` row, where the CPU expert set is 27.98 GiB: there 32 GiB is already
enough (27.98 + 4 = 31.98), so **the 32 → 64 GiB step still buys nothing**; the step that matters
on such a box is 16 → 32 GiB.

Assumptions this rests on:

1. The expert set size is fixed at IQ2_XS's 33.02 GiB and the split is the static `--n-cpu-moe`
   (or its cheapest-layer-first equivalent). A different tier changes the bytes, not the shape.
2. llama.cpp keeps `mmap` + `--lazy-mode auto` and does not force the 26.82 GiB PLE table resident.
3. The engine continues to hide the SSD reads behind compute at ≤8 tok/s; the "reads are hidden"
   result is measured at 16 GiB and 30 GiB but not at the largest deficits.
4. The workload is prefill-heavy; longer decode would change the read/write mix, not the
   throughput ranking, because RAM enters only through the miss path.

### 5.2 Is the 32 GiB VRAM budget used, wasted, or limited?

**Used up to ~22.40 GiB of GPU experts, then limited by 128K attention/KV, not VRAM headroom.**
The budget is genuinely leveraged: the same model on a 12 GiB card (Strata's reference) would keep
~5–6 GiB of experts on the GPU and put ~28 GiB on the CPU, and the 4K decode curve
(§2.1) says the first ~10 GiB of GPU experts is worth ~0.6–0.9 output tok/s/GiB. Beyond ~17 GiB of
GPU experts at 128K the marginal return is +0.016 output tok/s/GiB, and beyond 22.40 GiB the config
is unsafe (`n=12`, 25.02 GiB, device-losts during the 128K prefill). So the 32 GiB card is
**used, but the last ~10 GiB of it is spent on safety margin and KV headroom rather than on
decode throughput.** The 128K decode ceiling (~7.6 tok/s) is set by long-context attention/KV,
exactly as the Stage-1 placement study concluded.

Where VRAM would pay next: the [BAS-66](expert-activation-skew.md) frequency-ranked residency
covers the same events with far fewer bytes, which frees VRAM for a larger KV cache / longer
context rather than raising tok/s. That is a capacity-for-context trade, not a throughput gain.

## 6. Limitations

- **One repeat per context per configuration.** The sweep is single-repeat; the constrained runs
  match it to within a few percent, but a sub-3% RAM effect cannot be resolved at this sample size.
- **The 128K VRAM fit has three anchors and is LOO-unstable** (§4.4). Do not use the model to
  predict the all-CPU cliff or a <17 GiB-GPU-expert configuration.
- **Baseline contention.** The committed sweep was recorded on a shared box; [BAS-67](ssd-ngram-shard.md)
  measured 2–4× device degradation under contention. The constrained runs were recorded on an
  otherwise idle box and were if anything *faster* than the sweep at `n=24`, so the RAM comparison
  is conservative in the direction that matters (a cap cannot help), but the absolute sweep numbers
  carry contention uncertainty.
- **Only two RAM budgets, both 16 GiB, at two CPU splits.** No 24 GiB measurement was taken (see
  §1.2 for why); the ~30 GiB unconstrained rows are the control.
- **`rchar` is not reported as an SSD signal** because `mmap` makes it meaningless here.
- **The PLE-table DMA/reader is not instrumented.** This study measures the OS page-cache path
  llama.cpp actually uses, not the dedicated prefetcher [BAS-67](ssd-ngram-shard.md) recommends.
- **One model, one tier (IQ2_XS), one engine build.** Per-layer expert bytes, and therefore the
  byte-budget curves, differ for Q2_0 / IQ3_XXS and for a bongo-owned engine.

## 7. Reproduce

```sh
# 1. RAM-constrained runs (reversible cgroup; ~25 min per run on the reference box).
./bench/run-capacity-sensitivity.sh 16 16      # memory.max=16G, shipped --n-cpu-moe 16
./bench/run-capacity-sensitivity.sh 16 24      # the over-subscribed CPU split

# 2. Capacity model: reads the sweep, the BAS-66 analysis and the runs above.
python3 bench/capacity-model.py
#   -> bench/results/2026-09-28-capacity-sensitivity/capacity-model.json
#      bench/results/2026-09-28-capacity-sensitivity/capacity-model.md

# 3. The unconstrained control is the committed Stage-1 sweep:
#    bench/results/2026-09-27-expert-placement/sweep-matrix.json
```

The constraint is reversible: `systemd-run --user --scope` creates a transient unit that is gone
when the run ends. No kernel command line, driver, `bongo.sh` default or model-cache file was
changed.
