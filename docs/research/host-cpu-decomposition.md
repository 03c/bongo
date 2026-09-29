# M4.1 — the host/CPU critical path of the cached delta turn

Work for [BAS-144](/BAS/issues/BAS-144) (M4.1), part of [BAS-62](/BAS/issues/BAS-62).
Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`), Vulkan
device `Vulkan1`, tier `iq2_xs`, q8 KV, Stage 0 placement `--n-cpu-moe 16`,
`--flash-attn on`, `--ctx-size 131072`, `--parallel 1`.
Author: Coder (Paperclip). Date: 2026-09-29.

Builds on the M3.6 profile ([BAS-130](/BAS/issues/BAS-130)), which showed the
512-token warm-prefix delta turn is host/CPU-bound (80.1% non-GPU at 16K) but
could not name the components of that term. This document names them, lands one
flag-gated lever, and measures before/after.

Raw files: [`bench/results/2026-09-29-host-cpu/`](../../bench/results/2026-09-29-host-cpu/).
Reproduce with `bench/run-m4.1-host-cpu.sh` (one command, one GPU-lock hold,
resumable). The decomposition is `bench/profile-host-split.py` +
`bench/analyze-host-split.py`; the lever runner is
`bench/run-warm-prefix-profile.sh`; the correctness guard is
`bench/run-needle-check.sh`.

## TL;DR

1. **The host/CPU term is not CPU compute.** With `--n-cpu-moe 16` the CPU
   backend worker threads execute **0 ms** of every measured request (cold, hit,
   delta, decode). The host-resident MoE experts are not computed on the CPU;
   they are read by the GPU.
2. **The single largest host term is the main-thread host path** — graph
   build, Vulkan op dispatch and the handling of the host-resident expert
   weights — **2 180 ms of the 5 667 ms 16K delta turn (38.4%)** and **3 150 ms
   of the 7 662 ms 128K turn (41.1%)**.
3. **The second is an in-turn page-cache re-read of the mmap'd expert
   weights.** With the default `--load-mode auto` (mmap) the host-resident
   experts are a file mapping; under the 30 GiB box's normal memory pressure
   the delta turn re-reads **0.52 GiB from the NVMe in one turn** (1 535 major
   faults) and pays **~1.9 s** of I/O wait for it. That is the fixed per-batch
   cost M3.6 measured as ~1.7 s.
4. **Lever: `--load-mode none`.** With mmap off, the `-ot`/`--n-cpu-moe` CPU
   override selects the device's **host buffer type** (Vulkan host-visible
   memory) instead of the plain CPU buffer, so the GPU reads the weights in
   place. The same-session delta turn drops **−36.1% at 16K** and **−28.5% at
   128K**, and the in-turn re-reads go to ~0. The needle still passes.
5. **Against the frozen M3.6 baselines the 15% target is missed**: 16K
   3 569.4 ms vs 3 982.5 ms (**−10.4%**) and 128K 5 480.6 ms vs 5 899.5 ms
   (**−7.1%**). The mmap baseline is box-state dependent: in this session the
   *baseline* config measured 5 584.7 / 7 661.7 ms, i.e. 1.6–1.8 s worse than
   the M3.6 session with the same flags. See "Target check" below.

## Method

The M3.6 tools split the turn into GPU-busy (Vulkan device timestamps) and
"everything else". This probe splits the "everything else" from `/proc`, which
needs no profiler and no kernel change:

- **Per-thread CPU time.** `/proc/<pid>/task/<tid>/stat` before and after a
  request gives the CPU time each thread spent on it. The CPU backend worker
  threads are identified from the cold prefill of the same point (the busiest
  non-main threads); the main thread is the host thread that builds the graph,
  dispatches Vulkan and does the scheduler copies.
- **Storage I/O.** `/proc/<pid>/io` `read_bytes` and the page-fault counters for
  the same window. The host-resident expert weights are mmap'd from the GGUF, so
  a page-cache eviction shows up as a multi-hundred-MiB read inside one turn.
- **Server/client time.** `prompt_ms` and the streaming client wall for the same
  request, so the split is tied to the turn it belongs to.
- **Not-on-CPU residual.** `turn wall − Σ thread CPU` is GPU execution plus
  fence wait plus I/O wait. It can be negative for short cases where the
  thread-pool spin across 8 threads exceeds the wall.

The attribution is cross-checked two ways: the M3.6 ablations (the independent
leg) and the lever itself, which removes exactly one mechanism and moves exactly
the component that mechanism owns.

## The prior (M3.6) numbers this has to explain

| measurement (512-token delta) | value |
| --- | ---: |
| 16K `prompt_ms` | 3 982.5 ms |
| 16K Vulkan GPU busy | 793 ms (20%) |
| 16K non-GPU ("host/CPU + sync") | 3 190 ms (80%) |
| `--ubatch-size 128` (4 batches, 16K) | 9 123.7 ms (**+129%**) |
| `--n-cpu-moe 24` (+8 host layers, 16K) | 4 711.8 ms (+18.3%) |
| 16K `hit` (4 new tokens) | 231.3 ms |
| 128K `prompt_ms` | 5 899.5 ms |

`ub128` is the key result: four batches for the same 512 tokens costs **+5.1 s**,
so the turn is dominated by a cost paid *per batch* (~1.7 s). A 128-token and a
512-token delta cost almost the same (3 947.6 vs 3 982.5 ms), so the per-batch
cost saturates quickly and is not proportional to the new-token count.

## Result 1 — the named decomposition

Measured this session with `bench/profile-host-split.py`; raw files
`ctx16k/<config>/profile-host-split.json` and `ctx128k/<config>/…`.

### 16K, prefix 16 384, delta 510 (`grow_p16384_d512`)

| component | baseline (mmap) ms | % of turn | `--load-mode none` ms | % |
| --- | ---: | ---: | ---: | ---: |
| **main host thread** (graph build, Vulkan dispatch, host-weight handling, server) | **2 180** | **38.4** | 1 970 | 55.2 |
| **CPU backend worker threads** | **0** | **0.0** | **0** | **0.0** |
| other threads (HTTP / thread-pool spin) | 640 | 11.3 | 710 | 19.9 |
| **not-on-CPU** (GPU exec + fence wait + I/O wait) | **2 861** | **50.4** | 902 | 25.3 |
| — of which: in-turn storage reads | 0.516 GiB | — | 0.002 GiB | — |
| — of which: major faults | 1 535 | — | 426 | — |
| turn wall (client) | 5 680.6 | 100 | 3 582.3 | 100 |
| server `prompt_ms` | 5 666.9 | — | 3 569.4 | — |
| process RSS after the turn | 10.15 GiB | — | 0.89 GiB | — |
| cold 16K prefill `prompt_ms` | 84 557 | — | 82 519 | — |
| `hit` `prompt_ms` (4 tokens) | 235.7 | — | 228.8 | — |

### 128K, prefix 127 999, delta 516 (`grow_p127999_d512`)

| component | baseline (mmap) ms | % of turn | `--load-mode none` ms | % |
| --- | ---: | ---: | ---: | ---: |
| **main host thread** | **3 150** | **40.6** | 2 940 | 52.7 |
| **CPU backend worker threads** | **0** | **0.0** | **0** | **0.0** |
| other threads | 840 | 10.8 | 710 | 12.7 |
| **not-on-CPU** | **3 766** | **48.6** | 1 926 | 34.5 |
| — of which: in-turn storage reads | 0.510 GiB | — | 0.002 GiB | — |
| — of which: major faults | 1 718 | — | 426 | — |
| turn wall (client) | 7 755.6 | 100 | 5 576.2 | 100 |
| server `prompt_ms` | 7 661.7 | — | 5 480.6 | — |
| process RSS after the turn | 10.45 GiB | — | 0.93 GiB | — |
| cold 128K prefill `prompt_ms` | 963 967 | — | 957 773 | — |
| `hit` `prompt_ms` (4 tokens) | 402.2 | — | 393.9 | — |

Reading the table:

- **The CPU backend does nothing.** Worker CPU is 0 ms in every case at both
  contexts, including the 84-second cold prefill (main = 48 640 ms, workers =
  0 ms). `--n-cpu-moe N` does not compute the experts on the CPU; it keeps their
  weights in host memory and the MoE matmul runs on the GPU. This refines the
  M3.6 wording "CPU-resident expert FFN": the *weights* are host-resident, the
  *compute* is not on the CPU.
- **The largest named host term is the main-thread path** (38–41% of the turn).
- **The second is the in-turn re-read of the mmap'd host experts.** The lever
  removes 0.51 GiB of reads, ~1 100 major faults, and **1 958 ms (16K) /
  1 840 ms (128K)** of not-on-CPU time. Effective read throughput of the faulted
  pages is ~270 MB/s, which is the DRAM-less KIOXIA EXCERIA G3's random-read
  behaviour, not the ~2 GB/s sequential figure.
- **GPU execution is the rest of not-on-CPU** (~793 ms at 16K per the M3.6 perf
  logger; ~2.7 s at 128K derived from the +1 917 ms prefix increment).
- Serving/tokenisation is small: the `hit` case (4 new tokens) is 229–236 ms at
  16K and 394–402 ms at 128K, i.e. ≤7% of the turn.

Why the lever changes the memory placement: with `use_mmap=true` the loader
forces a CPU-overridden tensor off the device host buffer onto the plain CPU
buffer (llama.cpp `src/llama-model-loader.cpp`: "avoid using a host buffer when
using mmap"), and prints the warning
`tensor overrides to CPU are used with mmap enabled - consider using --load-mode none`.
With `--load-mode none` that branch is skipped and `make_cpu_buft_list`'s first
entry — the device host buffer type, `VK_EXT_external_memory_host`-style
host-visible memory — is selected, so the GPU reads the weights in place instead
of the scheduler copying the used experts into VRAM per split. The RSS drop
(10.1 → 0.9 GiB) is the same fact seen from the process side: the weights are no
longer file-backed anonymous pages.

## Result 2 — the lever screen (16K, same session, sequential restarts)

Shipped baseline flags plus exactly one change per config; prefix 16 384,
delta 512; `bench/run-warm-prefix-profile.sh`.

| config | change | cold 16K ms | hit ms | delta 512 ms | Δ vs session baseline | Δ vs frozen M3.6 (3 982.5) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | — | 84 204.6 | 229.3 | 5 584.7 | — | +40.2% |
| `lm_none` | `--load-mode none` | 83 401.0 | 234.5 | **3 569.4** | **−36.1%** | **−10.4%** |
| `no_op_offload` | `--no-op-offload` (expert matmul on CPU) | 226 044.9 | 244.0 | 9 047.9 | +62.0% | +127.2% |
| `threads16` | `--threads 16 --threads-batch 16` | 84 263.6 | 252.3 | 5 648.9 | +1.1% | +41.9% |

- **`--no-op-offload` is the proving negative.** It forces the host-resident
  expert matmul onto the CPU backend; the cold prefill goes 84 s → 226 s and the
  delta turn 5 585 → 9 048 ms. GPU-side compute with host-resident weights is
  strictly better than CPU compute for this workload, which is why the CPU
  backend stays idle.
- **`--threads 16` is neutral** for the turn (the worker pool is idle) and costs
  decode throughput (11.3 → 5.4 tok/s), so it is rejected.
- **`--load-mode none` is the lever.** It also speeds the cold prefill slightly
  (84.2 → 83.4 s) and does not change `hit`.

## Result 3 — correctness

`bench/run-needle-check.sh` plants the standard bongo sentinel in an 8 239-token
document and asks for it greedily (raw files `needle/<config>/needle.json`):

| config | status | answer |
| --- | --- | --- |
| `baseline` | **pass** | `…VAULT-COORD-7391-QXZ` |
| `lm_none` | **pass** | `…VAULT-COORD-7391-QXZ` |

The lever is a configuration change only (no weight, kernel or arithmetic
change), so the same answer is expected; the check confirms it.

## Target check

Acceptance target: **≥15% reduction of the 16K and 128K delta-turn `prompt_ms`
versus the M3.6 baseline (3 982.5 / 5 899.5 ms).**

| prefix | frozen M3.6 baseline | `--load-mode none` | reduction | target |
| ---: | ---: | ---: | ---: | --- |
| 16 384 | 3 982.5 ms | 3 569.4 ms | **−10.4%** | ≥15% → **missed by ~184 ms** |
| 127 999 | 5 899.5 ms | 5 480.6 ms | **−7.1%** | ≥15% → **missed by ~467 ms** |

Same-session control (the baseline config re-measured in this session, same
flags, same harness, immediately before the lever):

| prefix | session baseline | `--load-mode none` | reduction |
| ---: | ---: | ---: | ---: |
| 16 384 | 5 584.7 ms (host-split: 5 666.9) | 3 569.4 ms | **−36.1%** (−37.0%) |
| 127 999 | 7 661.7 ms | 5 480.6 ms | **−28.5%** |

**The lever is real and large, but the absolute 15% target is not met.**
The two readings disagree because the *baseline* is box-state dependent: with
the same flags the delta turn measured 5 584.7 ms this session and 3 982.5 ms in
the M3.6 session. The mechanism is the page-cache state of the 63 GiB GGUF
against 30 GiB of RAM: after a day of runs (BAS-130/132/139/145 output, slot
files, page cache), a larger fraction of the host-resident expert range is
evicted, so more of the turn's fixed per-batch cost is NVMe random reads.
`--load-mode none` is immune to that state, so its number is stable; the
baseline's is not.

Read this as: **the lever removes a 1.6–2.0 s box-state penalty that the shipped
default can incur, and its absolute number beats even the best frozen baseline
by 7–10%.** It does not by itself reach the ≤3 s / ≤5 s product targets.

## What remains — the next lever

After the lever the single largest host term is still the **main-thread host
path**: 1 970 ms at 16K and 2 940 ms at 128K (55% / 53% of the turn). It is
Vulkan graph dispatch plus the host-side handling of the host-resident expert
weights, paid once per batch. The M3.6 `ub128` result says a second batch costs
~1.7 s; this measurement says ~0.9 s of that is storage re-reads now removed,
leaving ~0.8 s per batch of main-thread host work plus GPU sync. Reducing that
needs engine-side work (a host-side phase profiler in the Vulkan backend, then
fewer/cheaper per-node host operations), not another config flag. That is the
recommended next M4 step.

## Shipped defaults / rollback

- No shipped default changes. `bench/run-m4.1-host-cpu.sh` is measurement-only;
  the engine pin is untouched; the Stage 0 Vulkan `--ctx 131072 --n-cpu-moe 16`
  baseline is still the default and still selectable.
- The lever is selectable and revertible through the engine's own `--load-mode`
  (already exposed as `bongo.sh --load-mode MODE`, with `--no-mmap` as an alias,
  in commit `294b5e6`). Removing the flag restores the previous behaviour.
- The pinned `b11223` `llama-server` rejects `--no-mmap` ("invalid argument");
  `--load-mode none` is the working spelling on this engine.

## Caveats

- Single measurement per config (no repeats). The 16K session baselines differ
  by 1.5% between the two measurement paths (`profile-host-split` vs
  `profile-warm-prefix`); the lever effect is 30–37%, far above that.
- `not-on-CPU` is a residual: it holds GPU execution, fence waits and I/O wait,
  and for the short `hit` cases it goes negative because eight thread-pool
  threads spin while the wall is ~250 ms.
- The 128K storage term (1.84 s) is isolated by the lever difference, not by a
  separate I/O profiler; the raw `read_bytes`/`majflt` counters are committed.
- The 128K GPU-busy split is the M3.6-derived number, not re-profiled here.
