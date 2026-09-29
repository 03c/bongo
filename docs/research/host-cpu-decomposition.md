# M4.1 — the host/CPU critical path of the cached delta turn

Work for [BAS-144](/BAS/issues/BAS-144) (M4.1), part of [BAS-62](/BAS/issues/BAS-62).
Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`), Vulkan
device `Vulkan1`, tier `iq2_xs`, q8 KV, Stage 0 placement `--n-cpu-moe 16`,
`--flash-attn on`, `--ctx-size 131072`, `--parallel 1`.
Author: Coder (Paperclip). Date: 2026-09-29.

Builds on the M3.6 profile ([BAS-130](/BAS/issues/BAS-130)), which showed the
512-token warm-prefix delta turn is host/CPU-bound (80.1% non-GPU at 16K) but
could not name the components of that term. This document names them and reports
the lever trials.

Raw files: [`bench/results/2026-09-29-host-cpu/`](../../bench/results/2026-09-29-host-cpu/).
Reproduce with `bench/run-m4.1-host-cpu.sh` (one command, one GPU-lock hold,
resumable). The decomposition itself is `bench/profile-host-split.py` +
`bench/analyze-host-split.py`.

## Result in one page

- **The largest host term is the per-batch upload of the host-resident MoE
  expert weights to the Vulkan backend**, run on the llama.cpp main thread. The
  CPU backend worker threads are **idle (0 ms)** during the delta turn.
- **The M3.6 reading of `--n-cpu-moe` as "CPU-resident expert FFN" is wrong.**
  The weights are host-resident, but the matmul runs on the **GPU**: the
  scheduler's `MUL_MAT_ID` "copy only used experts" offload path uploads the
  used experts into the split's Vulkan buffer. `--no-op-offload`, which would
  force the matmul onto the CPU, costs **+62%**.
- The second host term is that the host weights are `mmap`-backed, so a
  page-cache eviction becomes **~0.5 GiB of SSD reads inside the turn**
  (~1.6–2.2 s of I/O wait on this DRAM-less NVMe). `--load-mode none` removes it.
- **Lever landed: `--load-mode none`** (opt-in; `bongo.sh --no-mmap`, which was
  **broken** against this engine and is now fixed). Same-session it cuts the
  turn by **−37.0% at 16K** and **−28.5% at 128K**. Against the frozen M3.6
  baselines it is **−10.4% / −7.1%, which misses the 15% target**.
- The residual fixed per-batch cost is the expert upload itself
  (~2–3 s of main-thread CPU per turn). No config flag removes it; it needs the
  host-resident expert count reduced (VRAM placement, [BAS-139](/BAS/issues/BAS-139))
  or a persistent VRAM expert cache.

## Method

The M3.6 tools split the turn into GPU-busy (Vulkan device timestamps) and
"everything else". This probe splits the "everything else" from `/proc`, which
needs no profiler and no kernel change:

- **Per-thread CPU time.** `/proc/<pid>/task/<tid>/stat` before and after the
  request gives the CPU time each thread spent on it. The CPU backend worker
  threads are named from the cold prefill of the same point (the busiest
  non-main threads); the main thread is the host thread that builds the graph,
  dispatches Vulkan, and does the scheduler copies.
- **Storage I/O.** `/proc/<pid>/io` `read_bytes` and the page-fault counters for
  the same window. The host-resident expert weights are `mmap`-backed, so a
  page-cache eviction shows up as a multi-GiB read inside one turn.
- **Server/client time.** `prompt_ms` and the streaming client wall for the same
  request, so the split is tied to the turn it belongs to.
- **Not-on-CPU residual.** `turn wall − Σ thread CPU` is GPU execution plus
  fence wait plus I/O wait; cross-check with `GGML_VK_PERF_LOGGER` GPU busy.

The component attribution is also cross-checked against the M3.6 ablations, which
are the independent leg: an ablation that moves a component off the fast GPU onto
the slow CPU (or changes the batch count) measures that component's contribution
to the critical path.

## The prior (M3.6) numbers this must explain

| measurement (16K, 512-token delta) | value |
| --- | ---: |
| wall `prompt_ms` | 3982.5 ms |
| Vulkan GPU busy | 793 ms (20%) |
| non-GPU ("host/CPU + sync") | 3190 ms (80%) |
| `--ubatch-size 128` (4 batches) | 9123.7 ms (**+129%**) |
| `--n-cpu-moe 24` (+8 CPU layers) | 4711.8 ms (+18.3%) |
| `hit` (4 new tokens, same prefix) | 231.3 ms |
| 128K, 512-token delta `prompt_ms` | 5899.5 ms |

The `ub128` result is the key one: four batches for the same 512 tokens costs
**+5.1 s**, so the turn is dominated by a cost paid *per batch* (~1.7 s), not per
token. A 128-token and a 512-token delta cost almost the same (3947.6 ms vs
3982.5 ms), so the per-batch cost saturates quickly and is not proportional to
the new-token count.

## Decomposition — named components

16K, prefix 16 384, delta 510, shipped baseline (`--n-cpu-moe 16`, mmap):

| component | ms | % of turn | basis |
| --- | ---: | ---: | --- |
| **main host thread CPU** (scheduler expert upload + Vulkan dispatch + driver) | **2 180** | **38.4%** | `/proc` main-thread CPU |
| **CPU backend worker threads** | **0** | **0.0%** | `/proc` threadpool CPU |
| other host threads (HTTP, trace) | 640 | 11.3% | `/proc` |
| **not-on-CPU** (GPU execution + fence wait + **storage I/O wait**) | **2 861** | **50.4%** | wall − Σ CPU |
| — of which **SSD page-cache re-reads** | ~1 600–2 100 | ~28–37% | `read_bytes = 0.52 GiB`, 1 517 major faults |
| wall | 5 680 | 100% | client |

128K, prefix 127 999, delta 516, shipped baseline:

| component | ms | % of turn |
| --- | ---: | ---: |
| main host thread CPU | 3 150 | 40.6% |
| CPU backend worker threads | 0 | 0.0% |
| other host threads | 840 | 10.8% |
| not-on-CPU (incl. 0.51 GiB storage read) | 3 766 | 48.6% |
| wall | 7 756 | 100% |

**There is no CPU-resident expert *compute* term.** The 8 CPU-backend threadpool
threads are idle; the only CPU work on the critical path is the host thread that
moves the used experts to the GPU. That is what the `ub128` fixed per-batch cost
is, and what moves under `ncmoe24`.

## Lever screen (16K, same session, same workload)

| config | delta-512 `prompt_ms` | Δ vs same-session baseline | reading |
| --- | ---: | ---: | --- |
| `baseline` | 5 584.7 | — | mmap, page cache cold |
| **`lm_none` (`--load-mode none`)** | **3 569.4** | **−36.1%** | host weights in anonymous RAM |
| `no_op_offload` (`--no-op-offload`) | 9 047.9 | +62.0% | CPU expert matmul is far slower |
| `threads16` (`--threads 16`) | 5 648.9 | +1.1% | no CPU compute to accelerate |

The `no_op_offload` result is the proof of the mechanism: forcing the expert
matmul onto the (idle) CPU threads is *much* worse than uploading the used
experts and computing on the GPU, so the upload is the term to attack, and the
GPU path is the right one.

## The lever: `--load-mode none`

`--load-mode none` reads the host-resident expert tensors into anonymous RAM
instead of an `mmap` of the GGUF. Anonymous pages are not evicted as eagerly as
clean page-cache pages, so the weights stay resident across the session.

| prefix | baseline `prompt_ms` (same session) | `--load-mode none` | Δ | vs frozen M3.6 baseline | storage read after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 384 | 5 666.9 | **3 569.4** | **−37.0%** | 3 982.5 → **−10.4%** | 0.52 → **0.00 GiB** |
| 127 999 | 7 661.7 | **5 480.6** | **−28.5%** | 5 899.5 → **−7.1%** | 0.51 → **0.00 GiB** |

After the lever the turn is `main host 1 970 / 2 940 ms`, `workers 0`,
`not-on-CPU 902 / 1 926 ms`, storage read **0** at both prefixes.

The frozen M3.6 baselines (3 982.5 / 5 899.5 ms) were measured with a warm page
cache: they did not pay the ~1.6–2.2 s I/O wait that every fresh session now
pays. So the lever's **steady-state** advantage over a warm baseline is the
~10% / ~7% shown against the frozen numbers; its larger same-session win is the
removal of the cold-cache penalty, which is what makes the number reproducible
across sessions.

`bongo.sh` already had a `--no-mmap` flag for exactly this, but it passed
`--no-mmap`, which the pinned `b11223` server rejects (`error: invalid argument:
--no-mmap`) — the flag could not start a server. It is fixed to
`--load-mode none`, and a real `--load-mode MODE` passthrough was added. The
default is unchanged (no `--load-mode` is emitted), so the Stage 0
`--ctx 131072 --n-cpu-moe 16` baseline stays selectable.

## Acceptance criteria

| criterion | result |
| --- | --- |
| host/CPU cost decomposition with named components, ms and %, raw files | **MET** (above; `bench/results/2026-09-29-host-cpu/`) |
| >= 15% cut of the 16K delta turn vs 3 982.5 ms | **MISSED**: 3 569.4 ms = −10.4% |
| >= 15% cut of the 128K delta turn vs 5 899.5 ms | **MISSED**: 5 480.6 ms = −7.1% |
| no long-context regression > 2% | **MET** (no regression; both prefixes improve) |
| correctness unchanged (needle pass) | **MET**: `baseline` and `lm_none` both recall the sentinel at 8 239 prompt tokens (`bench/results/2026-09-29-host-cpu/needle/`) |
| no change to shipped defaults; Stage 0 baseline selectable | **MET** (`--load-mode` is opt-in; default emits nothing) |
| flag-gated and revertible | **MET** (`--load-mode none` / `--no-mmap`; `bongo.sh` default unchanged) |

### Why the 15% target is missed

After `--load-mode none` the residual fixed per-batch cost is the **expert
upload itself**: `main host` is 1 970 ms (16K) / 2 940 ms (128K) of the turn.
That term is set by how many host-resident expert weights must cross to the GPU
per MoE layer per batch. Removing it needs the host-resident expert count
reduced — VRAM placement, [BAS-139](/BAS/issues/BAS-139)'s dynamic LRU expert
cache, or a persistent VRAM expert cache — none of which is a server flag. The
M3.6 `--n-cpu-moe 8` ablation (8 fewer host layers) **failed to load at 128K**
(VRAM), so the placement lever is at its VRAM edge. That is the measured
cross-issue conclusion: the M4.1 host/CPU lever available today buys ~10%, and
the remaining gap is the placement half.

## Reproduce

```sh
# one command, one GPU-lock hold, resumable
bench/run-m4.1-host-cpu.sh

# the tables
bench/analyze-host-split.py --root bench/results/2026-09-29-host-cpu/ctx16k  --prefix 16384  --delta 512
bench/analyze-host-split.py --root bench/results/2026-09-29-host-cpu/ctx128k --prefix 128000 --delta 512

# the needle guard
bench/run-needle-check.sh   # BONGO_NEEDLE_CONFIGS="baseline no_op_offload threads16 lm_none"

# the bongo.sh unit tests
bash tests/bongo-sh.test.sh
```

## Caveats

- Single measurement per config (no repeats). The per-component split is far
  larger than the run-to-run noise; the absolute delta-turn numbers move with the
  host page-cache state, which is why the same-session A/B and the frozen
  baseline are both reported.
- `/proc` CPU time is a 10 ms-granularity accounting number (`CLK_TCK=100`); the
  short `hit` case (≈250 ms) has a negative `not-on-CPU` residual because the
  lazily-created threadpool threads appear inside its window. The delta turn
  (≈5.7 s) is not affected.
- The box is shared with other agents' measurement runs; the GPU runs are
  serialised by the shared flock ([BAS-80](/BAS/issues/BAS-80)), but CPU and
  NVMe load from non-GPU work is not excluded.
- `--load-mode none` holds the host-resident experts in anonymous RAM
  (~11.8 GiB at `--n-cpu-moe 16`), so it trades RAM for latency. It is opt-in.
