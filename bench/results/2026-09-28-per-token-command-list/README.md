# Per-token captured command list on the Arc Pro B70 (BAS-71)

Raw evidence for [BAS-71](/BAS/issues/BAS-71) — the one command-graph form that
[BAS-70](/BAS/issues/BAS-70) said is worth having on this stack: **one captured
`ze_command_list` per token holding ~2,000 real kernel nodes, submitted once**,
against the same kernels rebuilt (appended) into a fresh list per token.

BAS-70 measured the inputs to this claim ([raw](../2026-09-28-levelzero-submission/README.md),
[write-up](../../../docs/research/levelzero-submission.md)):

- one `zeCommandQueueExecuteCommandLists` costs ~1.4 µs **flat in N**;
- appending a node into a not-yet-closed list costs **0.88–1.36 µs/node**, paid once.

From those it derived that a captured list saves the append cost on every token:
`~1 µs/node × ~2,000 nodes ≈ 2.8 ms/token`. This directory measures that number
directly instead of deriving it.

- Box: Intel Arc Pro B70 (Battlemage G31, PCI `8086:e223`), Fedora 44 Server,
  kernel `7.0.13-200.fc44.x86_64`, `xe`; NEO/Level Zero `1.15.38646`, loader
  `1.28.6`, API 1.15, IGC `2.36.3+0`. No SYCL runtime / DPC++ compiler.
- Date: 2026-09-28 · Issue: [BAS-71](/BAS/issues/BAS-71) · Branch:
  `BAS-62-improve-speed-architecture`.

## Reproduce

One command (the script self-configures the Level Zero library paths and re-execs
itself, exactly like `bench/micro/levelzero_probe.py`):

```sh
python3 bench/micro/levelzero_per_token.py \
    --out bench/results/2026-09-28-per-token-command-list \
    --reps 1000 --warmup 50 --nodes 43,344,2064,4128
```

`43` is Strata's measured block-graph node count, so `43×1, 43×8, 43×48, 43×48×2`
= `43, 344, 2064, 4128`. Each node is a real `zeCommandListAppendLaunchKernel` of
`k_store(__global volatile int*, int)` (SPIR-V assembled by
[`spirv_kernels.py`](../../micro/spirv_kernels.py)); the kernel argument is set
before any launch, as BAS-70 requires.

## Files

| file | contents |
| --- | --- |
| `raw/per-token-command-list.json` | every raw sample (ns) per arm and N, plus the environment record and method description |
| `probe-run.txt` | stdout/stderr of the recorded run |
| `raw/time.txt` | `/usr/bin/time -v` record (wall clock, peak RSS) for the run |
| `raw/xe-journal.txt` | `xe` kernel messages during the run window (empty — no faults, no resets) |
| `raw/run-window.txt` | local start/end timestamps used to slice the journal |

## Method

For each N, all arms run over the **same** N-node sequence and the same queue,
timed on the submitting host thread with `time.perf_counter_ns`:

- **Arm A — captured.** One closed command list of N nodes, built **once**, then
  replayed `reps` times: `zeCommandQueueExecuteCommandLists` + `zeCommandQueueSynchronize`.
  This is the CUDA-graph-replay analogue; the rebuild cost is not paid per token.
- **Arm B — per-token rebuild.** Every rep appends all N nodes, closes the list,
  and submits + synchronises it the same way. Two variants:
  - `rebuild_reset` — one list object, `zeCommandListReset` + re-append per rep;
  - `rebuild_fresh` — `zeCommandListCreate`/`zeCommandListDestroy` per rep.
- **append-only.** Append N nodes into an open list, timed, never closed or
  submitted. This is the clean per-node capture cost (BAS-70's analogue), and it
  predicts `N × append` for the per-token rebuild.

`cold` = the first `--warmup` (50) reps of each series; `warm` = the remaining
1,000 measured reps. The very first rep of every series is also recorded as
`first_rep` (it contains process/pipeline first-touch).

## Results

Warm medians, µs. The saving a captured list buys is `Arm B − Arm A`, i.e. the
per-token append cost that a replay does not pay.

| N | A captured total | B rebuild-reset total | B rebuild-fresh total | B-reset − A | B-fresh − A | append-only µs/node | predicted `N×append` | captured per node |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 19.9 | 52.3 | 46.0 | 32.4 | 26.2 | 0.845 | 36.3 | 462 ns |
| 344 | 25.9 | 320.0 | 319.7 | 294.1 | 293.8 | 0.849 | 292.2 | 75 ns |
| 2064 | 129.3 | 1892.4 | 1907.1 | **1763.1** | **1777.8** | 0.857 | 1768.0 | 63 ns |
| 4128 | 253.5 | 3812.1 | 3783.6 | **3558.6** | **3530.1** | 0.869 | 3589.2 | 61 ns |

- **Submitting a captured list is flat in N** (submit warm median 1.39–1.40 µs
  for every N, reproducing BAS-70). The captured total grows only with GPU
  execution of the N trivial nodes: ~61–63 ns/node at N ≥ 2064.
- **The measured saving equals the append-only prediction to within ~1%.** The
  per-node delta (`(B − A) / N`) is a flat **0.85–0.86 µs/node** at every N,
  matching the standalone append measurement (0.845–0.869 µs/node).
- **Cold ≈ warm.** For every arm and N the cold-window median equals the warm
  median within noise (e.g. N=2064 rebuild-reset: 1898.8 µs cold vs 1892.4 µs
  warm). The only cold effect is the first rep of the whole process (Arm A N=43
  `first_rep` = 3616 µs, ~180× the warm value): module/pipeline/kernel first
  touch. It is paid once, not per token, and is recorded separately.

### Cold vs warm, medians (µs)

| N | A cold / warm | B-reset cold / warm | B-fresh cold / warm | append µs/node cold / warm |
| ---: | ---: | ---: | ---: | ---: |
| 43 | 19.8 / 19.9 | 54.9 / 52.3 | 46.2 / 46.0 | 0.845 / 0.845 |
| 344 | 25.8 / 25.9 | 319.7 / 320.0 | 319.8 / 319.7 | 0.847 / 0.849 |
| 2064 | 129.4 / 129.3 | 1898.8 / 1892.4 | 1893.1 / 1907.1 | 0.854 / 0.857 |
| 4128 | 253.6 / 253.5 | 3838.8 / 3812.1 | 3767.0 / 3783.6 | 0.854 / 0.869 |

## Verdict

**Yes, one captured command list per token beats rebuilding it per token — but
by ~1.76 ms/token at ~2,000 nodes, not the ~2.8 ms/token the derivation
predicted.**

- At N=2064 the captured list costs **0.129 ms/token**; rebuilding it costs
  **1.892 ms/token**; the capture saving is **1.76 ms/token** (fresh-list variant
  1.78 ms/token). The two rebuild variants are equivalent within noise, so the
  list-object reuse strategy does not matter.
- The derivation's `~2.8 ms` was the **upper end** of BAS-70's own 0.88–1.36
  µs/node range (`1.36 × 2064 ≈ 2.8 ms`). The measured append cost at this node
  count is **0.857 µs/node** — the **lower end** of that range — so the real
  saving is `0.857 × 2064 ≈ 1.77 ms`, i.e. about **63% of the derived figure**.
  The mechanism is confirmed; the headline number was over-predicted by ~1.6×.
- Scaling is linear and flat per node: 4128 nodes saves 3.55 ms/token; a ~1,000
  node token would save ~0.86 ms.
- The absolute cost of a replay is small: 129 µs/token at 2064 nodes, of which
  ~1.4 µs is the submission and the rest is the GPU running 2064 trivial nodes
  at ~63 ns each. Rebuilding is ~15× the replay cost.

### Caveat carried forward

This measures the **capture/append** saving only. The captured list in this
probe replays with *fixed* kernel arguments; a real token's kernels need
per-token values (activation/KV pointers, positions). The 1.76 ms/token is
realisable only if those arguments can be repointed without re-appending every
node (e.g. through a single indirection buffer whose contents are updated). If
each token forces a re-append of even a large fraction of the nodes, the saving
shrinks in proportion. That follow-up is not in scope for BAS-71.

## Box state after the run

- `raw/xe-journal.txt` is **empty**: no `ccs` engine resets, no faults, no lost
  device during the run window (`raw/run-window.txt`).
- The probe exited `0`; no orphan processes; no driver, firmware, or module
  changes were made (read-only on upstream repos; `xe`/firmware untouched).
- Wall clock 19.4 s, peak RSS 75 MB for all 16 measured cells (`raw/time.txt`).
