# `bench/results/2026-09-29-moe4all-b70/` — MoE4All/INFR full matrix (BAS-181)

Raw artifacts for the quant x context x MTP matrix of MoE4All/INFR on the Intel
Arc Pro B70. The analysis lives in
[`docs/research/moe4all-b70-matrix.md`](../../../docs/research/moe4all-b70-matrix.md).

## How to reproduce

```sh
# stop any llama-server first; one GPU job at a time
./bench/run-moe4all-b70.sh --all --cold
```

The driver holds `bench/gpu-lock.sh` around every case, runs each measurement in
its own `infr` process, and skips cases whose artifact already exists (so an
interrupted matrix resumes).

## Layout

| file | meaning |
| --- | --- |
| `raw/<case>.cmd` | the exact command for a single run (executed form) |
| `raw/<case>.log` | `infr` stderr (INFO trace + any MTP/phase summaries) |
| `raw/<case>.out` | `infr` stdout (the `--json` line, or the chat reply) |
| `raw/<case>.json` | the parsed `infr bench --json` object (`avg_ts`, `reps_ts`, …) |
| `raw/<case>.meta.json` | case metadata: model, ctx, cache, depth, exit, wall time, JSON |
| `raw/commands.txt` | append-only log of every exact invocation, in order |
| `hostdma-on-iq2xs-ctx4096-ub256-cache6GiB.log` | host-DMA device-lost reproduction (BAS-181 scope 3) |
| `upstream-issue-host-dma.md` | the ready-to-file upstream report (see the doc for the access gap) |

## Case naming

`<quant>_ctx<ctx>_cache<cache>_<metric>[_d<depth>]`:

- `pp512` — `infr bench -p 512 -n 0 -r 3`, prefill only.
- `tg128` — `infr bench -p 0 -n 128 -r 3`, decode at depth 0.
- `d<depth>_tg128` — decode at that context depth (`-d <depth>`).
- `iq2xs_ctx4096_<ordinary|mtp>_rep<N>` — MTP A/B one-shot chat turn
  (`infr run`, `--temp 0 --no-think`, `--max-new 32`), same prompt on both arms.

`--ctx`, `paging.cache`, `-u 512`, `--dev Vulkan1` and `INFR_NO_HOST_DMA=1` are
fixed in the driver; see `commands.txt` for the authoritative command line.
