# BAS-139 — MoE expert-cache engine (BAS-76 Step 2)

Dynamic VRAM LRU over RAM-pinned experts, seeded from the R4 offline frequency
profile and then tracked online. This directory holds the engine *recipe* and the
inputs for the on-GPU A/B.

## Engine revision

| | |
| --- | --- |
| Engine | llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), Vulkan |
| Patch | [`tools/patches/moe-expert-cache.patch`](../../../tools/patches/moe-expert-cache.patch) |
| Patch sha256 | `cc6032fdfee668fc4df5800757f405f256772495815b69190033ab9cb341dfe0` |
| Build | `tools/build-llama-vulkan-lru.sh ~/.bongo/engine/llama.cpp-lru llama-server` |
| Provenance | port of upstream draft ggml-org/llama.cpp#27861 (MIT) plus bongo profile-init + counters |
| Model | Swift-1.5-Qwen3.8-Flash-Next IQ2_XS (`qwen4exp`), 48 layers x 512 experts, 10 active |

The mechanism: per host-resident expert layer the engine allocates companion
`up/gate/down` cache tensors with `n_slots+1` experts in the device buffer of
that layer's router (last slot zero). An I32 table maps expert id to slot; the
device copy remaps the routing ids for a second `mul_mat_id` chain over the
cache, the host copy (`src[3]`) makes the CPU `mul_mat_id` skip cached ids. The
two down outputs sum, so the result is exact by construction. Uploads are
throttled (`--moe-expert-cache-inserts`) and asynchronous.

## Profile initialisation

`--moe-expert-cache-profile profile-iq2_xs-22.40.txt` seeds the cache: the top
`(layer, expert)` cells by captured count until the 22.40 GiB expert budget is
full, one `L <il> <expert> ...` line per layer (line length = that layer's slot
budget). The online LRU then tracks drift; the frozen profile is never the
policy.

| | |
| --- | --- |
| Budget | 22.40 GiB expert bytes (the `--n-cpu-moe 16` resident budget) |
| Cells | 16,735 |
| In-sample prefill coverage | 0.9852 |
| Metadata | `profile-iq2_xs-22.40.json` |
| Generator | `bench/gen-moe-cache-profile.py` |

Reproduce:

```sh
python3 bench/gen-moe-cache-profile.py \
  --raw bench/results/2026-09-28-expert-activation/raw \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --budget-gib 22.40 --corpora doc,code,chat,convo --tier iq2_xs \
  --out bench/results/2026-09-29-moe-cache-lru/profile-iq2_xs-22.40.txt \
  --json-out bench/results/2026-09-29-moe-cache-lru/profile-iq2_xs-22.40.json
```

## A/B protocol

`bench/run-moe-cache-ab.sh` runs both configs back to back on the shared single
Arc Pro B70 flock ([BAS-80](/BAS/issues/BAS-80)) through the same harness
(4K + 128K prefill/decode, VRAM, needle, prefix-cache path):

| config | binary | flags |
| --- | --- | --- |
| baseline | pinned Stage 0 | `--n-cpu-moe 16` (Stage-0 Vulkan) |
| lru | BAS-139 engine | `--n-cpu-moe 48 --moe-expert-cache-profile <profile> --moe-expert-cache-inserts 4 --moe-expert-cache-stats moe-cache-stats.json` |

The engine also exposes `llama_moe_cache_stats()` and the periodic JSON
`moe-cache-stats.json` snapshot (steps, hits, misses, hit rate, per-layer slots)
so the measured run carries engine-side coverage.

```sh
# build the engine, then the A/B
tools/build-llama-vulkan-lru.sh "$HOME/.bongo/engine/llama.cpp-lru" llama-server
bench/run-moe-cache-ab.sh --dry-run     # verify the two argv sets
bench/run-moe-cache-ab.sh               # both configs, one session
```

## Status

- Engine patch: **built** (`build-lru-vulkan/bin/llama-server`, Vulkan,
  `--help` shows `--moe-expert-cache{,-inserts,-profile,-stats}`).
- Profile: **generated** (16,735 cells, 22.40 GiB, coverage 0.9852).
- A/B selector + harness: **wired** (`bench/run-moe-cache-ab.sh --dry-run`
  verified).
- On-GPU A/B: **pending** — the single Arc holds the [BAS-130](/BAS/issues/BAS-130)
  warm-prefix run (`~/.bongo/gpu.lock.holder`). No result numbers are recorded
  here until that run completes.

Acceptance criteria ([BAS-139](/BAS/issues/BAS-139)): +15% turn TTFT and/or +20%
4K decode vs the Stage-0 `--n-cpu-moe 16` baseline, no >2% long-context
regression, output unchanged (needle), fallbacks (`--n-cpu-moe 16`, Step-1
byte-budget `-ot`) still selectable. The fallbacks are unchanged: the cache is
opt-in and both existing placement flags remain.
