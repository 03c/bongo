# BAS-139 — MoE expert-cache engine (BAS-76 Step 2)

Dynamic VRAM LRU over RAM-pinned experts, seeded from the R4 offline frequency
profile and then tracked online. This directory holds the engine *recipe* and the
inputs for the on-GPU A/B.

## Engine revision

| | |
| --- | --- |
| Engine | llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), Vulkan |
| Patch | [`tools/patches/moe-expert-cache.patch`](../../../tools/patches/moe-expert-cache.patch) |
| Patch sha256 | `196025b51ad0b37a423a1e642f078f6d32f45775d5f0278faa443addf22691e5` |
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
so the measured run carries engine-side coverage. `bench/run-moe-cache-ab.sh`
ends by running `bench/compare-moe-cache-ab.py` on the two raw `matrix.json` and
`prefix-cache.json` files, which writes `moe-cache-ab.json` + `moe-cache-ab.md`
and evaluates the acceptance criteria (≥ +20% 4K decode or ≤ −15% turn TTFT,
no long-context regression > 2%, needle on both sides).

```sh
# build the engine, then the A/B
tools/build-llama-vulkan-lru.sh "$HOME/.bongo/engine/llama.cpp-lru" llama-server
bench/run-moe-cache-ab.sh --dry-run     # verify the two argv sets
bench/run-moe-cache-ab.sh               # both configs, one session
```

## Status — measured A/B: **FAIL**

- Engine patch: **built** (`build-lru-vulkan/bin/llama-server`, Vulkan,
  `--help` shows `--moe-expert-cache{,-inserts,-profile,-stats}`).
- Profile: **generated** (16,735 cells, 22.40 GiB, coverage 0.9852).
- On-GPU A/B: **run 2026-09-29 03:2x-03:4xZ**, one session, one flock. Raw files
  below. **The engine misses every acceptance target.**

## Measured A/B (2026-09-29)

Same session, same box, shared flock. `A` = BAS-139 engine at
`--n-cpu-moe 48 --moe-expert-cache-profile profile-iq2_xs-22.40.txt`; `B` =
pinned Stage 0 at `--n-cpu-moe 16`. Engine `b11223` + patch `196025b5…`, tier
`iq2_xs`, q8 KV, flash-attn on, `Vulkan1`.

| metric | A (engine LRU) | B (Stage 0 n=16) | Δ |
| --- | ---: | ---: | ---: |
| 4K prompt tok/s | 2.519 | 14.824 | −83.0% |
| **4K decode tok/s** | **2.503** | **17.112** | **−85.4%** |
| 4K TTFT ms | 1266.8 | 257.4 | +392% |
| 128K prompt tok/s | 66.771 | 131.816 | −49.4% |
| **128K decode tok/s** | **3.041** | **8.044** | **−62.2%** |
| turn TTFT, 4K cached +512 (ms) | 43418.5 | 2979.1 | +1358% |
| needle | pass | pass | — |
| VRAM after load GiB | 30.92 | 30.86 | +0.06 |

Engine counters: 47 cached layers, 632 steps, 262,723 hits, 13,167 misses,
**hit rate 0.952**, per-layer slots 265–459.

Raw: `ncmoe-48-moe-lru/` (A) and `ncmoe-16-stage0/` (B) — `matrix.json`,
`raw/`, `prefix-cache/prefix-cache.json`, `server-flags.json`, plus
`moe-cache-stats.json` and the verdict in `moe-cache-ab.{md,json}`.

Reproduce the verdict from the raw files:

```sh
python3 bench/compare-moe-cache-ab.py \
  --a bench/results/2026-09-29-moe-cache-lru/ncmoe-48-moe-lru/matrix.json \
  --b bench/results/2026-09-29-moe-cache-lru/ncmoe-16-stage0/matrix.json \
  --a-prefix bench/results/2026-09-29-moe-cache-lru/ncmoe-48-moe-lru/prefix-cache/prefix-cache.json \
  --b-prefix bench/results/2026-09-29-moe-cache-lru/ncmoe-16-stage0/prefix-cache/prefix-cache.json \
  --cache-stats bench/results/2026-09-29-moe-cache-lru/moe-cache-stats.json \
  --out bench/results/2026-09-29-moe-cache-lru
```

### What the result says

The cache **engaged** and is **correct** (needle pass; VRAM after load 30.92 GiB
vs 6.33 GiB for a plain `--n-cpu-moe 48`, so ~24.6 GiB of cache tensors are
resident; 0.952 hit rate). It just does not buy throughput.

- Against the recorded **all-experts-on-CPU** control
  (`bench/results/2026-09-27-expert-placement`, same harness, stock binary,
  `--n-cpu-moe 48`: 4K decode 2.948, 128K decode 2.262 tok/s) the cache is flat
  at 4K and ~+34% at 128K — while Stage 0 is +480% / +256%. The 0.95 hit rate
  does not recover the residency the whole-layer rule gets for free.
- The design keeps a **CPU `mul_mat_id` per MoE layer** (it skips cached experts
  but still crosses the CPU/GPU boundary). At `--n-cpu-moe 48` all 48 layers pay
  that handoff every token, so the realised decode tracks the all-CPU regime, not
  the 0.66→0.99 coverage the offline sim predicted.
- The cache is **decode-only** (`n_tokens == 1`), so prefill gets no benefit and
  is much slower because every expert is host-resident. That alone breaks the
  128K prefill and the agentic turn-TTFT metric.

Hypotheses for the flat decode, in order: (1) per-layer CPU/GPU handoff dominates
and the two-chain split adds work without removing the boundary; (2) Vulkan
`mul_mat_id` over a ~350-slot quantised cache tensor is not the efficient
selected-row path; (3) copy/upload and scheduler overhead. The measurement here
cannot separate them; a controlled `n=48` ± cache run plus a graph-level profile
is the next diagnostic.

Acceptance ([BAS-139](/BAS/issues/BAS-139)): 4K decode ≥ +20% **fail**; turn TTFT
≤ −15% **fail**; no long-context regression **fail** (128K prefill −49%, 128K
decode −62%). The fallbacks (`--n-cpu-moe 16`, Step-1 byte-budget `-ot`) are
unchanged and still selectable.
