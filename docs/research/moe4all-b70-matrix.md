# MoE4All/INFR on the Arc Pro B70 — full quant × context × MTP matrix

Status: complete for [BAS-181](/BAS/issues/BAS-181) (child of [BAS-179](/BAS/issues/BAS-179)).
Date 2026-09-29. Author: Coder. Supersedes the single-run numbers in
[`moe4all-arc-b70-eval.md`](moe4all-arc-b70-eval.md) with 3-rep measurements and a
paired MTP A/B.

Raw artifacts, exact commands and the generated table are committed under
[`bench/results/2026-09-29-moe4all-b70/`](../../bench/results/2026-09-29-moe4all-b70/README.md).

## Method

- Engine: MoE4All/INFR `0.9.0`, commit `ed62393068679573afe94a1472454efe7eae0f15`
  (`release-0.9.0`), built from source.
- Device: Intel Arc Pro B70 (BMG G31, `8086:e223`, 31.9 GiB), Mesa ANV
  `26.1.8-1.fc44`, kernel `7.0.13-200.fc44.x86_64`.
- Every run: `--dev Vulkan1`, `-u 512`, Q8 K/V, `INFR_NO_HOST_DMA=1`, greedy.
- `on GPU` — one case per process, serialized with `bench/gpu-lock.sh`.
- **`infr bench` times one metric per process** (prefill when `-n 0`, decode
  when `-p 0`), so a (quant, ctx) cell is two runs. `-r 3` gives three reps in
  one process; `reps_ts` is the per-rep list.
- **`--ctx` is the engine capacity flag, not the measured depth.** At the
  measured depth it does not change the VRAM plan (see "What `ctx` does"), so
  the matrix also reports **decode at a real depth** (`-d`), which is the
  apples-to-apples metric against bongo's "4K decode".
- **Quant tiers:** `iq2xs` (GSQ-RCO IQ2_XS), `q2_0` (GSQ-RCO Q2_0), and
  `iq3xs` (downloaded from `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF`,
  IQ3_XS, 91.95 GB — disk headroom allowed it; 51 GB free afterwards).

## Quant × context (flag) — prefill + decode, depth 0

`paging.cache=16GiB`, `--ctx` 4096 / 32768, 3 reps. `avg` is INFR's own average
over the 3 reps; the `reps` column is the spread, which matters here.

| quant | `--ctx` | metric | avg tok/s | reps (tok/s) | median | min–max |
| --- | ---: | --- | ---: | --- | ---: | --- |
| IQ2_XS | 4096 | pp512 | 40.57 | 40.84, 40.72, 40.16 | 40.72 | 40.16–40.84 |
| IQ2_XS | 4096 | tg128 | 3.37 | 2.97, 4.80, 2.33 | 2.97 | 2.33–4.80 |
| IQ2_XS | 32768 | pp512 | 40.62 | 41.24, 40.27, 40.36 | 40.36 | 40.27–41.24 |
| IQ2_XS | 32768 | tg128 | 2.86 | 2.00, 4.42, 2.16 | 2.16 | 2.00–4.42 |
| Q2_0 | 4096 | pp512 | 45.38 | 43.96, 46.12, 46.06 | 46.06 | 43.96–46.12 |
| Q2_0 | 4096 | tg128 | 2.90 | 2.28, 3.88, 2.55 | 2.55 | 2.28–3.88 |
| Q2_0 | 32768 | pp512 | 38.42 | 46.84, 46.36, 22.06 | 46.36 | 22.06–46.84 |
| Q2_0 | 32768 | tg128 | 2.95 | 2.97, 2.10, 3.79 | 2.97 | 2.10–3.79 |
| IQ3_XS | 4096 | pp512 | 14.96 | 14.87, 15.34, 14.68 | 14.87 | 14.68–15.34 |
| IQ3_XS | 4096 | tg128 | 1.28 | 1.51, 1.07, 1.26 | 1.26 | 1.07–1.51 |
| IQ3_XS | 32768 | pp512 | 14.89 | 15.17, 14.93, 14.58 | 14.93 | 14.58–15.17 |
| IQ3_XS | 32768 | tg128 | 1.67 | 1.55, 1.67, 1.79 | 1.67 | 1.55–1.79 |

**Reading it.** Prefill is reproducible to a few percent (except one `q2_0`
`--ctx 32768` rep at 22.06, which drags that average down — treat 46 as the
steady value). Depth-0 decode is **not** reproducible at 3 reps: the per-rep
spread is up to 2× (2.00–4.80 tok/s). Depth-0 decode is therefore only an
order-of-magnitude signal; the depth table below is the usable number.

### What `ctx` does (and does not) change

The VRAM plan is identical for `--ctx 4096` and `--ctx 32768` at depth 0 — the
KV cache is grown lazily in 32K-token segments, so the measured working set
(`ctx=145` at `tg128`) is the same and the plan is:

```
expert_cache_target=16.00 GiB elastic_pool=16.97 GiB (dynamic-32k k=Q8_0, v=Q8_0, ctx=145)
```

So for this workload `--ctx` is a capacity ceiling, not a cost. The real
context sensitivity comes from the measured **depth**.

### Long-prompt prefill (pp32000)

`pp512` is the specified llama-bench point but it is short and
dominated by per-batch overhead; a real agentic prompt is much longer. At
`--ctx 32768`, `-p 32000 -n 0 -r 3` measures a realistic long prefill, and it is
**2.5–3.3× the `pp512` rate** because the batched prefill streams each expert
block once for thousands of rows instead of once for 512:

| quant | pp512 | pp32000 | reps (pp32000) | bongo 32K prompt | INFR/bongo |
| --- | ---: | ---: | --- | ---: | ---: |
| IQ2_XS | 40.62 | **101.66** | 101.49, 101.82, 101.67 | 174.55 | 0.58× |
| Q2_0 | 38.42 | **128.11** | 128.06, 126.82, 129.45 | 207.97 | 0.62× |

The `pp512` / `pp32000` ratio is the same order as bongo's (bongo's 1K→32K
prefill drops only ~25%, so its long prompt is also fast), but it means the
"prefill is 5–6× slower" headline is only true for *short* prompts. On a 32K
prompt INFR is **~1.6–1.7× slower**, not 5–6×.

## Decode at real depth — the comparable metric

`-d <depth>` pre-fills that many tokens untimed, then times `tg128`. This is
the same shape as bongo's "4K decode" (decode after a context of that size).

| quant | depth | avg tok/s | reps (tok/s) | median | bongo Stage 0, same tier |
| --- | ---: | ---: | --- | ---: | ---: |
| IQ2_XS | 0 | 3.37 | 2.97, 4.80, 2.33 | 2.97 | 19.94 @ 1K / 17.71 @ 4K |
| IQ2_XS | 4096 | 6.25 | 6.55, 3.90, 8.30 | 6.55 | **19.56 @ 4K** (M4 default) / 17.71 (Stage 0) |
| IQ2_XS | 8192 | 9.35 | 8.33, 9.35, 10.36 | 9.35 | — |
| IQ2_XS | 32768 | **12.54** | 12.47, 12.68, 12.49 | 12.49 | **11.67 @ 32K** |
| Q2_0 | 0 | 2.90 | 2.28, 3.88, 2.55 | 2.55 | 14.79 @ 1K |
| Q2_0 | 4096 | **13.68** | 13.58, 13.30, 14.16 | 13.58 | **13.24 @ 4K** |
| Q2_0 | 32768 | **15.77** | 15.75, 15.65, 15.91 | 15.75 | **10.36 @ 32K** |
| IQ3_XS | 0 | 1.28 | 1.51, 1.07, 1.26 | 1.26 | — |
| IQ3_XS | 4096 | 4.15 | 4.49, 3.81, 4.13 | 4.13 | — |

bongo references from `bench/results/2026-09-27-baseline/matrix.md` (iq2_xs)
and `bench/results/2026-09-28-q2_0/matrix.md` (q2_0); the M4 shipped default
figure is the iq2_xs 4K decode from `docs/final-overview.md`.

**Decode throughput rises with depth.** More pre-filled tokens means a warmer
expert pager (the long prefill routes and caches the prompt's active experts),
so the decode step is a cache hit more often. This is the opposite of the
llama.cpp/bongo curve and is the single most important result here: the
"3.4 tok/s decode" from the first pass was the cold-routing depth-0 number, not
the steady-state 4K decode.

- **At depth, `q2_0` reaches parity with bongo on its own tier and beats it at
  32K.** INFR 13.68 @ 4K vs bongo `q2_0` 13.24 @ 4K (+3%, within noise); INFR
  **15.77 @ 32K vs bongo `q2_0` 10.36 @ 32K (+52%)**. The depth-32768 cells are
  the tightest in the whole matrix (reps within ±1%).
- **`iq2xs` also overtakes bongo at 32K**: 12.54 vs 11.67 (+7.5%), while staying
  well behind at 4K (6.25 vs 17.71/19.56). The GSQ-RCO IQ2_XS decode is
  unusually depth-sensitive.
- `q2_0` is the best INFR tier at every depth. Its expert payload is one uniform
  size class (`shared[0.4 MiB] x73728`), which the pager handles better than
  `iq2xs`'s four classes.
- `iq3xs` (IQ3_XS) is 4.3× slower than `q2_0` at 4K — its expert payload is
  55.66 GiB / 3 size classes, so far less of it is resident.

## `paging.cache=22GiB` point (IQ2_XS, `--ctx` 4096, depth 0)

| cache | avg tok/s | reps | cached blocks |
| --- | ---: | --- | ---: |
| 16 GiB | 3.37 | 2.97, 4.80, 2.33 | 37,890 |
| 22 GiB | 3.22 | 3.10, 2.66, 3.91 | 51,287 |

More VRAM expert cache (22.97 GiB actual, 74% of the expert payload) does **not**
help depth-0 decode — within noise of the 16 GiB point. At this shape the decode
step is bound by host/SSD staging and per-layer residency checks, not by the
amount of VRAM cache, which is consistent with the first pass and with the
depth result (the win comes from routing/warming, not raw cache size).

## MTP A/B (`iq2xs`, `--ctx` 4096, depth 0)

Same prompt, `--temp 0 --no-think`, `--max-new 32`, one process per rep.
MTP arm: `INFR_MTP=1 INFR_SPEC_DRAFT=<mtp-shared-Q4_K_M.gguf> INFR_PAGER_PROFILE=1`.
Accept rate from the `[qwen4 mtp summary]` line.

| arm | rep | decode tok/s | accept (alpha) | draft/verify/catch-up |
| --- | ---: | ---: | --- | --- |
| ordinary | 1 | 3.6 | — | — |
| ordinary | 2 | 3.5 | — | — |
| ordinary | 3 | 3.5 | — | — |
| MTP | 1 | **7.1** | 24/24, alpha=1.000 | 3% / 97% / 0% |
| MTP | 2 | **5.5** | 24/24, alpha=1.000 | 2% / 97% / 0% |
| MTP | 3 | **4.5** | 24/24, alpha=1.000 | 3% / 97% / 0% |

- MTP **works on this target**: the shared Q4_K_M head pairs with the GSQ-RCO
  IQ2_XS file even though that GGUF has no `nextn.*` tensors.
- On this greedy, easy prompt (`List the first ten prime numbers…`) the accept
  rate is **1.000** (every drafted token accepted) and decode is **1.3–2.0×**
  the ordinary arm.
- **Output identity holds**: all six runs return the same
  `2, 3, 5, 7, 11, 13, 17, 19, 23,` — speculative decode does not change the
  greedy output.
- The phase split is **97% verify**, i.e. the cost is the target forward. MTP is
  a multiplier on top of a slow target, not a way around the pager bottleneck.

The first pass's "MTP gives no gain (3.3 tok/s)" is contradicted here: run
properly as a paired A/B at `--max-new 32`, MTP gains 1.3–2×. The absolute
number is still far below bongo.

## Intel host-DMA hang — reproduced

With the default host-DMA path (dedicated transfer queue) and a cleared
pipeline cache, `paging.cache=6GiB -u 256 --ctx 4096`, the device is lost on the
first prefill:

```
INFO  infr_vulkan: [infr] host DMA: dedicated transfer queue family 2 enabled
INFO  infr_vulkan: [infr] host DMA import total: 20.96/20.96 GiB across 4/4 arena(s)
ERROR infr_vulkan: [infr] pipelined queue_submit could not be submitted after recovery
      (The logical device has been lost. See <…devsandqueues-lost-device>);
      refusing later GPU submissions because pager residency may describe copies that never executed
Error: backend: backend: queue_submit: The logical device has been lost.
```

Exit code `1`. Full stderr:
[`raw/hostdma-on-iq2xs-ctx4096-ub256-cache6GiB.log`](../../bench/results/2026-09-29-moe4all-b70/raw/hostdma-on-iq2xs-ctx4096-ub256-cache6GiB.log).
`INFR_NO_HOST_DMA=1` makes the identical command complete, so the host-DMA
upload path is the differentiator.

**Upstream issue:** a complete report is prepared at
[`upstream-issue-host-dma.md`](../../bench/results/2026-09-29-moe4all-b70/upstream-issue-host-dma.md).
It could **not** be filed: the agent's managed GitHub credential has only
`READ` permission on `Headmaster218/MoE4All`, and `gh issue create` returns
`GraphQL: Resource not accessible by integration (createIssue)`. See
"Upstream filing gap" below.

## Cold vs warm

The first matrix case (`iq2xs_ctx4096_cache16G_pp512_cold`) ran with a cleared
`~/.cache/infr/vk-pipeline-cache-*` **and** a warm OS page cache, because it
followed the host-DMA run and the download. Its preload ran at 0.63–0.70 GiB/s
(OS page cache) and pp512 measured **81.84 tok/s** (reps 100.88, 52.35, 92.29).
Every later case preloaded at 0.24–0.27 GiB/s (SSD) and measured the steady
**40.57 tok/s**. So:

- **Steady-state (page-cache-cold) prefill: ~40 t/s** — the honest number.
- **Page-cache-warm prefill: ~82 t/s** — the best case, when the model's expert
  blocks are in host RAM.

The pipeline cache rebuild costs little; the host page cache dominates. The
"cold/warm" axis that matters on this box is the OS page cache, not the shader
cache.

## Where INFR stands vs bongo

bongo baselines from `docs/final-overview.md` and the committed Stage 0 matrices.
The two tiers are compared against their own bongo tier.

### Decode at depth

| tier | depth | bongo | INFR | INFR/bongo |
| --- | ---: | ---: | ---: | ---: |
| iq2_xs | 4096 | 19.56 (M4 default) / 17.71 (Stage 0) | 6.25 | 0.32× / 0.35× |
| iq2_xs | 32768 | 11.67 | **12.54** | **1.07×** |
| q2_0 | 4096 | 13.24 | **13.68** | **1.03×** |
| q2_0 | 32768 | 10.36 | **15.77** | **1.52×** |

### Prefill

| tier | short (pp512) | INFR/bongo | long (pp32000) | INFR/bongo | bongo 32K prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| iq2_xs | 40.57 | 0.18× | 101.66 | 0.58× | 174.55 |
| q2_0 | 45.38 | 0.16× | 128.11 | 0.62× | 207.97 |
| iq3xs | 14.96 | — | — | — | — |

**Verdict: do not switch bongo to INFR as the shipped engine on the B70 today,
but the picture is much closer than the first pass said, and the `q2_0` tier is
competitive at long context.**

- **Decode is competitive to better at real depth on the `q2_0` tier, and INFR
  overtakes bongo at 32K decode on both tiers.** The rigorous matrix contradicts
  the first pass's "~5× slower decode": that was the cold-routing depth-0
  number. At 4K/32K depth INFR is at parity (`q2_0` 4K) or ahead (`q2_0` and
  `iq2xs` 32K).
- **The `iq2_xs` 4K gap is still real and large** (6.25 vs 19.56): llama.cpp's
  IQ2_XS kernel is much faster than INFR's at this quant, and INFR's pager is
  the bottleneck only after routing warms.
- **Prefill is the remaining blocker, but it is length-dependent.** At `pp512`
  INFR is ~5–6× slower (40–45 vs 230–288 tok/s); at a realistic 32K prompt it is
  ~1.6–1.7× slower (102/128 vs 175/208). For a long-prompt agentic turn INFR is
  in the same order as bongo on prefill and ahead on decode at the `q2_0` tier.
- **MTP narrows decode further but does not change the verdict**: a 1.3–2.0×
  gain on the depth-0 3.5 tok/s base is 7.1 tok/s at best, and its phase split is
  97% verify (the target forward), so it cannot fix the short-prompt gap.
- The one result worth carrying forward remains the **MTP sidecar pairing**
  (base Qwen3.8 head + GSQ-RCO target), now confirmed to work with a real gain.

If the product need were long-context, decode-heavy chat on `q2_0`, INFR would
be worth a closer look — it is genuinely competitive there. For bongo's shipped
default (`iq2_xs`, short-prompt agentic turns) the `iq2_xs` decode gap and the
`pp512` overhead keep llama.cpp the right engine.

## Upstream filing gap

The host-DMA hang is a genuine Intel/ANV defect and reproduces cleanly, but the
agent cannot file it upstream: the managed GitHub integration is installed only
for `03c` and has `READ` on `Headmaster218/MoE4All`. The report is committed
ready-to-file. Resolution needs one of:

- grant the GitHub integration write access to `Headmaster218/MoE4All`, or
- file the committed report from an account with `public_repo` scope.

This is escalated on [BAS-181](/BAS/issues/BAS-181); it is the only open item
from the original scope.

## Reproduction

```sh
# whole matrix (resumes; skips cases with a committed result)
./bench/run-moe4all-b70.sh --all --cold

# one cell
./bench/run-moe4all-b70.sh --case q2_0_ctx4096_cache16G_d4096_tg128

# regenerate the table
./bench/summarize-moe4all-b70.py
```

Every case's exact command is in
[`raw/commands.txt`](../../bench/results/2026-09-29-moe4all-b70/raw/commands.txt);
the driver is [`bench/run-moe4all-b70.sh`](../../bench/run-moe4all-b70.sh).

## Caveats

- Decode is noisy at 3 reps (up to 2× spread). Depth-0 decode is an
  order-of-magnitude signal only; the depth cells are tighter.
- `--ctx 32768` at depth 0 is the capacity ceiling, not a 32K-context decode.
  The depth table is the context axis.
- The MTP accept rate is measured on one greedy, easy prompt; alpha=1.000 is a
  best case, not an average. A harder prompt would lower it.
- All INFR numbers use the `INFR_NO_HOST_DMA=1` fallback (host DMA loses the
  device), so they are a lower bound on the engine's own path.
