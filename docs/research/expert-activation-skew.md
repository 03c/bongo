# Expert-activation skew on the Arc Pro B70 — does a hot set beat the layer rule?

Measurement-only spike for [BAS-66](/BAS/issues/BAS-66), feeding the Stage-2 re-plan in
[BAS-62](/BAS/issues/BAS-62) after the Stage-1 `--n-cpu-moe` no-go
([`expert-placement.md`](expert-placement.md)). This is the measurement the Stage-1 note asked for under
"Measure expert activation skew": instrument llama.cpp's MoE router over a representative prompt set and
compare a frequency-ordered resident set against the layer-ordered one at the same byte budget.

- **Date:** 2026-09-28
- **Hardware:** Intel Arc Pro B70 (32 GiB VRAM), AMD Ryzen 7 9700X, 30 GiB system RAM, Fedora 44
- **Engine:** llama.cpp `b11223` (`0.5.0-dev`, commit `4da6337767f973e2b4d0797e5b323d77d8565e4a`), Vulkan build.
  The capture itself ran on the CPU (`n_gpu_layers=0`); router selections are a property of the weights, not of
  the backend, and CPU-only sidesteps the 33 GiB-expert / 32 GiB-VRAM ceiling.
- **Model:** `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`qwen4exp`), tier **IQ2_XS**,
  48 layers x 512 experts, 10 active per token per layer, 33.02 GiB of expert weights.
- **Raw data:** [`bench/results/2026-09-28-expert-activation/`](../../bench/results/2026-09-28-expert-activation/)
  (`raw/*.tsv.gz`, `corpora/*.txt`, `analysis.json`, `coverage.txt`)
- **Tooling:** [`bench/tools/route_capture.c`](../../bench/tools/route_capture.c),
  [`bench/run-expert-activation.sh`](../../bench/run-expert-activation.sh),
  [`bench/analyze-expert-activation.py`](../../bench/analyze-expert-activation.py)

## TL;DR

**Yes — a frequency-ranked hot set beats the `--n-cpu-moe` layer rule by a large margin, and the profile is
stable enough to build offline. But it buys prefill and short-context decode, not 128K decode.**

| expert budget | hot set (frequency) | `--n-cpu-moe` layer rule | best layer-granular rule | arrival-order fill | random set |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 GiB | **0.693** | 0.213 | 0.277 | 0.265 | 0.241 |
| 16 GiB | **0.923** | 0.447 | 0.511 | 0.527 | 0.486 |
| **22.40 GiB** (= `--n-cpu-moe 16`) | **0.985** | **0.660** | 0.702 | 0.732 | 0.679 |
| 25.00 GiB (= `--n-cpu-moe 13`) | **0.994** | 0.723 | 0.766 | 0.817 | 0.756 |

Coverage = fraction of routed `(layer, expert)` events served from the resident set. Pooled over 9,309
prefill tokens (doc, code, chat, conversation). The static column is the actual llama.cpp rule at that byte
budget; "best layer-granular" is whole-layer residency chosen cheapest-layer-first, i.e. an upper bound on
*any* layer-count rule.

- At the shipped budget the gap is **+32.5 percentage points** (0.985 vs 0.660).
- Out of sample (profile built on every corpus except the one being scored): **0.881 (code), 0.963 (doc),
  0.977 (conversation), 0.985 (chat)** at 22.40 GiB. Even the worst domain still beats the layer rule by
  +22 pp.
- A **teacher-forced decode** check on 256 real continuation tokens: the prefill-built profile covers
  **0.972** of decode events (Spearman 0.71 vs prefill), against 0.667 for the layer rule. Prefill routing
  transfers to decode.
- **Strata comparison:** its profile reported `h = 0.6447` against `0.4864` arrival-order (+15.8 pp). On
  bongo's model and workload the same kind of profile is worth roughly **twice** that gap.

### By how much does it matter for speed?

Using the measured Stage-1 placement marginals and the out-of-sample miss rate (1.5-11.9% depending on how
well the profile's corpus matches the serving domain), cutting the CPU expert share from the static 33.3%
(16/48 layers) to the profile's miss rate is worth, as a first-order linear estimate:

| workload | CPU share 33.3% (n=16) | profile (2-12% CPU) | gain |
| --- | ---: | ---: | ---: |
| 4K prefill (prompt tok/s) | 231.5 | ~289-316 | **+25% to +37%** |
| 128K prefill (prompt tok/s) | 133.4 | ~158-169 | **+18% to +27%** |
| 4K decode (output tok/s) | 16.1 | ~22-25 | **+38% to +57%** |
| 128K decode (output tok/s) | 7.62 | ~7.6-7.8 | **~0%** |

128K decode is flat across the whole feasible CPU-share range (33% -> 50% CPU costs 1.2%, inside noise) and
only collapses near 100% CPU, so a residency change has almost nothing to buy there. The picture is the same
as the Stage-1 finding: at long context the bottleneck is attention/KV, not expert placement.

**The catch:** llama.cpp cannot place individual experts. `--n-cpu-moe` is layer-count and `-ot` is
tensor-pattern granular, while a layer's experts are one `ffn_{gate,up,down}_exps` tensor. Expressing this
profile requires per-expert tensor splitting (a smaller, per-expert GGUF layout) or a bongo-owned engine. The
measurement establishes the upside; the build is still engine work, not a flag.

## Method

### Capturing the router

The reference box has no compiler, no llama.cpp source, and no sudo. The capture therefore reuses the
**existing** bongo libraries rather than building llama.cpp:

1. `llama-cli --version` reports `build 11223, commit 4da633776`. The `llama.h` / `ggml.h` headers are fetched
   at that exact commit, so the by-value `llama_context_params` ABI matches the shipped `.so`.
2. A small C tool ([`route_capture.c`](../../bench/tools/route_capture.c)) is compiled with Zig
   (`pip install --user ziglang`, user-local, no system change) and linked against the bongo Vulkan build's
   `libllama.so` / `libggml*.so`.
3. It installs `llama_context_params.cb_eval` — the same `ggml_backend_sched_eval_callback` hook that
   `llama-eval-callback` uses — and copies the `ffn_moe_topk-<layer>` tensor (I32, shape
   `[10, n_tokens]`) after each layer's graph node runs.

`ffn_moe_topk` is a strided view of the layer's argsort output, so `ggml_nbytes()` over-counts; the tool walks
the real `nb[]` strides and emits exactly `ne[0]*ne[1]*ne[2]*ne[3]` indices per layer.

The tool runs two phases:

- **prefill** — one `llama_decode` over the whole prompt (`n_ubatch = n_ctx`), so a single graph yields the
  routing for every token at every layer at once;
- **decode** (optional) — teacher-forced single-token decodes over the tail of the corpus, to capture genuine
  per-token decode routing (and the final layer, see below).

TSV format, one line per layer per graph:

```
TOPK  ffn_moe_topk-<il>  ne0  ne1  ne2  ne3  idx0 idx1 ...
```

`ne0 = 10` (active experts), `ne1 = tokens`; flat order is token-major (`element(k, token) = k + token*10`).
Decode files also carry `STEP <i>` markers before each decode step.

### Corpora

Four prompt classes, copied verbatim into
[`bench/results/2026-09-28-expert-activation/corpora/`](../../bench/results/2026-09-28-expert-activation/corpora/):

| corpus | content | prefill tokens | decode tokens |
| --- | --- | ---: | ---: |
| `code` | this repo's `bongo.sh` + `tools/gguf-inventory.py` | 3,794 | — |
| `doc` | this repo's `expert-placement.md` + `intel-arc-b70.md` | 3,971 | 256 |
| `chat` | a synthetic 10-turn chat about MoE placement | 536 | 128 |
| `convo` | the chat plus 10 more turns, for a growing conversation | 1,008 | — |

For the two corpora with a decode tail the prefill uses `tokens[0 .. n-k)` and decode teacher-forces the last
`k` tokens, so the profile-training and decode-reporting token sets are disjoint.

### The measurement artifact on layer 47

`ffn_moe_topk-47` reports `ne1 = 1` in every prefill: llama.cpp only needs the final position's output from
the last layer, so prefill evaluates layer 47's MoE for one token. Layer 47 is therefore nearly absent from
the prefill counts (40 events of 4.6 M). This is why the prefill static-layer coverage is `31/47 = 0.6596`
rather than `32/48`: the layer rule reserves layer 47's 773 MB of experts but collects almost no hits from it.
Decode runs all 48 layers per token, and the decode analysis recovers layer 47 (its top-10 share is 0.26-0.30,
in line with the 0.23-0.25 mean of the other layers).

### Interpreting coverage

For a resident set `R` of `(layer, expert)` pairs and an event multiset `E` of routed selections,
`coverage(R, E) = |{e in E : e in R}| / |E|`. `R` is built under a hard byte budget using the exact per-layer
expert bytes from [`expert-bytes-iq2_xs.json`](../../bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json)
(per-expert bytes vary by layer: 1.18 / 1.31 / 1.51 MB). Ranked sets are filled greedily by count (and,
identically within noise, by count-per-byte). The static rule is `--n-cpu-moe N`: experts of layers `0..N-1`
on the CPU, layers `N..47` resident.

## Results

### 1. The routing distribution is concentrated, but not trivially so

Pooled over all 9,309 prefill tokens and 4.47 M `(layer, expert)` selections:

- **23,014 of 24,576** `(layer, expert)` cells are touched at least once (93.6%).
- Entropy **9.38 bits**, i.e. **~11,900 effective experts** of 24,576.
- The top-N cells needed for a given event share: **3,233** for 50%, **8,112** for 80%, **11,025** for 90%,
  **13,428** for 95%, **17,619** for 99%.
- Per layer, the top-10 experts take a median **13.5%** of that layer's events (min 5.9%, max 37.5%), against
  the 2.0% a uniform router would give. So each layer has a real hot core, but the tail is long.
- Median distinct experts seen per layer: **492 of 512**.

This is why the wins below are large but not unbounded: the hot core is what a resident set can capture, and
the long tail is what it cannot.

### 2. Coverage curves: hot set versus the layer rule

Pooled prefill events, all budgets in GiB. `hot` = ranked by count; `static` = `--n-cpu-moe N`;
`layer greedy` = whole layers, cheapest first; `arrival` = first-touch order; `random` = size-matched random
cells. Full table in [`coverage.txt`](../../bench/results/2026-09-28-expert-activation/coverage.txt).

| GiB | hot cells | **hot** | static | n_cpu | layer greedy | arrival | random |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2,975 | **0.477** | 0.085 | 43 | 0.128 | 0.128 | 0.121 |
| 8 | 5,977 | **0.693** | 0.213 | 37 | 0.277 | 0.265 | 0.241 |
| 12 | 8,972 | **0.835** | 0.319 | 32 | 0.383 | 0.400 | 0.364 |
| 16 | 11,966 | **0.923** | 0.447 | 26 | 0.511 | 0.527 | 0.486 |
| 20 | 14,950 | **0.970** | 0.575 | 20 | 0.617 | 0.657 | 0.608 |
| **22.40** | 16,735 | **0.985** | **0.660** | **16** | 0.702 | 0.732 | 0.679 |
| 25.00 | 18,668 | **0.994** | 0.723 | 13 | 0.766 | 0.817 | 0.756 |
| 28 | 20,884 | **0.999** | 0.830 | 8 | 0.872 | 0.914 | 0.844 |
| 33.02 | 23,014 | 1.000 | 0.979 | 1 | 1.000 | 1.000 | 1.000 |

Three observations:

- **The hot set is worth +22 to +48 pp over the layer rule at every budget that fits.** The gap is widest in
  the middle (8-16 GiB) and narrows only when the budget approaches the whole expert set.
- **The layer rule is worse than a size-matched random set at small budgets** (0.085 vs 0.121 at 4 GiB). Two
  reasons: it reserves the near-empty layer 47, and the GPU-resident suffix happens to contain the
  byte-expensive layers, so fewer layers fit than the byte count suggests. `--n-cpu-moe` is a poor byte
  allocator even within its own granularity; a cheapest-layer-first rule recovers +4-8 pp for free.
- **The hot set is ~3x more byte-efficient.** The static 22.40 GiB rule's 0.660 coverage is matched by a hot
  set of only **~7 GiB** (interpolating 0.477 at 4 GiB and 0.693 at 8 GiB). Returning 15 GiB of VRAM to the
  KV/context budget while *beating* the layer rule is the practical version of this result.

### 3. Temporal stability: the profile transfers

Two ways to stress a static profile. First, train on one corpus and score another (22.40 GiB budget):

| train \ test | doc | code | chat | convo |
| --- | ---: | ---: | ---: | ---: |
| doc | 0.996 | 0.890 | 0.964 | 0.961 |
| code | 0.922 | 0.990 | 0.904 | 0.884 |
| chat | 0.872 | 0.697 | 1.000 | 0.993 |
| convo | 0.898 | 0.734 | 1.000 | 1.000 |

Code is the outlier domain, and the 536-token `chat` corpus is too small to single-handedly profile it
(0.697). Second, the honest held-out test — build on **all other** corpora, score the held-out one:

| held-out corpus | 12 GiB | 16 GiB | 22.40 GiB | 25 GiB | static @ 22.40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| doc | 0.731 | 0.860 | **0.963** | 0.981 | 0.660 |
| code | 0.579 | 0.720 | **0.881** | 0.921 | 0.660 |
| chat | 0.812 | 0.915 | **0.985** | 0.995 | 0.660 |
| convo | 0.764 | 0.887 | **0.977** | 0.990 | 0.660 |

A profile built on ~8,800 tokens of mixed traffic predicts a held-out domain at 0.88-0.99. The profile is
stable enough to build once, offline. The one operational rule: **the profiling mix must include the serving
domain** — a profile with no code in it loses ~9 pp on code, and at a 12 GiB budget that can matter (0.579).

### 4. Prefill routing predicts decode routing

The 256-token teacher-forced decode tail of `doc` (and 128 tokens of `chat`):

| quantity | doc | chat |
| --- | ---: | ---: |
| Spearman(prefill, decode) per layer | 0.708 | 0.713 |
| top-10 per-layer overlap(prefill, decode) | 0.319 | 0.431 |
| prefill profile @ 22.40 GiB, coverage of decode | **0.972** | **0.968** |
| layer rule @ 22.40 GiB, coverage of decode | 0.667 | 0.667 |
| decode-adjacent persistence | 0.309 | 0.329 |
| static per-layer top-10 predictor | 0.232 | 0.252 |

A profile built on prefill serves 97% of real decode selections, so the residency decision does not need a
separate decode profile. The rank correlation is moderate (0.71), but the top-of-distribution agreement — what
residency actually needs — is high.

### 5. Predictability: temporal structure exists but is not needed

Adjacent-token set overlap (prefill averages; decode values above):

| corpus | adjacent persistence | static per-layer top-10 predictor | random |
| --- | ---: | ---: | ---: |
| doc | 0.348 | 0.148 | 0.020 |
| code | 0.342 | 0.145 | 0.020 |
| chat | 0.339 | 0.151 | 0.020 |
| convo | 0.335 | 0.133 | 0.020 |
| pooled | 0.344 | 0.145 | 0.020 |

An expert set persists across adjacent tokens far above chance (0.34 vs 0.02), and a naive "same experts as
the previous token" predictor (0.34) beats a static per-layer top-10 predictor (0.145) by 2.4x. So the next
token's experts **are** partly predictable from the current token.

But it does not change the decision. A profile that predicts the *whole* hot set covers 0.88-0.99 out of
sample; the best a persistence predictor could do on top of it is cover the remaining 1-12% of misses, and a
persistence predictor only reaches 0.34 on its own. **A learned predictor is not worth it for residency.** It
could matter only if the VRAM budget were forced down to ~4-8 GiB, where static coverage is 0.48-0.69 and the
miss tail is large.

### 6. The bongo-specific budget

From the Stage-1 sweep plus the exact expert-byte inventory:

| quantity | value |
| --- | ---: |
| usable VRAM (32 GiB physical minus stolen) | 31.92 GiB |
| dense + KV + state + buffers @ 128K (measured: 29.26 - 22.40; 23.68 - 16.83) | 6.86 GiB |
| dense + KV + state + buffers @ 4K | 6.42 GiB |
| arithmetic expert budget @ 128K | 25.06 GiB |
| arithmetic expert budget @ 4K | 25.50 GiB |
| measured-safe expert budget @ 128K (`n=16` serves, `n=12` device-loses at 31.85 GiB) | **22.40 GiB** |
| total expert set | 33.02 GiB |

At 22.40 GiB the hot set serves **98.5%** of prefill events (88-99% held out) versus **66.0%** for the layer
rule. At the 4K arithmetic budget (25.5 GiB) it is 99.4% versus 72.3%.

## Verdict

1. **Does a hot-expert residency beat the layer rule on this box?** Yes, decisively: **+22 to +48 pp** of
   expert-activation coverage at every byte budget that fits, +32.5 pp at the shipped `--n-cpu-moe 16` budget.
   The gap is roughly twice Strata's measured profile-vs-arrival-order gap.
2. **Is a profile stable enough to build offline?** Yes. Held-out coverage is 0.88-0.99; a prefill-built
   profile serves 97% of real decode selections. Include the serving domain in the profiling mix.
3. **Is a learned predictor worth it?** No, not for residency. Adjacent-token persistence is real (0.34 vs
   0.02 chance) but the static profile already covers 0.88-0.99; the predictor's upside is the small miss
   tail. Revisit only for a deliberately tiny resident set.
4. **Is it enough to matter for decode vs prefill?** **Prefill: yes** (+18-27% at 128K, +25-37% at 4K, first-
   order). **4K decode: yes** (+38-57%). **128K decode: no (~0%)** — that path is attention/KV-bound, exactly
   as Stage 1 found. If the product target is 128K decode tok/s, this is not the lever; if it is TTFT/prefill
   and short-context turns, it is one of the largest single levers available.
5. **Can llama.cpp express it?** No. This needs per-expert tensor layout (per-expert GGUF) or a bongo-owned
   engine. It is Stage-2 engine work, not a config change. A cheaper interim step is available: replace
   `--n-cpu-moe N` with a cheapest-layer-first byte-budget placement (`-ot`), which recovers +4-8 pp at the
   same budget with no new engine.

### Recommended next step

Two options, both small relative to a full engine:

- **Cheap, immediate:** an `-ot` byte-budget placement rule (cheapest layers first) improves the static rule
  by +4-8 pp at no build cost. Re-run the Stage-1 sweep to confirm the throughput effect.
- **Stage 2 gate:** if the profile-level gain is worth the build, prototype a per-expert tensor layout on one
  or two tiers and measure the profile's realised prefill gain against the +18-27% first-order estimate. The
  measured marginals in [`expert-placement.md`](expert-placement.md) are the yardstick.

## Limitations

- **Prefill-heavy sample.** 9,309 prefill tokens plus 384 decode tokens. The decode check is strong enough to
  validate transfer, but a production-length decode study would need more.
- **CPU capture.** Routing was captured with `n_gpu_layers=0`. Router selections depend on weights, not on
  who executes the matmul, so this is safe — but it is one more reason to re-verify on the deployed path.
- **Linear throughput extrapolation.** The +18-37% estimates extend the measured CPU-share relationship below
  its measured floor of 33% CPU. The true curve flattens near 0% CPU, so treat them as upper bounds. The
  direction and the 128K-decode ~0 result are measured, not extrapolated.
- **Static profile only.** No online eviction/admission was measured; the finding is about one-time
  frequency-ranked residency, which is what Strata ships.
- **Layer 47 in prefill** is a 1-token sample (see Method); decode recovers it, and it is 1/48 of the work.
- **Single tier.** IQ2_XS only. Expert bytes per layer, and therefore the byte-budget curves, differ for
  Q2_0 / IQ3_XXS.

## Reproduce

```sh
# 1. Build the capture tool.  Needs the bongo llama.cpp build and (for the
#    headers) network access; installs zig user-locally via pip.
MODEL=~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
bench/run-expert-activation.sh --smoke --model "$MODEL"        # build + 15-token check
bench/run-expert-activation.sh --model "$MODEL" --decode 256    # the four corpora (~25 min)

# 2. Analyse.  Reads the raw captures in the results dir and writes analysis.json.
python3 bench/analyze-expert-activation.py \
  --raw bench/results/2026-09-28-expert-activation/raw \
  --decode doc=bench/results/2026-09-28-expert-activation/raw/doc_dec.tsv.gz,chat=bench/results/2026-09-28-expert-activation/raw/chat_dec.tsv.gz \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --out bench/results/2026-09-28-expert-activation/analysis.json \
  --corpora doc,code,chat,convo
```

`bench/run-expert-activation.sh` fetches `llama.h` / `ggml*.h` at commit `4da6337767…`, compiles
`bench/tools/route_capture.c` with zig, links the existing `libllama.so`, and runs each corpus. Each capture
is dominated by paging in the 68 GB mmapped model (2-8 minutes per corpus on the reference box).
