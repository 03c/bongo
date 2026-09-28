# Dynamic VRAM expert LRU — held-out A/B and engine design (BAS-76)

Feeds [BAS-76](/BAS/issues/BAS-76) (M3.3 placement) and the Stage-2 engine plan in
[ADR-0003](../adr/0003-engine-direction.md). This resolves the one evidence conflict flagged in the
[engine gap analysis](engine-gap-analysis.md#4-the-one-evidence-conflict-to-resolve-locally) §4: R4's
offline frequency profile (held-out coverage 0.88–0.99) against R3's citation of llama.cpp PR #27861
(a static top-32 ranking recovers ~10% out-of-sample, an online LRU 67–81%).

- **Date:** 2026-09-28
- **Hardware:** Intel Arc Pro B70 (32 GiB VRAM), AMD Ryzen 7 9700X, 30 GiB RAM, Fedora 44
- **Engine:** llama.cpp `b11223` (`0.5.0-dev`, commit `4da6337767f973e2b4d0797e5b323d77d8565e4a`), Vulkan
- **Model:** Swift-1.5-Qwen3.8-Flash-Next IQ2_XS (`qwen4exp`), 48 layers × 512 experts, 10 active/layer,
  33.02 GiB of expert weights
- **Inputs:** R4 router captures
  ([`bench/results/2026-09-28-expert-activation/`](../../bench/results/2026-09-28-expert-activation/)) and the
  exact per-layer expert bytes
  ([`expert-bytes-iq2_xs.json`](../../bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json)).
- **Raw output:** [`bench/results/2026-09-28-byte-budget-placement/`](../../bench/results/2026-09-28-byte-budget-placement/)
  (`placement-iq2_xs-22.40.json`, `lru-ab.json`, `lru-ab.txt`).

## TL;DR

1. **Step 1 (config only): the byte-budget `-ot` placement is worth +4.25 pp** of activation coverage at the
   same 22.40 GiB expert budget — the R4 "+4–8 pp" estimate, reproduced from the exact placement. It needs no
   engine build. The same-session GPU A/B confirms it translates to throughput: **128K prefill +2.91%,
   128K decode +0.74%, 128K TTFT −2.83%**, no regression, at 0.22 GiB less peak VRAM (see
   **Status of the GPU runs**).
2. **Step 2 conflict resolved: R3's regime is not bongo's.** On bongo's model a *frozen* frequency profile
   generalises at **0.881–0.985** held out (R4 confirmed exactly) while a *cold online LRU* reaches
   **0.947–0.987** — not 67–81%. Neither R3 nor R4 is wrong; they are different model/workload regimes.
3. **The ship rule wins the local A/B on every held-out corpus: profile-as-LRU-initialisation** (`profile_lru`)
   scores **0.986–0.993**, beating both the frozen profile (0.881–0.985) and the cold LRU (0.947–0.987). Do
   **not** ship the static profile as the policy, and do not ship a cold LRU: initialise from the profile and
   let the LRU track drift.
4. Decode transfer agrees: on real per-token decode routing the profile-initialised LRU serves **0.985–0.986**
   versus **0.945–0.962** frozen and **0.862–0.903** cold.
5. The dynamic LRU is **engine work** (a llama.cpp `qwen4exp` MoE patch), because llama.cpp cannot hold
   per-expert residency. The design is below; the throughput A/B needs the patch and a free box.

## Step 1 — cheapest-layer-first byte-budget `-ot`

`--n-cpu-moe N` is a layer-*count* rule: keep layers `0..N-1` on CPU, the suffix resident. It is a poor byte
allocator — it reserves the near-empty layer 47 in prefill and keeps the byte-expensive layers resident, so
fewer layers fit than the budget allows (R4 §2). The replacement keeps the cheapest-expert-byte layers
resident until the budget is full and offloads the complement:

```
-ot blk\.(32|33|34|35|36|38|39|40|41|42|43|44|45|46|47)\.ffn_(gate|up|down)_exps\.weight=CPU
```

at the measured-safe expert budget of **22.40 GiB** (`n=16`'s resident expert bytes; `n=12` device-loses at
31.85 GiB total):

| quantity | byte-budget `-ot` | `--n-cpu-moe 16` (Stage 0) |
| --- | ---: | ---: |
| resident expert bytes | 22.217 GiB | 22.40 GiB |
| resident layers | 33 | 32 |
| CPU layers | 15 | 16 |
| **activation coverage** | **0.7021** | 0.6596 |

**+4.25 pp** at 2.7% *less* resident expert bytes. The gain comes from fitting one more layer (the cheap
layers pack better) and from dropping layer 47, which contributes ~0.001% of prefill events. Coverage is the
whole-layer event fraction from R4's per-layer counts, so it is exact for whole-layer residency.

The tool is [`bench/gen-ot-placement.py`](../../bench/gen-ot-placement.py); the re-run harness is
[`bench/sweep-byte-budget-placement.sh`](../../bench/sweep-byte-budget-placement.sh), which launches
llama-server directly (no `bongo.sh` dependency) and refuses to start if port 8080 is already serving, so it
cannot contaminate another measurement. The Stage 0 `--n-cpu-moe 16` baseline stays pinned and selectable via
`--n-cpu-moe 16`.

### Measured throughput — same-session A/B

`bench/sweep-byte-budget-placement.sh` ran both configs back to back on the same box, engine
`b11223-4da633776`, Vulkan `Vulkan1`, tier `iq2_xs`, ctx 131072, agentic `cache_prompt=true` profile. The dump
to-CPU share falls only from 33.3% to 31.3% of layers (32 → 33 resident), so the first-order gain is
single-digit percent, smaller than R4's +18–37% estimate — that estimate assumed a *per-expert* residency
that cuts the CPU share to 2–12%, which only the Step 2 engine can express.

| 128K metric | byte-budget `-ot` | `--n-cpu-moe 16` | Δ |
| --- | ---: | ---: | ---: |
| prompt tok/s | 135.224 | 131.395 | **+2.91%** |
| output tok/s | 8.010 | 7.951 | **+0.74%** |
| TTFT ms | 937660.6 | 964978.7 | **−2.83%** |
| peak VRAM GiB | 29.29 | 29.51 | −0.22 GiB |
| needle | pass | pass | — |

The acceptance criterion for Step 1 is "+4–8 pp coverage at the same budget with no 128K decode regression";
coverage is met (+4.25 pp) and the 128K decode regression check passes (+0.74%). Raw files and the exact
argv are in [`bench/results/2026-09-28-byte-budget-placement/`](../../bench/results/2026-09-28-byte-budget-placement/);
the A/B is computed by [`bench/compare-placement-ab.py`](../../bench/compare-placement-ab.py).

## Step 2 — held-out A/B: frozen profile vs dynamic LRU

Every policy is evaluated at the **same 22.40 GiB budget** on the leave-one-corpus-out held-out trace
(`bench/sim-expert-lru.py`). Coverage = fraction of routed `(layer, expert)` events served from the resident
set. `profile_lru` preloads the profile built on the *other* corpora and then runs a standard LRU.

### Prefill, held out (train on the other three corpora, score the fourth)

| policy | doc | code | chat | convo |
| --- | ---: | ---: | ---: | ---: |
| `static_insample` (ceiling) | 0.9957 | 0.9897 | 1.0000 | 1.0000 |
| **`profile_lru` (ship rule)** | **0.9930** | **0.9863** | **0.9884** | **0.9916** |
| `cold_lru` (R3's online LRU) | 0.9873 | 0.9800 | 0.9467 | 0.9688 |
| `per_layer_lru` | 0.9831 | 0.9759 | 0.9465 | 0.9684 |
| `static_profile` (R4's frozen policy) | 0.9632 | 0.8807 | 0.9853 | 0.9771 |
| Step-1 `byte_budget_layers` | 0.7021 | 0.7021 | 0.7021 | 0.7021 |
| `static_layer_16` (Stage 0) | 0.6596 | 0.6596 | 0.6596 | 0.6596 |

The `static_profile` row reproduces R4's leave-one-corpus-out table to three decimals (doc 0.963, code 0.881,
chat 0.985, convo 0.977).

### Decode, held out (prefill profile from the other corpora, real decode trace)

| policy | doc | chat |
| --- | ---: | ---: |
| `profile_lru` | 0.9850 | 0.9862 |
| `static_profile` | 0.9451 | 0.9617 |
| `cold_lru` | 0.9031 | 0.8616 |

### Interpretation

- **R4 reproduces; R3 does not.** The frozen profile generalises at 0.881–0.985 on bongo. R3's "static
  ranking recovers ~10% out-of-sample, LRU 67–81%" was measured on another model and does not describe this
  workload. The conflict is a model/workload regime difference, not a contradiction.
- **The cold online LRU is strong (0.947–0.987), but loses to profile initialisation on every corpus.** The
  536-token `chat` trace is the worst case (0.947 vs 0.988): a cold LRU needs traffic to warm up, and a short
  session never does. Initialising from the profile removes the warmup penalty and keeps the LRU's ability to
  track drift.
- **The profile is not a fad and not a substitute for the LRU.** It is the best *prior*; the online policy is
  what protects against a serving-domain shift (the code corpus is where the frozen profile is weakest, 0.881,
  and where profile-LRU gains the most, +10.6 pp).
- **The per-layer LRU is not better than the global LRU here.** Equal byte share per layer wastes budget on
  cold layers; the global LRU is fine.

## Engine design — dynamic VRAM LRU over RAM-pinned experts

llama.cpp's `qwen4exp` MoE path keeps a whole `ffn_{gate,up,down}_exps` tensor in one buffer (`--n-cpu-moe`
and `-ot` choose the buffer per layer). A dynamic LRU needs **per-expert residency**, which the current
tensor layout cannot express. The design:

1. **Split the expert tensor by expert.** For each layer, keep the dense/shared tensors on the GPU and pin the
   `ffn_*_exps` weights in host memory (mmap, page-cache-resident; R7 rules out more RAM as a lever, so the
   host copy is already there). A `qwen4exp` expert `e` of layer `l` is a fixed-stride slice; expose it as a
   sub-buffer so a residency decision is per `(l, e)`, not per tensor.
2. **VRAM cache.** A fixed byte budget holds resident expert slices plus their quantization metadata.
   Initialise the cache from the offline profile (the counter file is already produced by the R4 capture
   path); then run a **global LRU** keyed by `(l, e)`, with an optional admission filter (do not admit a cell
   the LRU is about to evict again — the simple "2Q"/frequency-admission guard that Strata uses).
3. **Miss path.** On a miss, dequantise/copy the slice from the host pin to a free VRAM slot (or evict the LRU
   tail), then run the expert. Because the model already dequantises in prefill on both backends (R6), the
   copy is not a new op class; it replaces a CPU-side expert matmul with a GPU-side one. Batch the miss fetch
   across the 10 active experts of the layer so a token pays one transfer, not ten.
4. **Profile as initialisation only.** Ship the cache start state from the profile; never freeze it. Expose
   the policy as flags in the spirit of `--n-cpu-moe`, plus a counters dump, so the held-out A/B is
   repeatable. Keep `--n-cpu-moe 16` and the Step-1 `-ot` rule selectable as fallbacks.
5. **Numerics.** The resident slice is byte-identical to the host slice (same quantised weights); the cache
   only changes *where* the matmul runs, so output is unchanged. Verify with a KLD/needle check on the first
   build.

This is the same shape Strata ships (R1), expressed for llama.cpp's `qwen4exp` MoE rather than CUDA.

## Status of the GPU runs

The Step-1 Stage-1 A/B is **measured** (same-session byte-budget `-ot` vs pinned `--n-cpu-moe 16`, see the
table above): +2.91% 128K prefill, +0.74% 128K decode, −2.83% 128K TTFT, no regression, both needles pass. The
byte-budget run also measured the agentic prefix-cache path (cold 4K 172.5, 16K 202.9, 24K 190.4 prompt tok/s;
+512-token grow turns at 3.58 s / 4.56 s TTFT; slot restore verified). The engine throughput A/B for Step 2
still needs the patched engine. Reproduce:

```sh
# Step 1: byte-budget -ot at 4K + 128K, VRAM + throughput + needle, plus the prefix-cache path
bench/sweep-byte-budget-placement.sh
# the pinned Stage 0 baseline, for the same protocol
bench/sweep-byte-budget-placement.sh --n-cpu-moe 16

# Step 2: the held-out A/B (no GPU)
python3 bench/sim-expert-lru.py \
  --raw bench/results/2026-09-28-expert-activation/raw \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --budget-gib 22.40 --corpora doc,code,chat,convo \
  --decode doc=bench/results/2026-09-28-expert-activation/raw/doc_dec.tsv.gz,chat=bench/results/2026-09-28-expert-activation/raw/chat_dec.tsv.gz \
  --out bench/results/2026-09-28-byte-budget-placement/lru-ab.json
```

## Limitations

- **Coverage vs tokens/s.** The offline A/B measures expert-activation coverage; the Step-1 throughput
  translation is now measured (see above) at +0.74–2.91% for whole-layer residency. The Step-2 engine
  throughput is still pending the patched engine.
- **Simulated LRU.** The LRU is replayed over captured routing, not over a patched engine. It ignores copy
  latency, cache-line effects and batch behaviour; it bounds the residency policy, not the engine's realised
  speed.
- **Four corpora, 9,309 prefill tokens / 384 decode tokens.** Same sample as R4; the decode check is real but
  short.
- **IQ2_XS only.** Per-layer expert bytes, and therefore the byte-budget curves, differ for Q2_0 / IQ3_XXS.
- **Global LRU only.** Belady/optimal weighted caching was not computed; the in-sample static profile is used
  as the coverage ceiling instead.
