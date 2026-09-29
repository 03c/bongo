# Strata engine architecture — R1 deep-dive

Source-cited architecture description of the [Strata](https://github.com/Niko1221/Strata) engine, written to be
ported from. Companion to the CTO's first-pass recon ([`strata-ninfer-recon.md`](strata-ninfer-recon.md)); this
document supersedes that recon's §2.2 on two points (see §2.5) and answers the six questions in
[BAS-63](/BAS/issues/BAS-63).

- Date: 2026-09-28
- Source revision read: `Niko1221/Strata` @ **`b89c989a7155e984e544ddd90d1038dda7da9e3d`** ("prompt path: never
  write into the borrowable cache slots while laying out"), `CMakeLists.txt` project version **0.1.13**, tag
  `v0.1.13`. Read-only clone, shallow history (50 commits), so statements about the *absence* of a file are
  limited to this revision (flagged where relevant).
- Method: read the upstream tree (`README.md`, `docs/DETAILS.md`, `include/**`, `src/**`, `bench/results/**`,
  `setup.py`, `CMakeLists.txt`). No weights or multi-GB artifacts downloaded. No Strata code vendored. One small
  binary artifact committed in the repo (`data/expert-profile.bin`, 130 kB) was parsed for its format; the raw
  parse is recorded in
  [`bench/results/2026-09-28-strata-profile-header/header.txt`](../../bench/results/2026-09-28-strata-profile-header/header.txt).
- bongo-side baseline (from the recon): Arc Pro B70 + llama.cpp Vulkan ≈ **8 tok/s** at 128K. Strata:
  **52 tok/s** (IQ2_XS) / **65 tok/s** (Q2_0) at 128K on an RTX 5070 12 GB with 64 GB RAM.

## TL;DR — what actually produces Strata's speed

The tiering story in the README ("GPU does the every-token weights, RAM holds all experts and the CPU computes
the misses concurrently, SSD holds the lookup table") is the **enabler**, not the multiplier. The measured
multipliers, in the order the upstream benches attribute them:

| # | Mechanism | Where it lives | Measured contribution |
|---|---|---|---|
| 1 | **Per-layer CUDA graph replay** instead of ~2,000 launches/token; two graphs per layer split at the router | `src/core/session.cpp:159`, `src/core/layer.cpp:1091` and `:1240` | 43-node block graph **1.585 ms vs 2.393 ms** for the same work as direct launches = **1.51x** (`session.hpp`) |
| 2 | **Quantized-weight expert matmul** — weights are never dequantized | `src/kernels/cuda/*`, `src/prefill/moe_mmq.cu` | prefill 1052 → **1130** prompt tok/s (`bench/results/2026-09-28-prefill-speed/README.md`) |
| 3 | **A profile-filled VRAM expert tier**, per-layer split, GPU hits overlapped with CPU misses | `include/strata/core/expert_cache.hpp`, `src/core/expert_source.cpp:440` | CPU drain **19.076 → 10.312 ms/token**, **−2.7 ms/token** end-to-end at 4096 slots (`src/program/generate.cpp:384`) |
| 4 | **PCIe expert ring overlapped with the attention half** (prefill only) | `src/prefill/prefill.cpp:66-81`, `:802-850` | 1130 → **1290** prompt tok/s (Q2_0, 32K) |
| 5 | **Speculation** — MTP draft layer + suffix lookup, with an exact-equivalence verify window | `include/strata/core/verify.hpp`, `include/strata/spec/*` | 1.6–1.8x advertised; **2.4–3.2** tokens committed per pass (`docs/DETAILS.md`) |

Two structural facts drive everything else:

- **The residual chain is strictly serial, and the engine says so.** Verbatim
  (`include/strata/core/session.hpp`, note on `SessionGraphs::posts`): *"A per-layer CPU pool cannot be hidden
  behind a strictly serial residual chain, which is why the CPU term is answered by Phase 3's VRAM cache and not
  by this pipeline."* The measured consequence (`include/strata/core/expert_cache.hpp`): the CPU expert pool is
  *"663.6 MB of expert bytes per token at ~40 GB/s = 16.2 ms, on a token of ~53 ms"*, and *"`test_expert_pool`
  says the pipeline hides **1.055 ms of 19.035** — the CPU work is 96% exposed."* Overlap here is a per-layer
  window, not a pipeline.
- **The cost that is not exposed is the one they removed, not the one they hid.** The fix for the exposed CPU
  term was the VRAM tier (fewer bytes read on the CPU), not a cleverer schedule. Schedule work buys ~1.5x
  (launch overhead); quantized-weight kernels plus residency buy the rest.

**Verdict for bongo.** The 6–8x gap is dominated by items 1–3, and all three are portable in principle (§6):
graph-level submission, an MMQ-style kernel that keeps weights quantized, and a computed VRAM expert tier with a
per-layer admission split. Items 4 and 5 are the next tier. The *implementations* are heavily CUDA-specific
(45 `__shfl_xor_sync`, 30 `__shfl_down_sync`, 33 `__dp4a`, `mma.sync`, `__threadfence_system`,
`cudaHostAlloc(Mapped)`), which is what §6 is about.

---

## 1. Dispatch and schedule

### 1.1 The unit of work: one block, captured as two graphs split at the router

`session_capture` (`src/core/session.cpp:159-242`) captures **two `cudaGraphExec_t` per layer**:

| Graph | Contents, in order | Built by |
|---|---|---|
| `pre[l]` | `gr_read` (attn half) → GDN or QSA mixer → `gr_write` → `gr_read` (ffn half) → **router** → doorbell ring → shared expert | `block_layer_pre`, `src/core/layer.cpp:1091` |
| `post[l]` | `moe_combine` / `moe_finish` → `gr_write` | `block_layer_post`, `src/core/layer.cpp:1240` |

`gr_read` / `gr_write` are the **gated residual ("hyper-connection")** pair: `R` is a stack of `hc = 4`
residual streams, each `n_embd = 2560` wide — not one vector with a gate (`include/strata/kernels/gr.hpp`). Every
block reads the stack, mixes it, and writes back per stream, so *both* halves begin with a `gr_read` and end with
a `gr_write`, and the residual is updated in place.

The split point is exactly the router, and for a correctness reason rather than a performance one: the CPU pool
needs the router's ids, and `moe_combine` needs the pool's answer, so they cannot be in one graph (`session.hpp`,
note on `SessionGraphs::posts`).

The sub-stages are exposed for measurement through `block_layer_pre`'s `stage_prefix` parameter, which captures
five **prefix** graphs so consecutive differences are per-stage times measured *outside* the capture — an event
recorded inside a capture is silently dropped, which was measured (`include/strata/core/graph.hpp`, note 1).

| stage | contents |
|---|---|
| 0 | `gr_read` (attn half), with the previous half's write folded in when `fused_gr` is on |
| 1 | the mixer: `gdn_layer` (36 layers) or `qsa_layer` (12 layers) |
| 2 | `gr_write` (mutates `R` in place) |
| 3 | `gr_read` (ffn half) |
| 4 | router + doorbell + shared expert |

`stage 1` is 55.8% of the pure-GPU floor per the `session.hpp` comment on R0.10 — the single largest kernel
target. QSA layers are `layer % 4 == 3` (`is_qsa_layer`, `include/strata/core/layout.hpp`), i.e. 12 of 48.

### 1.2 The host loop, in exact order

`session_loop` (`src/core/session.cpp:534`) is the schedule, and its order is the load-bearing part:

```
launch pre[0]
for l in 0..47:
    spin until the doorbell ring for layer l is observed
    hits(Launch)                        # GPU: quantize activation + grouped resident experts -> hit_out
    pool()                              # CPU: the MISSES, into one pinned host buffer
    cudaMemcpyAsync(parts_dev, y_miss)   # H2D from pinned memory
    hits(Combine)                        # GPU: parts += hit_out
    launch post[l]                       # combine + gr_write
    launch pre[l+1]
```

- **Publish.** `moe_route` (`src/core/layer.cpp:351`) writes the normed activation `x`, the 10 routed ids and
  their weights into **mapped pinned host memory** with one 1024-thread kernel, then increments a sequence
  counter: `doorbell_publish_kernel`, `src/kernels/cuda/elementwise.cu:240`, with `__threadfence_system()`
  ordering the payload before the ring. The ring itself is a *kernel* (`doorbell_ring_kernel`), not
  `cudaEventRecord`, because an event recorded inside a capture is silently dropped (`graph.hpp`).
- **Wait.** `session_loop` spins on the mapped `h_seq` *and* calls `cudaEventQuery` every iteration. Three
  experiments are recorded in the comment as saying the driver call is required on this driver: a memory-only
  spin never observes the device's write to mapped pinned memory, even with `__threadfence_system()` on the
  writer and a volatile read on the reader (`src/core/session.cpp`, comment inside the poll loop).
- **`post[l]`, not `post[l+1]`.** A fixed bug that mattered for correctness: the earlier form fed layer `l`'s
  expert outputs into layer `l+1`'s router weights and produced *"finite, fluent, deterministic tokens that were
  not the model's"* (`src/core/session.cpp`, comment at the `cudaMemcpyAsync`; also
  `src/program/generate.cpp:1691`).

### 1.3 Overlap budget: where the CPU actually runs

The overlap is **one layer, bounded by the tail of `pre[l]` after the ring**. The ring fires at the end of
`pre[l]`'s stage 4, and the only GPU work after it is the shared expert — so the pool's window is
*ring → `post[l]` launch* and nothing else. The engine instruments the split precisely:

- `SessionGraphs::rings_mid_graph` — the number of layers where the ring was observed while the graph was *still
  running* (`session.hpp`). A ring seen only after `cudaEventQuery` returned success means **the pool ran with
  the GPU idle**, which is invisible in the loop's return value and in a per-layer timing.
- `SessionGraphs::ms_to_ring` (launch → ring observed) and `ms_host` (ring → `pre[l+1]` launched). The header
  states the budget explicitly: *"The pool's window is `layer period - this`, and that is the entire budget the
  overlap has: if the CPU's per-layer work exceeds it, the GPU goes idle for the difference on every layer and a
  'pipelined' loop measures the same as a serialised one."*
- The loop's own exposed-loop accounting: `--no-pool` measures **38.73 ms/token against a 26.32 ms pure-GPU
  floor**, so ~12.4 ms/token (0.258 ms/layer) is spent in the loop with no expert work at all
  (`session.hpp`, `ms_host` comment). That is the honest "exposed host round trip" figure.
- A/B arm: `overlap = false` in `session_loop` inserts a `cudaStreamSynchronize` after every layer. The harsher
  `--sync-every-layer` is a *debug* mode that turns an asynchronous fault into a failure at the layer that caused
  it (`session_token`).

**Consequence for a port.** On Strata's schedule the CPU expert term is ~96% exposed. A bongo design that keeps
the CPU as the *primary* expert compute must budget the whole drain on the serial path; the only levers that move
it are (a) fewer bytes read on the CPU (residency) and (b) a kernel that needs fewer bytes per expert
(quantized-weight matmul). Neither is a scheduling change.

### 1.4 The one-graph-per-token variant

Plan v0.3 P3/P4 adds `session_capture_token` / `session_run_token` (`src/core/session.cpp:795` and `:869`): the
**whole 48-layer token as one graph**, with a device-side wait. Between `pre[l]` and the parts copy, a one-thread
kernel (`doorbell_wait_kernel`, `elementwise.cu:209`) spins on the device until the host writes the ring flag, so
the host makes no driver call in the loop except when it must poll:

```
for l: pre[l] (.. router, ring, shared expert) -> doorbell_wait -> parts <- y_miss (H2D)
        -> post[l] (combine, gr_write)
```

Rationale, verbatim (`session.hpp`): *"Under WDDM every graph launch costs ~0.3-0.4 ms of submission latency, and
the per-layer loop above makes 96 of them per token: measured 48.6 ms/token for 12 ms of GPU kernels and 27 ms of
pool."* Two out-of-band measurements fix the design: **5.2 µs per device-flag handoff**
(`bench/micro/device_wait.cu`) and **67 flushes/token** for a copy-engine node, hence a plain kernel rather than a
memcpy node for the handoff (`session.cpp`, comment at `copy_from_mapped`). The token graph is also where the
VRAM tier's *device-side* hit path lives (`TokenHits`, `session.hpp`), and it is the path the verifier requires.

### 1.5 Where the "expert stream / ring" lives

Three different mechanisms could be called "streaming". Keeping them apart matters, because the recon's §2.2
conflated two of them.

| # | Mechanism | Scope | Where |
|---|---|---|---|
| A | **Residency plus direct PCIe read of a share of the misses.** `--pcie-frac` is the share of each layer's distinct missed experts the GPU reads (default 0.2 for the Q2_0 pack, 0.55 for native packs, `src/program/generate.cpp:1073`). Modes: `dma` (copy engine into 16 staging slots), `direct` (the grouped kernel reads the mapped arena over PCIe), `kernel` (a copy kernel inside the graph) | decode, verify windows | `GpuPlanSink` in `include/strata/core/expert_source.hpp`; `Verifier::set_pcie_mode` and `Verifier::fetch_dma` at `src/core/verify.cpp:917`; `fetch_blobs`/`rebase_ptrs` in `src/kernels/cuda/verify_kernels.cu` |
| B | **A fixed expert ring over the copy engine, overlapped with the current layer's attention** | **prefill only** | `src/prefill/prefill.cpp:66-81` (`STAGE = 8`, `RING_MAX = 512`, `STREAM_ALL_MIN = 2048`, `ring_slots()` = 384 when ≥90% of streamed experts are DMA-able from pinned RAM, else 96) and the streaming loop at `:802-850` |
| C | **Helper-thread staging of unpinned experts.** A 16-slot ring of pinned host buffers (`kRing = 16`) filled by workers; a **generation-tagged CAS** claim means a late thread from the previous layer can never take a job of this one (the same class of bug as the pool's issue #29) | prefill, beside B | `struct Stager`, `src/prefill/prefill.cpp:118` |

Mechanism B is the one the recon quotes as *"non-resident experts streamed in a fixed ring overlapped with the
current layer's attention"* (1130 → 1290 prompt tok/s). Its shape: at a chunk of ≥2048 tokens nearly *every*
expert is routed, so the engine precomputes the whole chunk's stream **in fixed order, layer by layer, expert by
expert** (`seq` / `seq_start`), assigns entry `k` to ring slot `k % ring`, and gates each copy on that slot's
previous consumer via `cudaStreamWaitEvent`. Net effect: the copy engine works on layer `l+1`'s experts while
layer `l`'s attention runs. The streaming step is **bit-identical** to the unstreamed order and was checked with a
GDN-state hash over all 36 recurrent layers.

**The SSD is not in this path.** The expert arena is pinned host RAM (`ArenaExpertSource`,
`include/strata/core/expert_source.hpp:262+`). The SSD holds the PLE/n-gram table (§3.4) and the model file at
load time only.

### 1.6 CUDA graphs: what they bake in

Three constraints, each documented as a measurement rather than a preference (`include/strata/core/graph.hpp`):

1. A graph **re-reads its input buffers on replay but not its kernel arguments**. Anything that varies per step —
   position, token id, step counts — must therefore be *data* in a device buffer. This is why every QSA layer
   stages its per-token counts through `st.host_step` / `st.host_pos` (`qsa_layer`, `src/core/layer.cpp`) and why
   `session_loop` calls `stage_token` before replay. A bug where *every token after the first replayed position
   0* passed the single-token test and failed only on a multi-token prompt (`src/core/session.cpp`, comment at
   the top of `session_loop`).
2. A capture that produced **zero nodes** is an error, not an empty graph to replay happily.
3. Fixed addresses are the caller's obligation; the registry cannot check them.

---

## 2. The expert cache

### 2.1 `profile.bin` (STRP): the format, verified

Reader: `read_expert_profile`, `src/core/expert_cache.cpp:11-66`. Verified layout of the shipped file
(`data/expert-profile.bin`, 130,328 B); full parse and the checks are in
[`bench/results/2026-09-28-strata-profile-header/header.txt`](../../bench/results/2026-09-28-strata-profile-header/header.txt):

```
magic    "STRP"                                            4 B
header   version, n_layers, n_expert, slots, n_ranked      5 x uint32 LE
ranked   n_ranked x (uint16 layer, uint16 expert)          order = descending routing frequency
trailer  n_layers x n_expert uint32                        0xFFFFFFFF = unranked, else the pair's index
```

Shipped values: `version=1, n_layers=48, n_expert=512, slots=8000, n_ranked=8000`. The size closes exactly:
`24 + 8000*4 + 48*512*4 = 130,328`.

Two findings from the parse:

- **The trailer is a rank table, not a frequency table.** `expert_cache.hpp:42` calls it a frequency table; in
  fact all 24,576 entries are either `0xFFFFFFFF` (16,576 of them) or a permutation of `0…7,999` equal to each
  pair's own index in the ranked list. Both properties were checked cell by cell. This is harmless — the reader
  explicitly discards the trailer (*"the frequency table is not returned: it is the profile's own working, and the
  engine measures `h` by running rather than by re-deriving what the file claims"*) — but it means the file carries
  **only an order**, no absolute counts, and no other tool can rescore it offline.
- **The 8,000-slot profile is roughly layer-uniform**: per layer min 118, max 217, ≈167 average. That is a
  deliberate avoidance of the failure the same header records — a global arrival-order fill gave ~26 layers of
  position 0 every slot and measured **2.97%**, against **21.4%** for 8 slots/layer and **70.4%** for 64
  (`expert_cache.hpp`, `set_per_layer_admission`; `src/program/generate.cpp:172-181`). So at 8,000 slots the
  shipped profile is ≈167 slots/layer, well above the 64 slots/layer measured at 70.4%.

### 2.2 What workload the ranking was built on — **not published**

`tools/make_profile.py` is referenced three times by the engine (`src/program/generate.cpp:187`, `:377-378`, and
the `--expert-profile` help text) and `tools/routing_kfold.py` twice (`expert_cache.hpp`), and **neither file
exists in the public repo at `b89c989`** (checked with `git log --all --diff-filter=A`, `grep -rn`, and a listing
of `tools/`). The published `tools/` contains only packing / GGUF / MTP scripts (`strata_pack.py`,
`pack_index.py`, `iq_pack.py`, `mtp_*.py`, `gguf_*.py`, …).

What the repo *does* establish:

- `bench/results/2026-09-27-cache-parity/README.md` records the command actually used:
  `--expert-profile pack/profile-decode-8k.bin --expert-cache auto`, 5,130 slots, over "the first 512 or 1,024
  tokens of three frozen prompts" — i.e. a **decode** profile over short-prompt decode traces. Note that
  `pack/profile-decode-8k.bin` is *not* the file committed to `data/`.
- `data/expert-profile.bin` (8,000 slots) is what shipped installs use (`setup.py:969`).

**So the builder and its corpus are an open unknown (§7 U1/U2). The format is fully reverse-engineered and the
engine's consumer of it is readable in full, so a bongo-side builder can be written without upstream help.**

### 2.3 Transferability, hit rate, and why the profile is a *file*

The reason residency is read from a file is stated where the design question was measured
(`expert_cache.hpp`):

- `h_expert` at 4,105 slots, **ten-fold leave-one-out: 0.6447 [0.6110, 0.6750]** (`tools/routing_kfold.py`). The
  README's published 0.6573 is in-sample; a single-prompt corpus measured 0.4720. The transfer claim is therefore
  "a profile built on one trace retains ≈0.64 of its hit rate on a held-out one", and the file format exists so
  that claim can be scored against another prompt.
- Compulsory-miss (arrival-order) fill measured **0.4864**.
- `h_layer` held-out is **0.0456** — only 4.6% of (layer, token) pairs have *all ten* experts resident. This is
  the measurement that rules out a per-layer grouped kernel with a fixed grid and forces
  `moe_hit_grouped_s2`'s design ("hit list as data, capacity-sized grid, device-side count").
- The CPU drain is linear in `h`, so the 0.6447-vs-0.4864 gap is *"worth about 4 ms of drain"*.

### 2.4 Admittance policy — three policies plus an adaptive swap

| Policy | Where | Measured |
|---|---|---|
| Compulsory miss, arrival order (`admit()`) | `ExpertCache::admit`, `src/core/expert_cache.cpp` | h = 0.4864; at 256 slots **2.97%**, because one counter shared by all layers hands ~26 layers of position 0 every slot |
| **Per-layer split** (`set_per_layer_admission`, `--expert-cache-per-layer`) | layer `l` may only use slots `[l*q, (l+1)*q)` where `q = slots/n_layers` | same routing: **21.4%** at 8 slots/layer, **70.4%** at 64 |
| **Profile fill** (`--expert-profile`) | `read_expert_profile` + `fill_slot_blocking` + `verify_slot` | h = 0.6447 (leave-one-out) at 4,105 slots |
| **Adaptive swap, LFU with decay** (`--adapt-every`, default 4; `--adapt-swaps`, default 96) | `src/program/generate.cpp:2399` (serve) and `:3544` (decode) | reported at exit: "N experts swapped into the VRAM tier (every 4 rounds, x ms/round)" |

Mechanics of the non-default policies, because these are the reusable parts:

- **Profile fill uses the *blocking* copy** (`fill_slot_blocking`), and the header says why: at startup there is no
  stream ordering to lean on, the source is pageable host memory, and `verify_slot` reads the slot back on the
  legacy stream. The first version used the async form and `verify_slot` refused the whole run with *"slot 0
  differs from the arena at byte 0"*.
- **`verify_slot` byte-compares every filled slot against the arena**, described in the header as *"the only thing
  that says the cache holds the expert it claims to"*.
- **The adaptive swap is an LFU with decay, not an LRU.** Per `(layer, expert)` `usage` counts accumulate per
  layer dispatch (`src/core/expert_source.cpp:275-277`); every `--adapt-every` rounds every count is multiplied by
  **0.7**; a *non-resident* expert is a candidate if its decayed count is **≥ 2.0**; it is paired within its own
  layer against that layer's least-routed *resident* expert and swapped only if it exceeds the victim by
  **≥ 1.5**; at most 96 swaps per period. Copies are issued on a separate non-blocking stream and the residency
  tables are updated only when that stream's event completes (`apply_pending`); the evicted expert is marked
  `kNotResident` immediately so the CPU computes it in the meantime.
- **There is no LRU and no clock eviction.** The header states eviction policy is *"a measured question (`R4.1`'s
  LFU-decay vs LRU sweep)"* and that a placeholder would set the hit rate everything downstream is sized against.
  The adaptive swap is that LFU-decay, and it is opt-in (`--adapt-every 0`).
- **Slots are not necessarily uniform.** For native packs, `open_sized` gives each slot the blob size of *its own*
  profile pair, which the repo says holds ≈30% more experts than slots sized for the largest blob
  (`src/program/generate.cpp:1456-1470`, `--expert-cache auto`).

### 2.5 Is `--expert-cache` wired into compute? **Yes at this revision**

The recon ([`strata-ninfer-recon.md`](strata-ninfer-recon.md) §2.2, second bullet) states: *"The file states that
the cache is, today, **not wired into the compute graph** — it is slot storage + residency only, and
`--expert-cache` defaults to 0. So Strata's headline numbers are **not** from the expert cache; they are from the
kernels and speculation below."* **The first clause is wrong at `b89c989`**, and the engine says so in its own
source:

- `moe_hit_grouped_s2` exists (`include/strata/kernels/s2_expert_grouped.hpp`; implementation at
  `src/kernels/cuda/s2_expert_grouped.cu:327`).
- It is wired: `expert_hit_run` (`src/core/expert_source.cpp:440`) is called by `session_loop` at
  `HitPhase::Launch` (before the pool) and `HitPhase::Combine` (after the misses are staged), and by the token
  graph at capture time (`session_capture_token`).
- The `Options::expert_cache` comment was rewritten to say the previous comment *"was FALSE and ROUND 328 measured
  it"*: *"The kernel exists … it is wired at line ~660 via `expert_hit_run`, and switching the cache on **does**
  move work off the CPU pool: the drain fell **19.076 → 10.312 ms/token** at 4096 per-layer slots, for **−2.7
  ms/token** end to end"* (`src/program/generate.cpp:172-181`).
- The long "what this file is and is not, today" paragraph in `include/strata/core/expert_cache.hpp` — the
  paragraph the recon quoted — is **stale**, and the same file now contradicts it in `set_per_layer_admission` and
  in the `admit` doc.

The recon's second clause is **right and still important**: `--expert-cache` defaults to `0`, and shipped builds
enable it via `setup.py` as `--expert-cache auto` with the profile. Its contribution is bounded and measured —
**−2.7 ms on a ~53 ms token at 4096 slots (≈5%)** — which is *not* where the 6–8x lives. So the recon's overall
conclusion ("the headline numbers are from the kernels and speculation") holds; its mechanism claim does not.

### 2.6 Cache-on/cache-off output parity, and the one numerical contract that matters

`bench/results/2026-09-27-cache-parity/README.md`: teacher-forced on 2,557 tokens of code / document / chat, with
`--expert-profile pack/profile-decode-8k.bin --expert-cache auto` (5,130 slots) versus cache off.

| Text | same top-1 | median KL | perplexity off → on | Δ nats/token |
|---|---:|---:|---|---:|
| Code (512) | 97.7% | 3.0e-04 | 2.672 → 2.630 | −0.016 ± 0.019 |
| Document (1,024) | 96.2% | 4.5e-03 | 29.844 → 29.843 | −0.000 ± 0.005 |
| Chat (1,024) | 95.0% | 4.4e-03 | 18.380 → 18.301 | −0.004 ± 0.006 |

The upstream reading: the cache flips near-ties at 2–5% of positions, neither side is more accurate, and *"a GPU
expert and the CPU compute the same quantized expert with a different floating-point order."*

The **root cause of that divergence is documented and is a portability-relevant detail**: the GPU path read the
activation scale out of the `block_q8_0`'s fp16 `d`, while the CPU multiplies by an fp32 `ActQ::scale` — measured
as **4.761e-04 max relative on 80 of 80 chunks** (`bench/micro/act_quant_parity.cu`, cited in
`s2_expert_grouped.hpp` on `x_scales`). R4.2h added `quantize_q8_0_scaled` to pass the fp32 scales instead. **Any
bongo port that mixes a GPU expert path with a CPU expert path must settle this contract first**, or it will ship
a silent 2–5% top-1 divergence and spend a week attributing it.

---

## 3. Kernels

Paths are `include/strata/kernels/**` (declarations; those headers carry the design comments and the
measurements) and `src/kernels/cuda/**` (implementations). The class column is §6's vocabulary.

### 3.1 Quantized-weight dense GEMV / MMVQ (per-token path)

The per-token path never dequantizes a weight to run a generic GEMM. Two independent families exist, and which is
used is decided **per tensor**, not per role.

| Kernel | Weight forms | Activation | Replaces vs a generic path | Class |
|---|---|---|---|---|
| `native_mmvq.cu` (+ `native_mmvq.hpp`) | GGUF blocks, unmodified row-major: Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS | **Q8_1** — 36 B per 32 elements holding an fp16 scale *and* an fp16 warp sum — up to 8 columns | dequantize-to-FP16 + f16 GEMM | (a)/(b) |
| `s2_gemv_q8.cu`, `s2_gemv_quads.cu`, `s2_gemv_fast.cu` | Strata's own S2 (Q2_0) planar form | Q8_0 | same | (a) |
| `s_gemv.cu` | S2 / S4 / S8 canonical forms — **one kernel**, attributes passed as arguments | Q8_0 **or** Q8_K, carried per tensor as `SForm::act_kind` | same | (a) |
| `iq_kernels.cu` | transcribed llama.cpp i-quant decoders and dot products (IQ1_M … IQ4_NL) plus Q2_0; codebooks from `third_party/ggml` | q8_1 (the llama.cpp CUDA contract) | same | (a) |
| `bf16_gemv.cu` | plain BF16 matrices. Six tensors per layer are in this state (`bf16_gemv.hpp`): `ssm_alpha.weight`, `ssm_beta.weight` (GDN), `indexer.q_proj.weight` [2560,512] and `indexer.k_proj.weight` [2560,128] (QSA), `ple_value.weight` (layer 1 only), and `ffn_gate_inp.weight` (the router, handled inside `router_top10`). Has a row-split variant because with one thread per output row the parallelism *is* the output width — `ssm_alpha` has 48 rows over 48 SMs, and the plain router GEMV measured **0.2711 ms for a 2.6 MB weight**, ~65x its memory-bound floor and the largest item left inside the MoE after the top-10 was fixed | dequantize + generic GEMM | (a) |

Notes that matter for a port:

- `native_mmvq` keeps **weights in GGUF blocks, unmodified**; only the *activation* is quantized. That is the
  contract that makes the decode path fast: the bytes read per expert are the packed bytes, not 2–10 MB of FP16
  (`include/strata/prefill/moe_mmq.hpp`: *"The dequantize-to-FP16 + cuBLAS path wrote ~10 MB of FP16 per expert;
  this reads the ~1.4-2 MB expert once"*).
- `SForm::act_kind` is **carried, not derived**, with a worked example of why: Q5_0 and Q5_K are both 8-bit with
  bias −16 and differ only in `has_offset`; IQ4_NL and IQ4_XS differ in *nothing* the struct otherwise holds
  (`include/strata/kernels/s_gemv.hpp`). Deriving it produced a silent 0.6–1.4% GEMV error on 10 of the 12 QSA
  layers, caught only by a parity gate (`src/core/layer.cpp`, comment on `attn_q`).
- Because the pack mixes types per layer, `layer.cpp` produces **both** quantized images of the same activation
  (`quantize_q8_0` *and* `quantize_q8_K`) and each projection picks the one its own tensor wants
  (`gemv_quantized`, `src/core/layer.cpp`).

### 3.2 MoE

| Kernel | What it does | Replaces | Class |
|---|---|---|---|
| `router_top10.cu` / `native_router.cu` | softmax over all 512 experts, stable descending argsort with ties broken by index, gather, renormalise with ggml's `2**-14` clamp | a generic top-k | (a) |
| `s2_expert_grouped.cu` — `moe_hit_grouped_s2` (+ `_dev`, `_multi`, `_cpu_order`) | the VRAM-resident expert evaluation: **four launches per layer** (gate+up, SwiGLU, quantize, down), not five per expert. The hit list arrives **as device data** (`slot_index`, `dst_index`, device-side count) | a per-expert GEMM loop | (a) |
| `s2_expert_grouped.cu` — `moe_grouped_s2`, `moe_group_resident` | the verify-window form: entries grouped by blob, each row of a blob read once for all of its entries; a group's blob address may be a VRAM slot **or a mapped host blob read over PCIe** | same | (a) |
| `shared_expert.cu` | `silu(x·W_gate^T) * (x·W_up^T)`, then `·W_down^T`, times `sigmoid(x·g)`; the result is **added plain** to the routed output | a generic dense MLP | (a) |
| `native_moe.cu` | `parts` combined with *this* layer's router weights, shared added | elementwise | (a) |

The `dst_index` mechanism is the whole trick and is worth restating: a layer routes ten experts, only some of them
resident, and `moe_combine` weights `parts[i]` by **router position** `i`. So the CPU zeroes the hit rows and
writes the misses at their routed indices, and the GPU kernel writes each hit's answer to `dst_index[h]` — the
router's index for that expert. The combine then cannot tell which engine produced which row
(`include/strata/kernels/s2_expert_grouped.hpp`).

### 3.3 Attention

| Kernel | Role | Class |
|---|---|---|
| `qsa.cu`, `qsa_select.cu`, `qsa_decode_attn.cu` | QSA (the 12 full-attention layers): K/V append, indexer key pooling (**one key per 4-token block, one shared key head**), block scores, block top-k, gather, attend. `qsa_select` is FP32 block scores plus a radix selection over *blocks*; `qsa_decode_attn` is split-K decode attention reading the KV pools **through the page table** and serving all query heads that share a KV head from one read per chunk | (a) |
| `native_qsa.cu`, `native_qsa_indexer.cu`, `native_qsa_score.cu` | pinned-llama.cpp variants of the norm/gate, the indexer append, and the scorer; independently switchable | (a) |
| `kv_q8.cu`, `kv_q4.cu`, `kv_stream.cu` | KV storage: INT8 codes plus one fp16 scale per 64 values (1,024 B + 32 B per layer per token, against 2,048 B for fp16); optional Q4_0 after a 256-point Hadamard rotation; **KV streaming**, where the authoritative K/V lives in pinned host RAM and VRAM holds a page-table-addressed slot ring | (a) |
| `gdn.cu`, `fused_gdn.cu` | GDN (the 36 linear-attention layers): causal conv (kernel 4) + SiLU, L2 norm of q and k, beta and gate, the delta-net recurrence, output norm with `sigmoid(z)`. `fused_gdn` fuses (conv+SiLU+L2) and (step+out-norm) into two kernels and is the default | (a) |
| `native_gdn.cu`, `native_gdn_preprocess.cu` | pinned-llama.cpp GDN variants, selected by `native_gdn_set_enabled` | (a) |
| `native_flash_attn.cu` | a diagnostic short-context pinned attention (`--native-flash-attn-short`, context ≤ 256) | (a) |
| `rope.cu`, `native_rope.cu` | NEOX partial RoPE (64 of 256 channels touched); interleaved M-RoPE for vision | (a) |

`fused_gdn.hpp` carries the clearest bandwidth argument in the tree, and it generalises: the native step gives
each warp one `(head, column)` pair and strides 32 lanes over the 128 rows of a state stored `[row][head][col]`,
so *"every lane's load is 24.5 KB from its neighbour's … each 32 B sector delivers 4 useful bytes — 47.6 µs per
layer for 6.3 MB of state traffic"*. The fused version gives one block per head with `(row group of 32, column)`
threads, so a row's 128 columns are one contiguous 512 B load, and it folds the RMS norm and the `sigmoid(z)` gate
into the same kernel. **This is the roofline lens applied to a 6.2 MB/token state read: a port must keep the state
layout coalesced or pay ~7.5x on that term.**

### 3.4 PLE / n-gram — the SSD-resident shard

| Piece | What it does | Class |
|---|---|---|
| `src/kernels/ngram.cpp` / `include/strata/kernels/ngram.hpp` — `ngram_rows` | the **host** hash: 16 row indices per token from the last three token ids, 64-bit multiply/xor, no tensor op. The qwen4exp source says host-side because *"ggml has no int64 and no xor"* | (b) — pure host code |
| `PleTable` | the row gather: 16 rows x 160 values into a 2,560-wide vector, head-slowest like `ggml_get_rows`. Two modes: `Mmap` (16 separate page faults into a 26.8 GB mapping, measured **2.10–2.61 ms/token**) and `Direct` (unbuffered 4 KiB reads through `platform::DirectFile`) | (b) |
| `PleReader` (`include/strata/ngram/ple_reader.hpp`) | split `issue`/`collect` so the reads overlap embedding + layer 0; page dedup, offsets sorted, bounded in-flight depth for prefill; a bounded **row cache** (~1M rows ≈ 95 MB) that would serve *"up to ~82% of reads"*, with 20–34% intra-prompt recurrence | (b) |
| `ple.cu`, `native_ple_postops.cu` | the PLE block: grouped norms, key/query/value projections, `s[c] = Σ_d key·query / sqrt(n_embd)`, a gated update of the residual stack against a conv history; applied at **layer 1 only** | (a) |
| `ple_prefetch_enable` | one `PrefetchVirtualMemory` call for all 16 rows instead of 16 serial faults; A/B switch `--no-ple-prefetch`. Explicitly Windows-only: *"THERE IS NO LINUX, SO THERE IS NO MADV_RANDOM HERE"* | (c) on Linux — the same fix needs `madvise(MADV_RANDOM)` on the mapping instead |

### 3.5 Global residual (hyper-connection), head, sampler

| Kernel | Role | Class |
|---|---|---|
| `gr.cu`, `fused_gr.cu` | `gr_read` (collapse the 4-stream stack to the vector the mixer consumes, plus a per-stream injection) and `gr_write` (write the block output back per stream). The naive native path is **six kernels per `gr_read`, 96 + 96 times per token**; the fused version is two kernels with the previous half's write folded in | (a) |
| `native_gr_norm.cu`, `native_gr_postops.cu` | pinned-llama.cpp RMSNorm and GR post-ops | (a) |
| `src/core/native_head.cpp` | the LM head through the Q5_K MMVQ (`--native-head-gguf`), used by the verify window | (a) |
| `sampler.cu` | Philox sampling, greedy/temperature/top-k/top-p, min-p, penalty history | (a) |
| `cvec.cu` | the control vector (`--cvec-mode add|project`, per-layer directions) — the "*experimental speed projection*". Measured **0.2–0.4% cost** on the same text; it removes refusals, it is not an optimisation | (a) |

### 3.6 The batched prompt (prefill) path — a different kernel set

| Kernel | What it does | Class |
|---|---|---|
| `src/prefill/gemm.cu` | `cublasGemmEx` bf16/f16 for a chunk's projections; quantized weights are dequantized into a reusable FP16/BF16 scratch first (`dequant_bf16`) | (a) |
| `src/prefill/moe_mmq.cu` | **llama.cpp's MMQ** (`ggml-cuda/mmq.cuh`, MIT) compiled from the pinned upstream checkout: weights stay quantized, activations are rounded to q8_1, and the products run **on int8 tensor cores**. `gather_*` packs a group of experts into one buffer as their blobs arrive; one launch per product | (a)/(b) |
| `src/prefill/kernels.cu` | dequantize / group / gather / SwiGLU helpers, including `blob_dequant_kernel` for the Strata pack layout | (a) |
| `src/prefill/prefill.cpp` | the chunk orchestrator: projections as GEMMs, recurrences walked inside one kernel, the expert ring (§1.5 B), the PLE for the whole chunk at once | (a) |

`include/strata/prefill/moe_mmq.hpp` states the contract crisply: MMQ *"reads the ~1.4-2 MB expert once"* instead
of writing ~10 MB of FP16 per expert. Measured step: **1052 → 1130 prompt tok/s** (Q2_0, 32K). Note the decode
path uses **MMVQ** (`native_mmvq`, `__dp4a`-class int8 SIMT ops — 33 `__dp4a` sites in the tree) while the prefill
path uses **MMQ** (tensor cores). They are two different int8 paths, and the CMake build compiles only the MMQ
template instances for the types the packs use (`q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs`,
`CMakeLists.txt:528-537`); IQ1_M is explicitly not covered.

### 3.7 Speculation kernels

`src/kernels/cuda/verify_kernels.cu` (declared in `include/strata/kernels/verify_kernels.hpp`) exists only for the
verify window: multi-token GDN conv/L2 and step/norm that are **bitwise the single-token kernels applied per
token**, with a separate commit half that replays only the accepted prefix so the 113 MB recurrent state needs no
snapshot; device-id-driven embedding gather; `fetch_blobs` / `rebase_ptrs` for the PCIe share; and the MTP draft
layer's helpers (`add_streams_broadcast`, `ident_hits`, `mtp_select`, `gather_rows`, `map_ids`, `row_top_prob`,
`dense_steps`, `window_ids`).

---

## 4. Speculation

Three drafters and one verifier. All drafters are *advisory*: the verify window decides every emitted token, so a
drafting approximation can only change speed, never output (`include/strata/core/mtp.hpp`).

### 4.1 The verify window

`include/strata/core/verify.hpp`, `src/core/verify.cpp`.

> **T tokens at consecutive positions p₀ … p₀+T−1 — the last accepted token and T−1 drafts — go through all 48
> layers in ONE captured graph**, and the head's argmax is produced for every one of them.

- **How many tokens are verified at once.** `kVerifyMaxT = 8` (`verify_kernels.hpp`; `Verifier::init`), with one
  captured graph **per T** (`cudaGraphExec_t exec_[9]` plus `commit_exec_`). The shipped default is
  `--spec 4 --spec-min-p 0.5` (`setup.py:970`) — i.e. **T = 4 per pass: the last accepted token plus 3 drafts**.
  When the suffix drafter is on (default `--suffix-draft 3`) and no explicit `--mtp-max-t` is given, the engine
  caps the MTP at `spec` drafts and grows the *window* to `min(spec + 2, 8) = 6`, so a long lookup match can use a
  longer window (`src/program/generate.cpp:976-978`).
- **Exactness.** Verbatim: *"Token t's argmax is what plain greedy decode would produce after token t, BIT FOR
  BIT: every kernel here is either the single-token kernel applied per token, or a multi-token kernel whose
  per-token arithmetic is the single-token kernel's (multi-column MMVQ in exact mode, the T-token GDN kernels, the
  per-token hit activation, the multi-token CPU expert rows). So a draft is accepted exactly when greedy decode
  would have produced it."* Greedy only — the sampler runs outside the captured graph precisely because its
  parameters would otherwise be baked forever (`Verifier::set_sampling`).
- **Acceptance is measured in-process, not inferred**: `while (a < T-1 && window[a+1] == outv[a]) ++a;`, then
  `commit(a+1)`. Reported at exit as `drafts accepted k of n (ratio)`, `tokens per round`, an accepted-per-round
  histogram and a window-size histogram (`src/program/generate.cpp`).
- **Published acceptance.** The MTP draft layer accepts **0.89 / 0.86 / 0.85** of greedy drafts at steps 1–3
  (`include/strata/core/mtp.hpp`, from `tools/mtp_probe.py`). `docs/DETAILS.md` reports **2.4–3.2 tokens committed
  per pass** on average and **1.6–1.8x** overall. Suffix / prompt-lookup acceptance on copied or edited text is
  **≈91%** (`docs/DETAILS.md`; the ESP README notes a code prompt where the model copied its source and prompt
  lookup drafted at 91%).
- **State handling** (`verify.hpp`). The window appends K/V and indexer keys for all T positions and leaves the
  GDN state untouched. `commit(n_keep)` then (i) advances the GDN conv history and recurrent state by *replaying*
  the first `n_keep` tokens from inputs the window stored, (ii) restores the indexer key tail from a snapshot and
  re-appends the accepted keys (a rejected key can land in a slot the current block still needs), and (iii) sets
  the PLE history to its snapshot after token `n_keep − 1`. K/V cells past the accepted prefix are overwritten
  when those positions are processed again. **No snapshot of the 113 MB recurrent state is kept.**
- **The window's cost is the design constraint.** From `verify.hpp`: the dense weights are read *once* for T
  tokens, but the union of the T tokens' **missed** experts grows — *"measured on decode traces: 1.75x one token's
  misses for T=2, 2.4x for 3, 3.05x for 4"*. That is the term that bounds the window, and the reason the window is
  not simply made as long as possible.

### 4.2 Drafter A — the MTP draft layer (GPU; needs the MTP head)

`include/strata/core/mtp.hpp`, `src/core/mtp.cpp`. One extra layer (vLLM 0.30.0 `qwen4_exp` MTP), ≈0.9 GB of VRAM,
708 MB of which is the 512 experts — **all resident**. Cell `i` pairs the main model's final 4-stream residual at
position `i` with the token at `i+1` at RoPE position `i`; its output predicts the token at `i+2`, and its own
residual feeds the next draft step. Its runtime approximations only affect *draft quality*: large projections run
as Q8_0 through multi-column MMVQ; attention is **dense** over the cells the layer has seen (identical to the
model's sparse selection below 2,051 cells, so speculative cells never touch indexer state); and all 512 experts
run through `moe_group_resident`.

**bongo cannot use this** — `CONTEXT.md`: the published GGUF drops the MTP head and llama.cpp `qwen4exp` cannot
convert or run it. It is included here because it is the reference for the *verify* structure and because its
acceptance numbers set the ceiling the suffix drafter has to beat.

### 4.3 Drafter B — suffix / prompt lookup (no weights, no GPU)

`include/strata/spec/suffix_drafter.hpp`, `src/spec/suffix_drafter.cpp`. Every trigram of history (prompt plus
accepted output) maps to its **4 most recent end positions** in a fixed-size open-addressing table
(`WAYS = 4`, `min_match = 3`, `max_match = 32`, ≈20 bytes per history token, `O(1)` appends). A proposal extends
each candidate backwards, takes the longest match (most recent on ties), and proposes up to `max_k` following
tokens. This is the **only speculation lever available to bongo**, and the recon already routed it to the plan.

Measured value (`docs/DETAILS.md`, engine 0.1.7): code edits **6–11% faster**, other text unchanged, and the
drafts are verified like the MTP's so the output is identical.

### 4.4 Which drafter, and how long — the selection policy

Two layers, both online-learning, both affecting *which drafts are verified* and never the output:

- `spec::DraftPolicy` (`include/strata/spec/draft_policy.hpp`) chooses **MTP window vs lookup window** once per
  round by comparing expected committed tokens per millisecond, with `margin = 0.03`. Its lookup side is
  `E(k) = 1 + q + q² + … + q^k`, where `q` is an EMA **per match-length bucket** (4 buckets: `<5`, `<8`, `<16`,
  `≥16`, because *"a long match is far more reliable than a trigram"*), and round costs are the measured
  per-window-size EMAs (unseen sizes scaled from seen ones by a prior shape). The header records why this exists:
  taking the lookup window whenever it proposed more than the MTP *"lost 2-8% on ordinary text: a long lookup
  window costs much more to verify than the MTP's usual 3-4 tokens"*.
- `spec::Controller` (`include/strata/spec/controller.hpp`) is the analytic version: maximise
  `E[tokens committed] / T(step)` over `{none, lookup, MTP}`, with `K_MAX = 8`, a cost model
  `T(n) = dense(n) + experts(n) + sync + draft`, and `min_gain = 0.05`. Its defaults are dated 23-Sep
  measurements: `dense_ms = 11.0`; `dense_ratio` for n = 1…9 = 1.0/1.05/1.3/1.45/1.85/2.2/2.6/3.0/3.3;
  `cpu_all_miss_ms = 15.8` (480 experts on CPU); `hit_rate = 0.55`; `distinct_ratio` 1.0…5.2;
  `extra_use_cost = 0.25`; `sync_ms = 2.4`; `mtp_draft_ms = 1.2`; `lookup_draft_ms = 0.01`. The header says to
  replace them with in-engine measurements through `CostModel` — **these constants are the most directly reusable
  artifact in the speculation subsystem for a bongo cost model.**

Also present and **measured and rejected**: `--spec-split` splits the window into two token groups and pipelines
one group's CPU experts with the other group's GPU work. It is exact but **~7% slower** and off by default
(`src/program/generate.cpp:908`, `Verifier::set_split`). Recorded because it is the natural idea.

---

## 5. Memory plan

### 5.1 Geometry — every field is a kernel contract, not a tuning knob

From `include/strata/core/layout.hpp` and `include/strata/plan/plan.hpp`:

| | value | note |
|---|---|---|
| `n_embd` | 2,560 | |
| `n_layers` / `qsa_interval` | 48 / 4 | QSA at `layer % 4 == 3` → 12 layers; GDN → 36 |
| GDN | `state=128, k_heads=16, v_heads=48, d_conv=4, conv_channels=10240, value_dim=6144` | |
| QSA | `n_head=24, n_head_kv=2, head_dim=256` | |
| indexer | `q_heads=4 x 128` (**not cached**), **`key_heads=1 x 128` (cached, pooled one per 4 tokens)** | the plan comment records this term being first 2x and then 4x too big: *"A head count in the metadata does not say WHICH tensor it counts"* |
| residual | `hc = 4` streams, `hc_lr = 320` | |
| MoE | `n_expert = 512`, `n_ff = 640`, top-**10** | 24,576 experts total |
| expert blob | **1,382,400 B** = `3 * (2560*640*18/64)` | |

### 5.2 The arithmetic

`include/strata/plan/plan.hpp`:

```
pool          = 5,943,000,000 B          # the ONE number taken from a table rather than derived
kv_bytes      = 13,056 B/token x max_context
state_bytes   = 36 x (128*48*128*4 + 3*10240*4) = 117,669,888 B   # fixed, NOT evictable
fixed         = dense + embd + mtp + workspace + state
if kv_bytes + fixed > pool: throw DoesNotClose   # refuse, do not overcommit
cache_bytes   = pool - kv_bytes - fixed
cache_slots   = cache_bytes / 1,382,400
vram_used     = pool - cache_bytes % 1,382,400   # the slot remainder is unusable, not free
if cache_slots == 0: throw DoesNotClose
```

Worked numbers for the geometry above:

| term | bytes/token | 4K | 32K | 128K | 262K |
|---|---:|---:|---:|---:|---:|
| INT8 KV (`2 * 2 * 256 = 1,024` code bytes + `32` scale bytes per layer, x12) | 12,672 | 51.9 MB | 415 MB | 1.66 GB | 3.32 GB |
| indexer keys (`1 x 128 / 4` per layer, x12) | 384 | 1.6 MB | 12.6 MB | 50.3 MB | 100.7 MB |
| **`kv_bytes_per_token`** | **13,056** | 53.5 MB | 428 MB | **1.712 GB** | 3.423 GB |
| GDN recurrence + conv history (fixed) | — | 117.7 MB | 117.7 MB | 117.7 MB | 117.7 MB |

Three points the upstream comments insist on, each from a real error:

1. **The GDN state is a fixed cost that was silently zero.** `Costs::state_bytes` existed and `--state` existed,
   but nothing passed it, so the planner planned without **117.7 MB = 85 expert slots**. `strata-plan` now
   defaults it from the geometry (`src/plan/plan_main.cpp:58`).
2. **The slot remainder is not free** (`vram_used = pool − cache_bytes % blob`).
3. **A plan that returns something it cannot honour is worse than one that throws**, because the failure then
   happens at token 4000 instead of at startup (`DoesNotClose`).

### 5.3 The live engine does not call `make_plan`

This matters for a port. `plan.hpp` / `strata-plan` is the *design tool* that makes the arithmetic testable
without a GPU. The engine sizes the tier from a **live measurement** instead: `--expert-cache auto` reads
`cudaMemGetInfo`, subtracts `--vram-reserve-mib` (default 700) and any prefill-chunk reservation, and divides by
the expert blob (`src/program/generate.cpp:1437-1451`). Under WDDM an allocation is **not resident until it is
touched**, so it then zeroes the slots, re-reads free VRAM and shrinks the cache in up to six attempts until the
reserve is free (`src/program/generate.cpp:1473-1513`). A first version sized from the pre-allocation figure filled
the card to 0 MiB; the driver then paged, and a request that needed a page back while the verify graph spun on a
host flag never finished.

### 5.4 The VRAM / RAM / SSD split

| Tier | Contents | Size | Source |
|---|---|---|---|
| **VRAM** | dense weights in **engine form**, embedding, session (KV + indexer + GDN state + scratch), MTP draft layer (≈0.9 GB), LM head, and **all remaining VRAM as expert-cache slots** | engine form **4.067 GiB dense + 0.444 GiB embd + 10.88 MiB widened scales = 4.522 GiB** (`weights.hpp:17-18`) on a 12 GB card; measured **6,414 MiB free** at steady state versus **5,413 MiB** for 4,105 slots (`expert_cache.hpp`) | |
| **RAM** | **all 24,576 experts, pinned** (33.97 GB); the K/V cache when KV streaming is on (~13.7 KB/token = 1.7 GB at 128K); up to 6 conversation checkpoints (~118 MB each); the PLE row cache in `Direct` mode | 33.97 GB of experts | `docs/DETAILS.md`, `expert_source.hpp:250` |
| **SSD** | the PLE / n-gram table **only**: 320,001,536 rows x 90 B = **26.8 GiB**, read a few rows per token, never held in RAM; the model file at load time | 26.8 GiB | `include/strata/ngram/ple_reader.hpp` |

Two refinements worth carrying over:

- **KV streaming** (`--kv-resident N`, engine 0.1.5). From 64K up, the authoritative K/V lives in pinned host RAM
  and only `N` cells per QSA layer stay in VRAM, addressed through a residency **page table** so every reader is
  unchanged (`include/strata/kernels/kv_stream.hpp`). Measured: Q2_0 at 262K **50.9 → 62.6 tok/s**
  (1,589 → 3,872 experts in VRAM — **the freed VRAM buys expert slots**); ~13.7 KB of RAM per context token with
  INT8 KV.
- **The prompt path borrows the top expert-cache slots** for its chunk buffers and refills them afterwards
  (`--no-prefill-borrow` reserves them for the whole session instead; `setup.py` passes `--prefill auto`).
  Borrowing is why `--prefill auto` can pick the *largest* chunk whose buffers fit — 973 prompt tok/s at 8,192
  versus 791 at 4,096 (`bench/results/2026-09-28-prefill-speed/README.md`) — and it caused a real corruption bug
  fixed in the very commit read here (*"prompt path: never write into the borrowable cache slots while laying
  out (a resident expert was corrupted)"*). **Any bongo scheme that lends cache slots to another phase needs an
  explicit ownership rule at the layout step, not an assertion.**

---

## 6. Portability to Arc B70 / SYCL

Classes: **(a) CUDA-specific** — no SYCL path without a rewrite; **(b) portable** — llama.cpp/ggml already has a
SYCL implementation of the same primitive, or the code is plain C++ host code; **(c) new SYCL work** — no
existing equivalent; must be written and validated on the B70.

### 6.1 Scheduling and submission

| Mechanism | Strata path | Class | bongo path |
|---|---|---|---|
| Per-layer graph capture/replay, inputs as fixed-address data only | `graph.hpp`, `session_capture` | **(c)** | SYCL graph (`sycl::ext::oneapi::experimental::command_graph`) or Level-Zero immediate command lists. llama.cpp's SYCL backend does not use graphs today. **This is the largest single mechanism to port and the largest risk** — and its whole value (1.51x on the launch-bound part) depends on B70/Level-Zero submission cost being comparable to the 10–22 µs/driver-call measured here |
| Two graphs per layer split at the router; host order `ring → hits → pool → parts → combine → post → next pre` | `session.cpp:534` | **(b)** | plain host scheduling; no CUDA concept beyond "launch a recorded sequence" |
| Doorbell: mapped pinned payload plus a sequence word ordered by `__threadfence_system` | `elementwise.cu:204`, `:240` | **(c)** | SYCL USM **shared** allocation is the same device-visible/host-writable object. The ordering primitive becomes a `sycl::atomic_ref` release store. **The measured "the write only becomes host-visible on driver entry" behaviour must be re-measured on Level Zero, not assumed** — three CUDA experiments were spent on it |
| Device-side wait on a host-written flag inside the graph (`doorbell_wait_kernel`, 5.2 µs/handoff) | `elementwise.cu:209`, `session_capture_token` | **(c)** | a SYCL kernel spinning on USM shared memory; must be validated for forward progress, and for whether a submission is needed before the host's write is observed |
| Pinned host staging for the pool answer (`cudaMemcpyAsync` from pageable memory is **not** async — it bounces synchronously) | `SessionLoopScratch`, `session.cpp` | **(b)** | `sycl::malloc_host` + `queue.memcpy`. The DMA-versus-bounce property is not guaranteed to be the same on Level Zero — measure it |
| One whole token as one graph (plan v0.3 P3) | `session_capture_token` | **(c)** | same graph API. Note the motive was **WDDM** submission latency (~0.3–0.4 ms/launch, 96 launches/token); on Linux/Level Zero that motive is weaker and must be re-measured rather than ported |
| Helper-thread staging ring with generation-tagged CAS claims | `struct Stager`, `prefill.cpp:118` | **(b)** | pure host threads; the CAS discipline is the reusable part |

### 6.2 Expert cache

| Mechanism | Class | bongo path |
|---|---|---|
| STRP profile **read** (format fully specified in §2.1, including the rank-trailer correction) | **(b)** | plain C++/Python; can be written today |
| Profile **builder** and its leave-one-out scoring (`make_profile.py`, `routing_kfold.py`) | **(c)** | not published. New work: build from a bongo routing trace and score it held-out (§7 U1/U2) |
| Per-layer admission split (`q = slots / n_layers`) | **(b)** | arithmetic on the residency table |
| LFU-decay adaptive swap on a second queue, residency table updated on an event | **(b)** | SYCL queue + event; residency table in shared/device memory |
| Slot fill with **byte-compare verification** (`verify_slot`) | **(b)** | plain copies. Keep the check — it caught a real corruption |
| Uniform versus per-layer sized slots (`open_sized`) | **(b)** | arithmetic |
| VRAM-slot-write-then-re-measure-and-shrink sizing | **(b)** | the analogous B70 behaviour (VRAM not resident until touched; `xe` driver) must be re-measured |
| `moe_hit_grouped_s2` — grouped expert over a slot arena, hit list as device data, device-side count | **(a)** | must be written in SYCL/ESIMD. The *structure* (4 launches/layer, `dst_index` as data, capacity-sized grid) ports directly |
| The GPU/CPU activation-scale contract (fp32 scales on both sides, R4.2h) | **(b)** | a design rule, not code. **Adopt it before the first mixed GPU/CPU expert path exists** (§2.6) |

### 6.3 Kernels

| Kernel family | Class | SYCL / oneAPI mapping |
|---|---|---|
| MMVQ over GGUF blocks (Q2_0, IQ2_XS, IQ4_XS, Q3_K/Q4_K/Q5_K/Q6_K/Q8_0) with a Q8_1 (fp16 scale + fp16 sum) activation | **(b)** | **llama.cpp's SYCL backend already implements MMVQ/MMQ for these types with the same layouts.** Highest-value, lowest-risk port in this document |
| MMQ on int8 tensor cores for the prompt path | **(b)** | llama.cpp SYCL MMQ uses `joint_matrix` / DPAS. The packs' types are covered, but bongo must check the template-instance list (upstream does not instantiate every type, and Strata's own list is `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs`) |
| Strata's own S2/S4/S8 GEMV with `__dp4a`, one thread per row | **(a)** | rewrite with SYCL `sycl::dot` on `char`/`short` vectors, or drop entirely in favour of the llama.cpp SYCL MMVQ for the same type (the canonical-form machinery exists to serve a pack format bongo does not have) |
| i-quant decoders/dots transcribed from llama.cpp CUDA (33 `__dp4a` sites) | **(b)** | the algorithms are llama.cpp's and the SYCL backend has them; the *transcription* is what is CUDA-specific |
| Warp shuffles for reductions (`__shfl_xor_sync` 45 occurrences, `__shfl_down_sync` 30, across ~33 files) | **(b)** | `sub_group::reduce_over_group` / `sycl::ext::oneapi::reduce`. **The layouts chosen for warp-strided loads may need re-tuning for 32-wide Xe subgroups** — treat every "measured at 32 threads/row" comment as CUDA-specific evidence |
| Grouped expert over a slot arena (`moe_hit_grouped_s2`, `moe_grouped_s2`, `moe_grouped_resident`) | **(a)/(c)** | new SYCL/ESIMD work. Prerequisite for the bongo expert tier |
| GDN fused conv+L2 and step+out-norm, with the state layout `[row][head][col]` fixed to be coalesced | **(a)** | new SYCL kernel; the *layout argument* is the deliverable, not the code (§3.3). One block per head, 32-row groups x columns |
| QSA block scores (FP32) + radix top-k over blocks | **(a)** | new SYCL; a segment-sort/radix on Arc, or a two-pass reduction. This is the long-context *selection* cost, ~0.1–0.2 ms per query per QSA layer at 32K in the unoptimised form |
| Split-K decode attention reading KV pools through a page table, serving all Q heads sharing a KV head from one read | **(a)** | new SYCL; llama.cpp SYCL has flash-attention but not a page-table KV layout |
| INT8 KV (codes + fp16 scale per 64) and Q4_0 KV with a 256-point Hadamard rotation | **(b)** | straightforward SYCL; the Hadamard rotation is a data transform, and the PR cites the precision trade (perplexity +8–12%) |
| KV streaming (authoritative K/V in host RAM, VRAM holds a paged slot ring, every reader unchanged) | **(c)** | needs USM host memory plus a device-written residency map. **The concept — not the code — is the important transfer**, because it trades attention-side VRAM for expert slots |
| PLE host hash (`ngram_rows`, 64-bit multiply/xor) | **(b)** | already host code; no GPU work |
| PLE / n-gram table reads: `issue`/`collect` split, page dedup, bounded in-flight depth, bounded row cache, 16 rows/token | **(b)** | POSIX `pread`/`io_uring` instead of `DirectFile`; **the row cache and prefetch-overlap design ports directly** and is exactly what BAS-65 needs |
| 16 serial page faults → one prefetch call (`PrefetchVirtualMemory`) | **(c)** | the Windows API has no Linux equivalent used here; on Linux use `madvise(MADV_RANDOM)` on the mapping and an explicit batched read (see `ngram.hpp`'s own note) |
| Transcribed hyper-connection kernels (2 kernels instead of 6 per read, previous write folded in) | **(a)** | new SYCL; the *fusion structure* (2 vs 6 launches per `gr_read`, 96+96 per token) is the ported idea |
| MMVQ LM head (`native_head.cpp`) | **(b)** | llama.cpp SYCL supports MMVQ output heads |
| Sampler (Philox, greedy/top-k/top-p/min-p/penalties) | **(b)** | portable; host or trivial device code |
| Control vector (`--cvec-mode add|project`) | **(a)** | trivial kernel, but note the *finding*: it is 0.2–0.4% cost and it removes refusals — it is a quality/safety change, not an optimisation. bongo should not adopt it |
| `cudaMalloc` inside a capture is illegal; per-token host allocations implicitly synchronise (`cudaFreeHost`) | **(b)** | a design rule that applies identically on Level Zero: no allocation on the token path, no implicit sync |

### 6.4 The two portability boundaries that are *not* about code

1. **WDDM is a Windows property, and several load-bearing measurements are WDDM measurements**: 96 graph
   launches/token at 0.3–0.4 ms each (the motive for the whole-token graph), lazy submission needing a driver
   entry to flush, allocation not resident until touched. On Fedora 44 + Level Zero none of these is assumed true.
   **Re-measure the submission cost per graph launch on the B70 first**; if it is ~50 µs, the token-graph work is
   worth less than the per-layer loop.
2. **`sm_120` is hard-coded.** `CMakeLists.txt` refuses to build for compute capability < 8.0, defaults to 120,
   and says the engine *"REJECTS any architecture other than 120"* at run time (`include/strata/core/device.hpp`).
   Every "measured" number in the headers is an RTX 5070 number. Treat all of them as **relative** evidence and
   re-derive the absolute terms on the B70.

---

## 7. Open unknowns, each with the experiment that would resolve it

| # | Unknown | Why it matters | Exact resolution |
|---|---|---|---|
| U1 | **What workload built `data/expert-profile.bin` and `pack/profile-decode-8k.bin`** — corpus, length, sampling, and whether the ranking is over the whole trace or per layer. `tools/make_profile.py` is not published. | A bongo profile is only useful if the builder's sampling matches bongo's workload. The shipped file's ~layer-uniform 8,000-pair shape is evidence but not proof. | Write a bongo-side builder (the format is in §2.1), build two profiles from two *different* prompt corpora, and score each on the other (leave-one-out as `routing_kfold.py` does). If cross-corpus `h` collapses toward 0.4864, the ranking is corpus-specific and bongo must build per-workload profiles. |
| U2 | **Does a profile transfer across *prompt length* and *tier*?** The published measurements are at 4,105 slots with `h_expert` LOO 0.6447, but the shipped file is 8,000 slots, and the cache-parity bench used 5,130. | bongo's reference box has 32 GB RAM + 32 GiB VRAM against Strata's 64 GB RAM, so bongo's hit-rate-vs-slots curve starts from a *smaller* resident fraction — the curve's shape near the low end decides whether the tier is worth building at all. | Sweep slots {512, 1024, 2048, 4096, 8000} × context {4K, 32K, 128K} on a bongo routing trace; report `h_expert` and `h_layer` per cell. This is BAS-64's question; this document only fixes the metric definitions. |
| U3 | **Whether Strata's per-layer split beats a bongo-specific admission plan** (global-with-eviction, per-layer, or a predictor). The header itself calls eviction *"a measured question (`R4.1`'s LFU-decay vs LRU sweep)"* and the LFU constants (decay 0.7, candidate ≥ 2.0, gain ≥ 1.5, 96 swaps / 4 rounds) are hand-set, not swept. | These constants set the steady-state hit rate, and the adaptive swap runs concurrently with the verify window — a bad constant costs performance without changing correctness, so it will not be caught by any parity test. | A/B the four constants in one session on the reference box (the harness already supports per-run flags). Report drain ms/token, not tok/s, so the comparison is not confounded by speculation. |
| U4 | **The true cost of a graph launch on the Arc B70 through Level Zero** — and therefore whether per-layer graphs or a whole-token command list is the right unit. | It is the single largest ported mechanism (item 1 in the TL;DR) and its value is entirely set by this number. | Microbenchmark: replay a captured graph with N nodes, 1,000 times, against the same N kernels launched individually. Report µs per submission and the crossover node count. This is the first experiment the port should run. **Measured 2026-09-28 — see §9: 1.34 µs per submission, no crossover.** |
| U5 | **Whether a device-side spin on USM shared memory is observed by the host without a submission** (the CUDA `__threadfence_system` + driver-entry behaviour). | The whole-token-graph overlap design depends on it, and on CUDA it cost three experiments to find out. | Write the two-thread probe (`bench/micro/device_wait.cu`'s analogue) on Level Zero: kernel writes a USM flag, host spins and timestamps; then with an intervening `zeCommandList` submission. Report host-visible latency both ways. **Measured 2026-09-28 — see §9: device→host free, host→device impossible.** |
| U6 | **The `msg`/`h_layer = 0.0456` grouped-kernel design on Xe**: whether one capacity-sized grid with a device-side count and a device-side `dst_index` vector is as effective in SYCL as in CUDA. | It is what makes the hit path expressible as four launches per layer instead of five per expert. | Port `moe_hit_grouped_s2` for one tier and one layer, and compare against the CPU pool for the same layer at the same hit list, measuring the layer's GPU time and the CPU drain separately. |
| U7 | **Whether the B70's VRAM behaves like WDDM for the slice-sizing loop** (not resident until touched, `cudaMemGetInfo`-style free readings being optimistic). | The shrink loop exists because a naive sizing filled the card to 0 MiB and hung a request. | Allocate a large VRAM block on the B70, read the free figure before and after touching it, and check whether a subsequent allocation of the reserve succeeds. If the B70 reports honestly, the loop is unnecessary complexity in a bongo port. |
| U8 | **Whether the n-gram/PLE `issue`/`collect` overlap window (embedding + layer 0) is long enough on the B70** for the Direct-read path. Strata needs 16 rows on 16 different 4 KiB pages; the window is embedding + layer 0. | It decides whether bongo needs the row cache (≈95 MB for 1M rows, ~82% of reads) or can afford raw reads. | BAS-65 measures exactly this on the reference box; this document supplies the design being measured (page dedup, bounded in-flight depth, `issue` before layer 1). |
| U9 | **The `session_loop` vs token-graph choice under the spec window's device waits**: Strata requires the token graph for `--spec`, and `--spec-split` (the more overlap-friendly order) measured ~7% *slower*. | It tells bongo whether to invest in the wait-flag machinery at all, or to run the verifier with a per-layer loop and accept 96 submissions/token. | On the B70, run the verifier with (a) per-layer graphs and host-side waits and (b) a single token graph with device-side waits, at the same T, and report the exposed `ms_wait` and `ms_pool` sums. |

Not investigated here, listed so the next reader does not assume coverage:

- `serve/` (the HTTP layer, prompt cache, checkpointing, KV streaming on the server path) — read only where it
  determines engine flags (`setup.py`, `serve/server.py`), not audited.
- `src/artifact/**` (GGUF reader, dequant), `tools/strata_pack.py`, `tools/pack_index.py` (the pack format and
  index) — these describe a bongo-independent artifact format; they matter only if bongo adopts a pack of its own.
- `docs/paper/Strata-Paper.pdf` — a binary in the repo, not read here; the engine's comments cite the same
  measurements that appear in `bench/results/`.
- The `Memory/`, `phases/`, `docs/` in-tree design notes referenced throughout the source (`Memory/R4-design-note.md`
  §7, `docs/semantics.md`, `docs/pack-format.md`, `docs/activation-contract.md`, `phases/phase-2-correct-engine.md`,
  `bench/micro/*`). They are **not in the published tree** at this revision, which is why some measurements are
  quoted here only second-hand from the headers that cite them.

---

## 8. What to take, in order

Ordered by expected value per unit of bongo engineering, derived from the measured deltas in §1–§5:

1. **Fixed from llama.cpp's SYCL backend: an MMVQ/MMQ expert path that never materialises FP16 weights.**
   Biggest single term in the modelled token, and the only one with an existing Arc implementation.
2. **A VRAM expert tier with a *computed* per-layer split and a profile/trace-driven ranking** — plus the
   fp32-activation-scale contract (§2.6) decided up front. Bounded upside on Strata's token (−2.7 ms), but bongo's
   box has less RAM, so the CPU drain starts *higher* and the same removal is worth more.
3. **Submission-cost measurement on the B70, then graph-per-layer or command-list-per-token** (U4/U5). Do not
   port the workaround before measuring the cost it works around.
4. **Batched-chunk prefill with an expert ring overlapped on the attention half, plus PLE prefetch.** The prefill
   numbers are the largest percentage gains in the repo (572 → 1,290 prompt tok/s), and BAS-65 covers the PLE half.
5. **Suffix/prompt-lookup speculation with the exact verify window.** Strata's own numbers are the modest ones
   (6–11% on code edits, unchanged elsewhere) and its cost model constants are published in
   `spec/controller.hpp`; adopt the *policy* (learn the acceptance online, compare expected tokens per
   millisecond) rather than a fixed window.

---

## 9. U4 and U5 measured on the B70 (2026-09-28, [BAS-70](/BAS/issues/BAS-70))

Raw data: [`bench/results/2026-09-28-levelzero-submission/`](../../bench/results/2026-09-28-levelzero-submission/README.md);
full write-up: [`levelzero-submission.md`](levelzero-submission.md). Measured through `ze_api.h`
(ctypes) because the box has no SYCL runtime or DPC++ compiler; the kernels are hand-assembled SPIR-V.

- **U4 — one Level Zero submission costs 1.34 µs** (closed command list, N = 0…200 nodes), flat in N
  and identical cold and warm. Submitting the same nodes one at a time costs the same 1.4 µs each, so
  there is **no crossover**; batching only removes *submissions*, `(N-1) × 1.4 µs`. Appending a node
  into a list (capture) costs 0.88–1.36 µs and is paid once. **96 submissions/token = 0.13 ms/token**
  against the 29–38 ms/token the WDDM motive in §1.4 assumed: the per-layer-graph rationale is a
  Windows property and does not transfer. What survives is a much smaller claim — a captured list
  recovers the ~1 µs/node append cost, ~2.8 ms/token for ~2,000 nodes — and that needs one command
  list per token, not 96 graph submissions.
- **U5 — the doorbell is half-usable.** A device store to USM shared memory *is* host-visible with no
  submission and no driver entry (30/30, median 33.7 µs, observed while the kernel was still spinning
  for 34 ms). A host store to the same memory is **never** seen by a running kernel: 0/30 with no
  driver call, 0/30 with a `zeFenceQueryStatus`, 0/30 with an extra queue submission, 0/30 with a
  `BIAS_UNCACHED` allocation — while a control that presets the flag reads it 30/30. The coherence
  point is the submission boundary. **`session_capture_token`'s `doorbell_wait_kernel` therefore has
  no working release path on this stack**, and the whole-token graph should not be ported.
- Two hazards recorded for any port: a SPIR-V kernel needing a `__global` pointer must use storage
  class 5 `CrossWorkgroup` (IGC rejects anything else) and `OpMemoryBarrier` makes IGC 2.36.3 abort;
  and a kernel launched with an unset pointer argument faults the GPU and loses the device
  (`ZE_RESULT_ERROR_DEVICE_LOST`, with `ccs` engine resets in the journal).
