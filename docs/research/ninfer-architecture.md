# NInfer engine architecture — deep-dive for bongo

**Task:** [BAS-64](/BAS/issues/BAS-64) (R2), child of [BAS-62](/BAS/issues/BAS-62). **Status:** complete, 2026-09-28.
**Author:** Researcher. **Branch:** `BAS-62-improve-speed-architecture`.

**Upstream read (read-only):** [Neroued/ninfer](https://github.com/Neroued/ninfer) at commit
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d` ("docs: align linear guidance and refresh q4 performance
report", 2026-09-26). Cloned shallow, `GIT_LFS_SKIP_SMUDGE=1`, 75 MB, into the run scratch dir. No
upstream code is vendored into bongo and no weights were downloaded. Raw inventory of what was read is in
[`bench/results/2026-09-28-ninfer-source-survey/op-inventory.txt`](../../bench/results/2026-09-28-ninfer-source-survey/op-inventory.txt).

**Method / trust boundary.** Every claim below is traced to a repo path. I read the full `docs/` tree
(including the Chinese-language maintainer references, which are the authoritative design documents), the
semantic Op contract headers under `include/ninfer/ops/`, the `src/ops/` file inventory, and targeted
planner/graph source. I did **not** read the `.cu` kernel bodies line by line, and I did not read
`src/models/qwen3_5/config.cpp` in full; claims about kernel *internals* are therefore marked as
"doc-level" — the design is documented, the arithmetic is not independently verified here. Where I
recompute a number from documented constants I label it **inference** and state the assumption.

**Read the CTO's recon first:** [`strata-ninfer-recon.md`](strata-ninfer-recon.md) §3. This document is the
detail behind that section, plus the take/not-take verdict.

---

## 0. What NInfer actually is (and what it is not)

NInfer is a from-scratch C++/CUDA engine for Qwen3.5-family Dense and MoE models on exactly one RTX 5090,
one resident model, 1–8 concurrent requests, one shared KV pool, from a custom `.ninfer` v3 artifact
(`README.md` §Capabilities and limits). Its published artifacts are **Qwen3.6/3.8-27B Dense** and
**Qwen3.6-35B-A3B MoE** (`README.md` table) — *not* the 125B `Qwen3.8-Flash-Next` that bongo targets.

That gap matters for reading this document. NInfer is the same **architecture family** as bongo
(`qwen3_5_text` / `qwen3_5_moe_text`: pre-norm residual, gated GQA on some layers, Gated DeltaNet (GDN)
linear attention on the rest, interleaved partial MRoPE, routed SwiGLU MoE with a gated shared expert —
`docs/maintainer/qwen3_5-model.md`), and it is bongo's **per-request quality/perf ceiling** because it
never offloads a weight. It is *not* a tiered-memory engine:

- the product boundary explicitly excludes "weight offload, multi-GPU, ... preemption, priority/QoS,
  active-request swapping" (`README.md` §Capabilities and limits);
- its planner sizes a *shared, fully-resident* capacity vector, not an offload schedule
  (`docs/maintainer/engine-architecture.md` §3.4, `docs/maintainer/resource-scheduling-and-context-cache.md` §3.1).

So: **NInfer's planners and formats are portable design; its kernels are the ceiling that bongo's kernels
must be measured as a fraction of; its memory-tier logic is reference material for KV/state, not for weights.**

### 0.1 Published ceiling numbers (RTX 5090, `README.md` §Performance)

| Point | Value |
|---|---:|
| Qwen3.8-27B nvfp4, 7,680-token prefill | 8,340.4 tok/s |
| Qwen3.8-27B nvfp4, 260,096-token prefill | 2,203.1 tok/s |
| Qwen3.8-27B nvfp4, structured MTP3 decode | 219.8 tok/s |
| Qwen3.8-27B groupwise-int, structured MTP3 decode | 224.4 tok/s |
| Qwen3.6-35B-A3B groupwise-int, 7,680-token prefill | 17,705.4 tok/s |
| Concurrent MTP3, Qwen3.8-27B nvfp4: C=1 → C=8 | 143.8 → 766.6 tok/s (5.33×) |

Per-request acceptance and scenario spread are in `docs/performance/qwen3.8-27b.md`; reproduced in §2.6.

---

## 1. Artifact format: `.ninfer` v3

Authorities: `docs/maintainer/artifact-container.md` (container contract),
`docs/maintainer/tensor-formats.md` (numeric codecs), `docs/maintainer/storage-layouts.md` (byte layouts),
`docs/weight-conversion.md` (recipes), `src/artifact/` (implementation).

### 1.1 What a v3 artifact carries

A v3 artifact is one entry file plus optional continuation volumes
(`example.ninfer`, `example.ninfer.part-0001`, ...). The entry has a 32-byte header — magic
`4e 49 4e 46 45 52 00 03` (`NINFER\0\3`), `json_bytes`, and a 16-byte `artifact_id`; continuation
volumes use magic `NINPRT\0\3` plus `part_index` and the same `artifact_id`. `entry_payload_start` is
`align_up(32 + json_bytes, 4096)`; the default single-file limit is **32,000,000,000 bytes (32 GB decimal)**,
and each shard's logical payload start is kept 4 KiB-aligned for direct I/O
(`artifact-container.md` §2, §3).

The JSON directory has five required roots plus two optional ones:

| Root | Meaning |
|---|---|
| `components` | actual model components. `text` required; `vision`, `mtp`, `dflash`, `dflash2` optional. Each has `config`, optional `target`, optional `resources` (role → resource object ID), optional `proposal` (text only: `domain: full|indexed`, `rows: Ns`) |
| `objects` | physical payloads, offset-ordered. `tensor` = `{id, kind, shape, format, layout, offset, bytes}`; `resource` = `{id, kind, encoding: raw_bytes_v1, offset, bytes}` |
| `bindings` | **logical parameter name → Binding**. A Binding is either a whole object (`{object: ID}`, shape must equal the parameter) or an ordered `parts` list of `{object, range:[begin,end)}` element ranges |
| `uses` | array of `{parameter, input, activation_policy, auxiliaries}` — one per `(parameter, input)` pair |
| `files` | non-empty file directory `[{path, payload_bytes}]`, entry first |
| `metadata` | open object; optional non-empty `name` |
| `provenance` | open object; source checkpoint/revision, component training pairing, encoder/recipe notes |

The semantic separation is the point: **a physical parent object need not equal a logical parameter.**
`bindings.parts` expresses "the source stores each head's Q rows next to that head's gate rows" and
recovers four logical projections out of one or two physical parents (`qwen3_5-model.md` §Logical
parameters and physical bindings). The logical element sequence excludes layout padding, and scale planes
are not weight elements (`artifact-container.md` §7.1). Objects are deduplicated by ID, so a frozen and a
trainable view of the same bytes materialize once (§5.3).

`uses.activation_policy` is a closed enumeration — `A16Only = {A16}`, `AllowA8 = {A16, A8}`,
`AllowA4 = {A16, A8, A4}` — and `auxiliaries` binds per-use scalars such as
`activation_input_divisor` (§8, §12.3). The runtime takes the **intersection** of the policies of all
fused uses.

Registration is explicit and closed:

| `format` | numeric meaning |
|---|---|
| `bf16`, `fp32`, `int32` | raw words |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16` | signed 4/5/6-bit codes, group 64, FP16 scale |
| `q8_g32_fp16` | codes in `[-127,127]`, group 32, FP16 scale |
| `nvfp4` | E2M1 codes, G16 E4M3FN block scale, FP32 weight divisor |
| `fp8_e4m3fn_row_bf16` | E4M3FN codes, one BF16 row scale |

and layouts are `contiguous_le_v1`, `row_split_k128_v1` (q4/q5/q6/q8 rank-2), `block_scale_k16_m128x4_v1`
(nvfp4), `row_scale_v1` (fp8), `raw_bytes_v1` (resources); tensor objects are 256-byte aligned, resources
1-byte (`artifact-container.md` §6). Layouts own the planes, padding and encoded-size formula; the Op
receives typed pointers, not the container's names (§5.1, §7.3).

The container is one product format: `.ninfer` "is the only C++ product artifact, with no extension
detection, compatibility shim or second product entry" (`engine-architecture.md` §10).

### 1.2 What a bongo-owned equivalent should carry, versus GGUF

GGUF is a name→(quant type, bytes) tensor soup plus a metadata KV block. It carries no logical parameter
indirection, no per-use activation permission, no capability declaration, and no role-based resource map.
`.ninfer` v3 carries all four. The delta that matters for bongo:

| Capability | GGUF | `.ninfer` v3 | What bongo should do |
|---|---|---|---|
| Physical parent ≠ logical parameter | absent; tensor name *is* the semantic name | `bindings.parts` over element ranges | **Take the idea.** A sidecar manifest can record, per GGUF tensor, which logical parameter(s) it serves and which row ranges belong to each — without re-encoding any weight |
| Per-use activation permission | absent | `uses.activation_policy` (A16/A8/A4) | **Take.** The missing knob that lets an MMQ-style path be declared legal per use instead of inferred per tensor |
| Closed capability surface | runtime tries every tensor | reader/binder reject unsupported; failure at Op prep/warmup | **Take the discipline**, not the registry: a bongo manifest should declare exactly which `(quant type, shape)` paths the engine implements |
| Frontend resources as data | tokenizer/chat template in metadata KV | `resource` objects + role map | Optional. bongo inherits llama.cpp's resource handling; a manifest role map is only worth it if bongo ships its own frontend |
| Sharding for 32 GB+ payloads | single file | entry + numbered parts, 4 KiB-aligned, `artifact_id`-checked | **Take for the sidecar**, but bongo's weights stay in GGUF |
| Provenance / conversion recipe | `general.*` metadata | open `provenance` + recipe document | **Already covered** by `bench/results/` + ADR discipline |

**Verdict:** do *not* build a bongo weight container. bongo's weights are published GGUF tiers
(`Q2_0` / `IQ2_XS` / `IQ3_XXS`), bongo does not own conversion, and re-encoding ~80 GB is out of scope.
Build the **narrow slice** of v3 that is cheap and high-leverage: a small JSON manifest (tensor → logical
role, fused-parent grouping, per-use activation policy, engine capability list) emitted by an offline tool
that header-range-reads the GGUF the way `tools/gguf-inventory.py` already does. That buys the
binding/`uses`/capability discipline, keeps the tiers intact, and stays read-only on GGUF. This is a
**design-only** item; it is not required before Stage 2 kernel work.

---

## 2. Speculation: MTP, DFlash, DFlash2, and the suffix drafter

Authorities: `docs/maintainer/dflash.md`, `docs/maintainer/qwen3_5-model.md` §Prefill, decode and MTP,
`docs/maintainer/replayssm-gdn.md`, `include/ninfer/ops/speculative_round.h`, `mtp_pack.h`, `mtp_round.h`,
`candidate_selector.h`, `src/models/qwen3_5/program/speculative/`.

### 2.1 MTP: one trained predictor layer, draft window 1–5

MTP is a **one-layer** predictor, not a second model. Conditioned on the final-normalized target hidden
`h_t` and the next token `x_(t+1)` (`qwen3_5-model.md`):

```text
e = offset_rmsnorm(composed_embedding(x_(t+1)), pre_fc_norm_embedding)
h = offset_rmsnorm(h_t, pre_fc_norm_hidden)
u = input_projection(concat(e, h))
u = one_gated_attention_block_with_target_ffn(u)
draft_hidden = offset_rmsnorm(u, mtp_final_norm)
draft_logits = target_output_head(draft_hidden)     # shares Text's head
```

The stem is `[embedding | hidden]` in that order (`mtp_pack.h`, `mtp_pack_fc_input`);
`mtp_split_attn_in` splits the fused input-projection output into Q/K/Gate/V rows by fixed offsets. MTP
has private stem/block/final-norm weights and private KV, and shares Text's attention/RoPE/FFN geometry
and embedding/output semantics. It proposes **recursively** (the previous MTP hidden feeds the next
proposal). The draft window is **1–5, a runtime choice, not a weight dimension** (`README.md`;
`mtp_round.h` requires `1<=K<=5`).

MTP prefill alignment is the subtle part: tokens shift by one, hidden states and positions do not
(`mtp_alignment.h`):

```text
token inputs    = [x_1, x_2, ..., x_n, x_(n+1)]
hidden inputs   = [h_0, h_1, ..., h_(n-1), h_n]
position inputs = [p_0, p_1, ..., p_(n-1), p_n]
```

### 2.2 The verify/accept structure (head-agnostic — this is the reusable part)

`speculative_round.h` defines a prepare op and an accept op around one target forward:

1. `speculative_prepare_verify_inputs`: for row `b`, `verify_ids[0]=anchor`, `verify_ids[j]=drafts[j-1]`
   for `0<j<=Pcur[b]`, and `positions[j] = base_positions[b] + min(j, Pcur[b])` with `Pcur[b]` in `[0,K]`.
2. The target runs one forward over `T = P+1` columns through the *normal* target graph (attention + GDN).
3. `speculative_accept_greedy_drafts`: per row, accept the longest available draft prefix matching the
   per-column penalty-adjusted argmax and commit that argmax at the first mismatch; accept-everything
   commits the bonus column. Sampling mode uses the proposal distribution `q`, accepts draft `i` with
   probability `min(1, p_i(draft_i)/q_i(draft_i))`, and on first rejection samples the residual
   `r_i(v) = max(p_i(v) - q_i(v), 0)`; all-accept samples the bonus from `p_P`.

This is exactly the shape a suffix/ngram drafter needs: the Op takes a proposal *distribution* per
position, and a deterministic suffix lookup is the point-mass case, which is already a supported case —
DFlash proposals are one-hot. The unit of reuse is
`prepare_verify_inputs → target forward → accept → commit`, not any particular drafter.

### 2.3 Recurrent-state handling under speculation: raw-input replay (ReplaySSM)

GDN's persistent state cannot be rolled back to an arbitrary earlier position cheaply. NInfer verifies
without saving a state trajectory: verify **records the raw inputs** that drove each transition, and after
the accepted length is known a single **Fold** replays the accepted prefix from the committed checkpoint
(`gdn_replay.h`, `src/ops/linear_attention/gated_delta_net/replay.cpp`, `replayssm-gdn.md`).

The state sizes make the motivation concrete (`replayssm-gdn.md` §1.1):

| Model | GDN layers | value heads | one recurrent state image |
|---|---:|---:|---:|
| Qwen3.6/3.8-27B | 48 | 48 | 144 MiB |
| Qwen3.6-35B-A3B | 30 | 32 | 60 MiB |

Snapshot-per-verify-column would cost one full state image per extra column. The doc's hard requirement is
that Fold must execute the **same finite-precision transition** as verify recurrence, not a
mathematically-equivalent form: GDN recurrence has several algebraically equal writings that differ in
normalization, reduction association, FMA grouping, Tensor-Core precision and cast boundaries, and the
state is carried across rounds, so a small reconstruction error propagates. Hence ReplaySSM's acceptance
test is a **bitwise clone** of the snapshot, not an FP64 reference (`op-development.md` §6.1).

### 2.4 DFlash / DFlash2: masked-block drafting from target residual features

Both drafters propose `K` tokens with **one** masked-block forward conditioned on committed target
**residual features** — not on the MTP stem (`dflash.md`). For processed position `t`, target block `l`
in `target_layer_ids`:

```text
s_t = concat(r_t^l) ;  c_t = plain_rmsnorm(W_feature s_t, context_norm)
k_ctx^l(t) = rope_1d(plain_head_rmsnorm(W_context_key^l c_t, key_norm^l), position=t)
v_ctx^l(t) = W_context_value^l c_t
```

Context prefill computes only the feature projection and these K/V projections; prompt tokens do **not**
go through the draft residual/MLP stack. Context uses absolute scalar Text cache positions and full-head
split-half RoPE, not the target's partial three-axis MRoPE.

The query block is `W=K+1` columns: the anchor at absolute `F`, then `K` mask tokens
(`prepare_masked_block.h`). Draft attention is **non-causal** over context + query block, with a local
layer using `allowed(p_q,p_k) = |p_k - p_q| < S` (endpoints at `S-1` included, `S` excluded). Consequently
"a shorter block is not required to match the prefix of a longer block."

Proposal heads differ:

- **DFlash**: per-column argmax over the full Text head or an indexed proposal head (`domain: indexed`,
  `rows: Ns` with a token-ID row map). Its `q` is a point mass even when the target samples.
- **DFlash2**: a **conditional left-to-right walk** over a top-16 unary shortlist with rank-256
  predecessor/successor codebooks (`candidate_selector.h`):

```text
E_i[p,c] = u_i[c] + sum_r W_pred[pred_token(i,p),r] * g_i[r] * W_succ[C_i[c],r]
pred_token(i,p) = anchor if i=0 else C_(i-1)[p]
temperature <= 0: j_i = lowest-rank argmax_c E_i[j_(i-1),c]      (q_i one-hot)
temperature > 0 : q_i = softmax_c(E_i[j_(i-1),c]/temperature); j_i ~ q_i
```

  The retained FP32 `q_i` is what the target-correction path consumes. DFlash2 ignores target
  top-k/top-p/min-p and uses its own RNG domain (`F+i` plus a distinct proposal purpose).

Current geometries (`dflash.md` table), as the family's parameter range:

| | 35B-A3B DFlash | 27B DFlash2 |
|---|---:|---:|
| Hidden width / draft layers / intermediate | 2048 / 6 / 6144 | 5120 / 5 / 17408 |
| Q/KV heads / head dim | 32 / 8 / 128 | 32 / 8 / 128 |
| target block IDs | `[1,6,11,16,22,27,32,37]` | `[5,19,33,47,61]` |
| attention pattern | 5 local + 1 full | 5 local |
| sliding window S | 4096 | 2048 |
| dynamic grouped conv | — | 2 taps, group 16 |
| selector | per-position argmax | top-16 conditional path, rank 256 |

Startup draft count is `K ∈ 1..15`, concurrency `B ∈ 1..8`, physical block `W=K+1`.

### 2.5 Live widths and state alignment (the failure mode to copy the fix from)

The two backends differ near the request tail (`dflash.md` §Live widths):

| Extent | DFlash | DFlash2 |
|---|---|---|
| Proposal attention / masked input | `Q=P+1` live columns | all `W=K+1` columns |
| Target attention and GDN | `Q=P+1` live columns | `Q=P+1` live columns |
| Candidate prefix checked | first `P` proposals | first `P` with their actual `q` |

Commit is a transaction: if acceptance yields `A` drafts and licenses `y=[d_1..d_A, correction|bonus]`
(`L=A+1`), the Frontend preview chooses a final prefix `N ∈ [0,L]`; then the transaction commits `N`
target input rows `[a, d_1..d_{N-1}]`, publishes `N` outputs, takes continuation hidden from verify column
`N-1`, and advances target KV, GDN Fold, counters and frontier **together**. Pending draft-context features
may lag the target frontier by up to `N` and must be materialized before the next proposal or before any
checkpoint/retain/Host replica. `N=0` commits nothing. DFlash2's ring state is
`5 layers × (2+2) bytes × 8 heads × 128 × 2048 = 40 MiB` per state image, independent of `K`.

### 2.6 Measured acceptance (RTX 5090, `docs/performance/qwen3.8-27b.md`)

Qwen3.8-27B, K=3 MTP and K=7 DFlash2, per-request phase statistics from the C=1 corpus points:

| Workload category | MTP3 acceptance | MTP3 tok/round | DFlash2 acceptance | DFlash2 tok/round |
|---|---:|---:|---:|---:|
| Long reasoning `aime26_01` (nvfp4) | 76.0% | 3.28 | 67.2% | 5.70 |
| Long reasoning `aime26_15` (nvfp4) | 56.2% | 2.69 | 35.2% | 3.47 |
| Long reasoning `aime26_30` (nvfp4) | 64.6% | 2.94 | 38.6% | 3.70 |
| Code | 76.4% | 3.29 | 53.9% | 4.77 |
| Story | 37.4% | 2.12 | 16.8% | 2.17 |
| Translation | 75.0% | 3.25 | 51.1% | 4.58 |
| Structured | 90.8% | 3.72 | 78.1% | 6.46 |

Two lessons for bongo. First, **acceptance is workload-dominated, not drafter-dominated**: the same MTP3
head ranges 37%→91% across categories. Second, a wider drafter does not dominate: DFlash2 K=7 beats MTP3
K=3 on structured output (6.46 vs 3.72 tok/round) and ties it on narrative text (2.17 vs 2.12) at 2.3×
the draft cost. Any bongo speedup claim from speculation must therefore be reported per workload class
with its acceptance and tokens/round, not as a single multiplier.

### 2.7 Is any of it usable without a dedicated head?

| Mechanism | Needs trained companion weights? | Usable by bongo? |
|---|---|---|
| MTP predictor | yes (`mtp` component: stem/block/final-norm) | **No.** The published GGUF drops the MTP head and llama.cpp `qwen4exp` cannot convert or run it (`CONTEXT.md`). `mtp_pack`/`mtp_round` are MTP-geometry-specific |
| DFlash / DFlash2 | yes (`dflash`/`dflash2`: draft layers, block IDs, conv, selector, codebooks) | **No** as trained heads |
| `speculative_prepare_verify_inputs` + `speculative_accept_*` + the commit transaction | **no** — they consume a proposal distribution | **Yes.** The head-agnostic core, and the most valuable speculation artifact in the repo for bongo |
| ReplaySSM record/Fold | no — a property of GDN verify | **Yes, and required.** bongo's 36 GDN layers have the same rollback problem |
| Verify-block bookkeeping (live widths, pending features, `N`-prefix commit) | no | **Yes**, as the correctness template for committing a speculative prefix atomically |

**Answer to the task's question: the verify/accept/commit machinery and ReplaySSM are usable and worth
taking; the drafters are not.** bongo's only proposal source remains the suffix/ngram drafter
(`strata-ninfer-recon.md` §2.4), whose "hit" is a point-mass proposal and therefore fits the DFlash-style
one-hot path exactly. The residual-correction path only becomes relevant if bongo ever samples with
temperature > 0 and wants exact-equivalence; then the suffix drafter must expose its proposal probability
as a point mass rather than assuming the argmax.

---

## 3. KV and state tiers: Device/Host checkpoints, the KV planner, and eviction

Authorities: `docs/maintainer/resource-scheduling-and-context-cache.md` (the algorithm),
`docs/maintainer/paged-kv-cache.md` (physical store), `docs/maintainer/engine-architecture.md` §4/§6,
`src/runtime/engine/context_cache/`, `src/core/paged_kv_cache.*`, `src/core/host_kv_arena.*`.

### 3.1 The two ownership rules that make the rest work

1. **Scheduler picks the request; the resource layer only optimizes that request's materialization.**
   Resource conditions cannot reorder the FIFO. A verified cache hit cannot jump the queue.
2. **Once a request is Active, its maximum legal execution range is fully guaranteed.** Inactive cache
   policy cannot borrow that capacity. Backfill is allowed only with a *persistent-safe proof* that does
   not use "the borrower will probably finish first" timing assumptions
   (`engine-architecture.md` §5.2, `resource-scheduling-and-context-cache.md` §6.3).

The correctness fallback is explicit and worth copying verbatim: if no cached target is feasible, the
planner must fall back to `root candidate + release all unprotected inactive cache`. A heuristic, budget or
wall-clock failure must never turn a runnable request into a blocked one (§8.6). For bongo that is the
difference between "the expert cache is a performance feature" and "the expert cache can deadlock the
server".

### 3.2 "Prefix hit" is a strong predicate

A reusable frontier requires **all** of (`resource-scheduling-and-context-cache.md` §4.1):

1. a complete StateImage at that frontier;
2. Main KV at the target-typed coverage;
3. the selected backend's KV and fixed state on the same continuation;
4. token, position, Vision and mode identity exactly equal to the incoming prompt.

If only the tokens match, or only KV bytes match, or only pages match, that is **not a partial hit** — it
is a miss. A valid checkpoint may still be Device-*unready* (it needs an H2D restore). Exact identity uses
token IDs and types, position/MRoPE axes and `rope_delta`, Vision spans and media digest, template/runtime
mode, and the checkpoint frontier. Session keys, markers, hashes and prefix indexes only shrink the
candidate set; they never prove a hit.

`rewrite_execution_frontiers` is part of identity: the exact token frontiers where replay/root prefill must
be split, so the reconstruction path uses the same execution decomposition as the original generation. A
Frontend-detected "model-output reconstruction boundary" is committed atomically with the resident prefix
identity, and it only fires when the canonical boundary lands exactly on a token frontier. Practical
consequence: NInfer's own unmodified output can continue from an exact endpoint; a client that reorders
tool JSON, fills defaults, or rewrites history is a *different rendered identity* and can only match an
earlier legal checkpoint (§4.5).

### 3.3 Checkpoint kinds — a direct fit for bongo's agentic/chat workloads

| Kind | Semantics |
|---|---|
| `SessionEndpoint` | latest continuable state of a finished request |
| `TurnClosure` | stable state before the replaceable assistant suffix of the current turn |
| `ResponseReplay` | stable state before the generation opener, so the response can be regenerated |
| `LongAnchor` | an older long-context recovery point chosen by retention policy |
| `SharedStablePrefix` | immutable stable prefix shared by several histories |

`TurnClosure`/`ResponseReplay` must sit *before* the assistant suffix the next request may replace. That is
precisely the shape of a chat/agent turn: append a user message, or regenerate the tail, without losing the
stable system+history prefix. bongo should carry these five kinds, not a single "prefix cache".

### 3.4 The eviction decision is a cost comparison, not an LRU

The public value of a checkpoint `p` in portfolio state `S` is
`Saving(p,S) = max(0, Rebuild(p) - Recovery(p,S))`, where `Rebuild` is the canonical root-to-frontier
prefill cost and `Recovery` is the **minimum over supported restore/copy/interval-prefill recipes** priced
with the same immutable machine cost model used by this planning problem (§8.3). Then:

```text
EmpiricalValue(S)    = sum_q max_{p matches q} Saving(p,S)   # one demand counts its best checkpoint once
PublicValue(S)       = EmpiricalValue(S) + SharedCredit(S)
PrivateLoss_o(Sb,St) = w_o * max_p max(0, Saving(p,Sb) - Saving(p,St))
J(c,T)               = Now(c,T) + FutureLoss_c(T)
```

with per-owner retention priors `Disposable 1`, `RecentPrivate 4`, `LiveSession 16`. Taking the max per
demand stops nested prefixes (tools ⊂ instructions ⊂ full prompt) from being triple-counted; taking the max
per owner stops a still-live endpoint from hiding the loss of an earlier `TurnClosure` or `LongAnchor`.
Private retention is modelled as "each owner has one more unobserved future reuse". Shared publication is an
**optional investment**: emitted only on strictly positive `NetGain > 0` versus the private-only baseline
at the same frontier; ties, overflow, or unprovable gains keep the baseline (§8.3, §7.2).

The machine cost model itself is portable and simple (§8.2):

```text
Immediate      = ordered transfer phases + remaining Text/Vision prefill + required State/KV copy work
Transfer       = max(batch_ns + operations * operation_ns, bytes * ns_per_byte)
AttentionPairs = B*S + S(S+1)/2        # B = reused prefix tokens, S = remaining suffix tokens
```

Transfer coefficients are per hardware; prefill coefficients are selected by a `prefill_signature` derived
from hardware class + actual Text/Vision config, bindings and Uses (never from checkpoint names or weight
values). Missing calibration falls back to generic coefficients. Cost only *orders feasible* targets — it
can never mark an infeasible target feasible or change physical readiness (§8.1).

### 3.5 Physical store: what actually bounds bongo's 128K KV

Facts from `paged-kv-cache.md`:

- Growing KV uses startup-fixed homogeneous pools: Main Text, plus MTP or Draft-Full when the selected
  backend needs it. Speculative backends are mutually exclusive, so at most two growing pools.
- **Page size `P = 64` tokens** for every growing pool. Three independent granularities: allocation (64),
  valid frontier (1 token), reusable state (target-defined checkpoint). A page boundary is *not* an
  attention-mask or prefix-hit boundary.
- Main capacity `M` (physical page-groups) must satisfy `M ∈ [max(L,C), C·L]` with `L = ceil(S/P)` for
  per-sequence ceiling `S` and concurrency `C`; an explicit `kv_capacity K_main` resolves to
  `M = ceil(K_main/P)` and requires `K_main >= S`. Automatic resolves once after weight load from the affine
  `B(M) = B_min + (M - M_min)·B_step` curve with a 1 GiB sizing headroom, and page rounding never raises
  the logical per-sequence ceiling.
- Per-token/head payload at `D=256` (K+V, code + scale):

| profile | K | V | K+V |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256+8 B | 256+8 B | 528 B |
| FP8-E4M3FN-row256 | 256+2 B | 256+2 B | 516 B |
| NVFP4-G16 | 128+16 B | 128+16 B | 288 B |
| K8V4 | 256+2 B | 128+16 B | 402 B |

- Two closed device plane orders: page-major `[X,P,H,N_physical]` for Main/MTP, head-major
  `[X,P,N_physical,H]` for DFlash Full. Consumers address through a per-sequence block table; kernel
  correctness may **not** assume adjacent logical pages map to adjacent physical IDs, and paging must not
  introduce a gather-to-contiguous staging copy proportional to context length (§10.4).
- Logical pages carry `content_epoch` + `committed columns`; a replica is valid for the first `n` columns
  iff `replica.epoch == page.epoch && replica.coverage >= n`. Speculative/uncommitted bytes never extend
  committed coverage.
- Non-aligned frontiers do not round down: with `P=64` and frontier `F=1000`, 15 full immutable pages are
  shared and the 40-column partial tail is COW'd into a private page. Same-pool pages have exactly one
  writer; shared full pages are immutable.
- Host replicas live in one startup-fixed pinned `HostKVArena` in logical order (no device page IDs, no
  block-table holes); Main and backend layouts can have different strides in the same arena. Replica
  replacement publishes only after copy + epoch/coverage verification; source and destination are both
  pinned during the copy.
- CUDA graphs: plane bases and the block-table matrix base are Engine-lifetime stable; only table content,
  row selectors, positions, context lengths, valid counts and state selectors vary per replay. Page IDs,
  request identity and physical contiguity are **not** part of the graph key.

**bongo cross-check (inference).** `CONTEXT.md` claims 128K attention KV is "~2-4 GB". Using the INT8-G64
row above and only the 12 full-attention layers: `12 layers × H_kv × 528 B/token`; at `H_kv = 4` that is
25,344 B/token ⇒ 3.32 GB at 131,072 tokens, and at `H_kv = 2` it is 1.66 GB. The claim holds for
`H_kv ∈ {2,4}` — the range NInfer's MoE (2) and Dense (4) geometries use. *Assumption: bongo's
full-attention layers use `head_dim = 256` with the same scheme; this was not verified against the GGUF.*
The conclusion is the important part: **KV is not bongo's binding constraint; expert bytes are.** The KV
planner is worth porting because it converts idle VRAM into reuse, not because it rescues the budget.

### 3.6 What the State tier costs (the asymmetry bongo should exploit)

NInfer's Device StateImage slots are `max_concurrency + device_state_slots`, and the first `C` slots form
the active guarantee; all slots come from one Program pool, not lane-paired
(`resource-scheduling-and-context-cache.md` §5.1). A 27B StateImage is ~144 MiB and a 35B-A3B one
~60 MiB. For bongo, a full continuation checkpoint is therefore **tiny compared to an expert bank** and
small compared to its KV: state snapshots are the cheap thing to keep resident, and KV pages are the thing
to demote. That ordering — pin state, tier KV, stream experts — is the specific inversion NInfer's tier
semantics suggest for bongo's box.

---

## 4. Kernels: op inventory and the strategies bongo should copy

Authorities: `include/ninfer/ops/*.h` (semantic contracts), `src/ops/` (implementations),
`docs/maintainer/op-development.md`, `docs/maintainer/linear-tuning.md`,
`docs/maintainer/storage-layouts.md` §8. Full listing:
[`op-inventory.txt`](../../bench/results/2026-09-28-ninfer-source-survey/op-inventory.txt).

### 4.1 Op architecture (the process rule worth copying)

An **Op** is a semantically closed, host-callable computation
`(outputs, new_state) = F(inputs, weights, old_state, semantic_parameters)`; workspace/stream/device are
execution resources that may select an implementation but cannot change the result
(`op-development.md` §2). Each Op has one authoritative contract comment with fixed fields
(Math/indexing, Logical shapes, Supported domain, Numeric, Effects, Workspace, Execution). The
implementation chain is `contract → wrapper (validation + finite dispatch) → launcher (private launch
policy) → kernel`, and the dependency direction is enforced: `core <- ops <- model execution <- runtime`.
Wrappers may not dispatch on model name, weight role, or Program phase (§4.1). Every Op is qualified
against an **independent naive FP32/FP64 oracle over the represented logical values**, and the oracle must
decode packed weights itself — pairing against another GPU route is not an oracle (§6.1).

Two consequences that matter for bongo: (1) an Op may keep a mainstream activation and a private
low-precision activation path behind a declared *policy*; (2) a fused Op's oracle evaluates the complete
fused formula rather than composing production Ops, which is what lets fusion change arithmetic without
re-deriving the reference.

### 4.2 Op families

| Family | Ops (`include/ninfer/ops/`) | Notes |
|---|---|---|
| Linear | `linear`, `linear_add`, `linear_pair`, `linear_swiglu`, `linear_topk`, `weight_input` | all read a quantized `Weight` view + BF16 activation; `linear_topk` fuses projection + top-k selection (router / expert selection) |
| Input projections | `attn_input_proj`, `gdn_input_proj`, `gdn_gating_proj`, `gdn_gating`, `causal_conv1d_silu` | fused Q/K/gate/V projections for GQA and GDN, with the depthwise causal conv + SiLU folded in |
| Recurrent linear attention | `gated_delta_net` (chunked + recurrent), `kimi_delta_attention`, `gdn_replay` | a chunked-vs-recurrent crossover exists, with `--force-chunked` / `--recurrent-only` benchmark controls (`op-development.md` §7) |
| Dense / softmax attention | `softmax_attention` (dense/causal-cache prompt + small-T, dense/context, dense/packed), `sliding_window_attention`, `kv_cache_append`, `context_kv_materialize`, `rmsnorm_rope` | separate prompt and decode paths, a separate small-T path, and per-KV-codec fused consumers |
| MoE | `sparse_moe` | single closed Op: router + selection + routed gate/up/down + shared expert + `AddResidual` epilogue |
| Drafters | `prepare_masked_block`, `prepare_ragged_prefix`, `candidate_selector`, `dynamic_grouped_conv`, `mtp_pack`, `mtp_round`, `speculative_round`, `target_logprobs` | DFlash2 conv + selector; speculative prepare/accept |
| Norm / activation / elementwise | `rmsnorm`, `rmsnorm_pack_tail`, `gated_rmsnorm`, `layer_norm`, `l2norm`, `gelu`, `silu_mul`, `sigmoid_mul`, `add_bias`, `residual_add`, `cast`, `scalar`, `scatter`, `embedding`, `position`, `argmax`, `sampling` | small, fusable |
| Vision | `vision_pos_embed` | out of bongo's scope |

### 4.3 The quantized-weight path (the actual gap)

`linear.h` is explicit: the public activation and output tensors are always BF16; weights are never
expanded. `LinearPolicy` gates the private activation profile — `A16Only`, `AllowA8`, `AllowA4` — and the
doc states plainly that "a permission does not require a corresponding low-precision route: the resolved
plan may remain A16 when that is the qualified choice." Every policy permits the existing A16
implementations of BF16 and Q4/Q5/Q6/Q8. FP8 resolves per registered shape and `T` (e.g. `[14336,5120]`
→ A16 through `T=11`, A8 from `T=12`); NVFP4 A4 is selected by the private resolver when the use permits
it. Registered execution domains are **finite and enumerated**: each `(format, N, K)` is a registered
problem, and "a valid encoding and alignment do not imply arbitrary N/K support."

`src/ops/linear/<fmt>/` shows the mechanism set per format: **A16 MMA, A16 SIMT, GEMV, sliced-K MMA** for
q4/q5/q6/q8, plus **A8/A4** and **TMA** variants for fp8/nvfp4 (and `q8_grouped_sliced_k` for
grouped/ragged work). `q4/q4_dispatch.h` exposes exactly two selection entry points:
`select_q4_a16_launch(n,k,t)` and `select_q4_launch(n,k,t,policy)`. That is the MMQ shape: **weights stay
in their stored encoding and are consumed by the contraction**, with the low-precision question asked
about the *activation*, not the weight.

`src/ops/linear/q4/shapes/` registers concrete `N×K` problems:
`n1024_k5120, n131072_k2048, n131072_k5120, n3456_k1152, n34816_k5120, n4096_k5120, n4304_k1152,
n5120_k6144, n6144_k5120, n7168_k5120` — plus analogous sets per format. This is a hand-curated tuning
matrix, not a generic GEMM.

`sparse_moe.h` is similarly closed: "Closed sparse-MoE Op for the exact future 35B-A3B geometry" —
2048 hidden, 256 routed experts, top-8, one always-on shared expert; routed banks admit Q4+Q5, Q4+Q6 or
Q8+Q8 and both shared banks are Q8; expert `e` "directly selects its stored row spans; no selected-weight
gather or repack occurs". Every positive `T` is supported. One cosmetic-scale hint exists:
`SparseMoeHints.next_weight_prefetch` issues fire-and-forget L2 prefetches over the named span for the
next decode step's weight consumer, with no numeric effect.

### 4.4 Strategies bongo should copy (in expected-value order)

1. **Keep weights quantized; make the activation profile an explicit per-use policy.** The largest lever,
   and the one the Strata recon already identified as the M2 gap. Concretely: bongo should consume
   `Q2_0`/`IQ2_XS`/`IQ3_XXS` codes directly in the matmul and expose `A16 | A8 | A4` as a *policy*, not a
   kernel choice. Do not build an `A16Only`-only path and call it done.
2. **A per-`T` mechanism set with a measured crossover, not one kernel family.** NInfer keeps SIMT/GEMV,
   MMA, and sliced-K MMA alive simultaneously and selects per `(N,K,T)`. bongo's decode (`T=1`) needs the
   GEMV/SIMT route; prefill needs the MMA route. Keep the dispatch documented per quant type.
3. **Fuse the epilogues the model actually chains.** `linear_swiglu`, `linear_add`, `linear_pair`,
   `linear_topk`, `attn_input_proj`, `gdn_input_proj`, `gdn_gating_proj`, `rmsnorm_pack_tail`,
   `causal_conv1d_silu`. Each deletes an intermediate write + read of a full-size activation tensor,
   which on a bandwidth-bound box is the whole cost.
4. **Fuse the KV *decode* into each attention variant.** `kv_cache_append` codecs are BF16, INT8-G64,
   FP8-E4M3FN-row256, NVFP4-G16 and K8V4, and `softmax_attention` has a per-codec consumer
   (`prompt_i8.cuh`, `small_t_i8.cuh`, `prompt_k8v4.cu`, ...). A quantized-KV tier is worthless if the
   attention op dequantizes the whole cache first.
5. **Asymmetric K/V profiles are legal and useful.** K8V4 pins K at FP8-row256 and V at NVFP4-G16 and
   reports 402 B/token/head versus 528 for INT8 and 1024 for BF16. Take the idea even if the codecs differ
   on Arc.
6. **Fuse the GDN projection + causal conv + SiLU** and keep chunked (prefill) and recurrent (decode)
   routes as separate qualified implementations with a measured crossover (`gdn_input_proj`,
   `gdn_projected_conv`).
7. **ReplaySSM-style raw-input record + Fold** if bongo adds speculation (see §2.7). Snapshotting GDN
   state per verify column does not scale.
8. **Fused router + expert + shared expert with a fused residual epilogue** (`sparse_moe`). For bongo this
   is the shape of the *resident* expert path; the streamed path needs a different decomposition, but the
   "router selects row spans, no repack" rule is the right one.
9. **Fused decode-step L2 prefetch hint** for the next weight span. Cheap, no numeric effect, and directly
   targets per-token weight-streaming latency. The closest thing in the repo to explicit
   prefetch-overlap control — and it is a *hint*, not a schedule.
10. **Declare supported domains and reject loudly.** `linear.h` and `sparse_moe.h` list their registered
    shapes. The failure mode bongo must avoid is silently falling back to a slow generic path; the failure
    mode to accept is refusing a configuration at startup.

### 4.5 What bongo must not copy

- The **closed shape registry** (`n1024_k5120`, ...; MoE fixed at 256 experts / top-8). bongo's model has
  different dimensions and 512 experts × 48 layers, and its expert banks are streamed, not resident. Copy
  the *pattern* (explicit capability list, no generic fallback), not the list.
- The **NVFP4 / FP8 / TMA / mbarrier / PDL kernels**. Blackwell-specific (`pdl.cuh`, the `sm_120a` build
  gate, the non-RDC NVFP4 translation units). Not portable to Battlemage.
- **`mtp_pack` / `mtp_round`** — MTP-geometry-specific index remaps for a head bongo does not have.
- The **Vision tower**.
- Any assumption that the whole model is resident (see §0).

---

## 5. Prefill/decode scheduling and the caps

Authorities: `docs/maintainer/engine-architecture.md` §1/§5/§8; `docs/maintainer/paged-kv-cache.md` §11;
`include/ninfer/types.h`; `src/models/qwen3_5/program/planning/`, `program/decode.cpp`, `prefill.cpp`,
`planning/graph_profiles.cpp`.

### 5.1 Execution model and the single mutation owner

```text
Gateway   -- protocol / transport ------------------▶ product
Frontend  -- PreparedPrompt / OutputSession --------▶ prompt + output semantics
Engine    -- order / lifecycle / publication -------▶ control plane
Program   -- physical resources / model execution --▶ execution
```

Exactly one Engine **worker** mutates request records, Scheduler, ResourceManager and Program. Ingress,
consumer and transport threads interact only through queues, cancellation flags and response events. The
worker's per-boundary order is fixed, and three orderings are stated as architecture, not implementation
detail:

1. an in-flight GPU unit must reach a stable boundary before its resource mappings change;
2. Program commit must precede user-visible output publication;
3. a resource result must be adopted before a lane may change visible state, and no second global resource
   topology transition may start while one is unsettled.

Request states: `Waiting → Materializing → Prefill → DecodeReady | ControlReady → TerminalPending →
Finished`. Lane states: `Free → Materializing → Active → TerminalPending → Free`. Control lane, StateImage
slot, KV execution row and decode-batch row are four different identities and must not be derived from one
another (§4.2).

### 5.2 Prefill

- **One staged-prefill request at a time**, and existing decode work may not be starved by consecutive
  prefill (`engine-architecture.md` §5.3). The concrete "don't let prefill monopolize the box" rule.
- Chunk size is `--prefill-chunk`, **default 1024**, must be a nonzero multiple of **128**, and is clamped
  to the context capacity at startup (`include/ninfer/types.h`: `prefill_chunk = 1024`;
  `planning/startup.cpp`: "prefill_chunk must be a nonzero multiple of 128"; `apps/cli/options.cpp`).
  Chunks per suffix is `1 + ceil((suffix-1)/chunk)` (`src/runtime/contract/resources.h`).
- The published serving profile uses chunk 1024 (`docs/performance/methodology.md` §Common serving
  profile), and the scheduler benchmark has explicit `--prefill-chunk 128` and `4096` points to measure
  the crossover (`tools/bench/ttft/README.md`).
- Prefix reuse only reduces materialization or suffix prefill; it does not create a second scheduling path
  (`engine-architecture.md` §5.3).
- Planner prefill cost is `AttentionPairs = B·S + S(S+1)/2` with `B` reused prefix tokens and `S` suffix
  tokens (§8.2) — a portable, cheap cost model that captures the quadratic term bongo's planner also needs.

### 5.3 Decode

- A decode round contains **all and only** the currently decode-ready requests, and uses the **exact** `B`;
  it is not padded to `max_concurrency` with inactive lanes (`engine-architecture.md` §5.3).
- **CUDA graphs are per exact-`B` topology.** `planning/graph_profiles.cpp` enumerates frontier ranges at
  measured split-policy transitions; for ordinary decode the ranges end at
  `{127, 511, 2047, 4095, 8197, 16389, 32767}` and the final range runs to `capacity-1` — about 8 graph
  profiles per batch size. The comments state that the early ranges limit empty producer CTAs and the later
  ones follow split-policy transitions. MTP profiles track `E+2K`; DFlash2 profiles use
  `{96, 511, 2047, 8191, 32767}`.
- The graph key excludes page IDs, request identity and physical contiguity; what varies per replay is
  table content, row selectors, positions, context lengths, valid counts and state selectors
  (`paged-kv-cache.md` §11). Addresses are Engine-lifetime stable.
- Ordinary decode does **not** run a catalog scan, pressure search or background replica scan
  (`engine-architecture.md` §8). Cache policy runs only at admission/capture/finish/inactive-release
  boundaries.
- `--max-context` is a per-sequence logical ceiling; `--kv-capacity` sizes the shared pool; `auto` resolves
  at startup from remaining memory minus a 1 GiB headroom and is then fixed for the process lifetime
  (`README.md`).
- Warmup runs the same public Engine path but with request-level caching forced off, and must leave no
  externally hittable continuation (`engine-architecture.md` §8).

### 5.4 Caps (as read; they will not transfer literally)

| Cap | Value | Source |
|---|---|---|
| Active requests | `1..8` (`kMaximumConcurrency = 8`) | `include/ninfer/types.h`; `README.md` |
| Prefill chunk | default 1024, multiple of 128, clamped to capacity | `types.h`, `planning/startup.cpp`, CLI |
| MTP draft window `K` | `1..5` | `README.md`, `mtp_round.h` |
| DFlash / DFlash2 draft count `K` | `1..15`; block `W=K+1` | `dflash.md` |
| KV page size `P` | 64 | `paged-kv-cache.md` §2, §4.4 |
| Main KV physical pages `M` | `max(L,C) ≤ M ≤ C·L`, `L = ceil(S/64)` | `paged-kv-cache.md` §3.2 |
| MTP extra pages | `C · ceil((K-1)/64)` | `paged-kv-cache.md` §3.4 |
| Explicit prompt cache markers | ≤ 4 | `types.h`; `resource-scheduling...md` §7.2 |
| Frontend-generated candidates | ≤ 3 (all-tools, leading system/developer, full prompt) | `resource-scheduling...md` §7.2 |
| Demand window | 32 recent materializations; shared credit expires after 32 | §8.3 |
| Vision | ≤131,072 raw patches / 32,768 merged tokens aggregate; ≤16,384 merged tokens per item | `qwen3_5-model.md` |
| Host KV | one startup-fixed pinned arena, packed bytes + allocator geometry | `paged-kv-cache.md` §3.5 |

### 5.5 What bongo should take from the schedule

- **One mutation owner with a fixed boundary order.** bongo's runtime must not have a cache thread and an
  inference thread racing over the same device state.
- **Exact-`B` decode rounds, no lane padding**, and an exact-`B` graph topology keyed on shape only.
- **Chunk size as an aligned, clamped startup parameter** (128-aligned, default in the 1024 range) with a
  measured crossover, not a hard-coded constant.
- **The scheduler gates**: at most one staged prefill; decode cannot be starved; a decode round is exactly
  the decode-ready set.
- **Persistent-safe backfill proof** instead of ETA heuristics.
- **The `B·S + S(S+1)/2` prefill cost model** for admission and reuse decisions.
- **Automatic capacity from an affine curve measured from the real layout**, not from probing allocations.

Portability caveat: the graph-profile boundaries are derived from a 5090 occupancy cap and its split-policy
transitions. bongo must re-derive its own boundaries from Arc/Xe2 occupancy; the *structure* (stable
addresses + exact-`B` topology + scalar-only updates) is the portable part. CUDA graph capture itself needs
a substitute (see §6, row 26).

---

## 6. Portability table

Legend — **NVIDIA-specific**: requires CUDA/Blackwell mechanisms absent on Arc, or depends on
NVFP4/TMA/mbarrier/PDL; **SYCL-portable**: host-side or algorithm-level, no CUDA dependency in the
mechanism; **design-only**: a scheduling/correctness idea that transfers regardless of backend, with no
code transfer; **via-substitute**: the design transfers but needs an Arc equivalent mechanism.

| # | Item | Repo path(s) | Class | Note |
|---|---|---|---|---|
| 1 | v3 container framing (32 B header, magic, JSON, 4 KiB alignment, 32 GB sharding, `artifact_id`) | `docs/maintainer/artifact-container.md` §2–3, `src/artifact/framing.h` | **SYCL-portable** | pure host-side bytes/JSON |
| 2 | `components` / `objects` / `bindings(parts)` / `uses(activation_policy)` / `files` directory model | same §4–8; `src/artifact/binder.*` | **SYCL-portable** | host resolution; the binding/uses idea is the reusable part (§1.2) |
| 3 | Codecs `q4_g64_fp16`, `q5`, `q6`, `q8_g32_fp16` | §6.1; `docs/maintainer/tensor-formats.md` | **SYCL-portable** | groupwise integer codecs decode with integer ops |
| 4 | `nvfp4` codec (E2M1 + G16 E4M3FN block scale + FP32 divisor) | §6.1; `src/ops/linear/nvfp4/` | **NVIDIA-specific** | Blackwell FP4 path; no Arc equivalent |
| 5 | `fp8_e4m3fn_row_bf16` codec + `row_scale_v1` layout | §6; `src/ops/linear/fp8/` | **via-substitute** | needs verification that Xe2 exposes a usable FP8 matmul path |
| 6 | `row_split_k128_v1` layout (code / high-bit / scale planes) | `docs/maintainer/storage-layouts.md` | **SYCL-portable** | the layout is a byte plan; the consumers are not |
| 7 | `linear` A16 MMA / SIMT / GEMV / sliced-K per-format dispatch | `linear.h`, `src/ops/linear/q4/q4_dispatch.h`, `q4_shapes.h` | **via-substitute** | the mechanism maps to XMX/int8 on Xe2; the kernels do not |
| 8 | `LinearPolicy` A16/A8/A4 as a per-use contract | `linear.h`, `artifact-container.md` §8 | **SYCL-portable** (as a design contract) | the highest-value portable item |
| 9 | Fused `linear_swiglu` / `linear_add` / `linear_pair` / `linear_topk` epilogues | `include/ninfer/ops/linear_*.h` | **via-substitute** | the fusion pattern transfers |
| 10 | Fused `attn_input_proj` / `gdn_input_proj` / `gdn_gating_proj` / `causal_conv1d_silu` | `include/ninfer/ops/` | **via-substitute** | same |
| 11 | `gated_delta_net` chunked + recurrent routes and their crossover | `src/ops/linear_attention/gated_delta_net/` | **design-only** for the crossover; kernels NVIDIA-specific | the GDN math is portable (`qwen3_5-model.md`) |
| 12 | `gdn_replay` raw-input record + Fold (ReplaySSM) | `include/ninfer/ops/gdn_replay.h`, `docs/maintainer/replayssm-gdn.md` | **design-only** | the algorithm and the bitwise-clone acceptance rule transfer |
| 13 | `sparse_moe` closed Op (router + routed + shared + AddResidual) | `sparse_moe.h` | **design-only** | geometry is closed to 256/top-8; the pattern transfers |
| 14 | `SparseMoeHints.next_weight_prefetch` (L2 prefetch hint) | `sparse_moe.h` | **NVIDIA-specific** as written; **design-only** as a pattern | Arc needs its own prefetch/cache mechanism |
| 15 | `softmax_attention` dense/prompt, small-T, context, packed; per-KV-codec consumers | `src/ops/softmax_attention/` | **via-substitute** | split by phase and by codec is the portable lesson |
| 16 | `sliding_window_attention` local-window masking | `include/ninfer/ops/sliding_window_attention.h` | **SYCL-portable** (algorithm) | |
| 17 | Paged KV store: `P=64`, page-major/head-major closed plane orders, block tables | `paged-kv-cache.md` §4; `src/core/paged_kv_cache.*` | **SYCL-portable** (host design) | the device kernels are not |
| 18 | Host KV arena + pinned replica transfer and publish-after-verify order | `paged-kv-cache.md` §5.3–5.4; `src/core/host_kv_arena.*` | **via-substitute** | pinned host memory is a CUDA API; the ordering rule is portable |
| 19 | Logical page `content_epoch` + committed-coverage validation | `paged-kv-cache.md` §5.1 | **SYCL-portable** | pure bookkeeping |
| 20 | Exact-prefix identity (`rewrite_execution_frontiers`, reconstruction boundary) | `resource-scheduling-and-context-cache.md` §4.5 | **SYCL-portable** | pure design; high value for bongo |
| 21 | Five checkpoint kinds (`SessionEndpoint` … `SharedStablePrefix`) | §4.3 | **SYCL-portable** | pure design |
| 22 | `Rebuild` vs `Recovery` saving, `PublicValue`, `PrivateLoss`, `J`, retention priors 1/4/16 | §8.3 | **SYCL-portable** | portable algorithm; bongo must calibrate its own cost coefficients |
| 23 | Ordered-stage peak feasibility (`reserve → copy → verify → publish → release`) | §3.4, §9.3 | **SYCL-portable** | pure design |
| 24 | Device/Host StateImage slots `C + device_state_slots`; Move / Fork / Freeze / Snapshot | §5.1 | **SYCL-portable** | pure design |
| 25 | Pressure planner, bounded search, stop reasons, root fallback | §8.6–8.9 | **SYCL-portable** | pure algorithm |
| 26 | CUDA Graph exact-`B` decode profiles and split-policy boundaries | `src/core/decode_graph.*`, `src/models/qwen3_5/program/planning/graph_profiles.cpp` | **via-substitute** | needs a SYCL command-graph / Level Zero equivalent; boundaries must be re-derived |
| 27 | Programmatic dependent launch | `src/core/pdl.cuh` | **NVIDIA-specific** | no Arc equivalent |
| 28 | TMA / `mbarrier` data movement | `src/ops/common/mbarrier.cuh`, `src/ops/common/mma.cuh` | **NVIDIA-specific** | |
| 29 | `sm_120a` build gate; non-RDC NVFP4 translation units | `CMakeLists.txt`, `docs/maintainer/build-system.md` | **NVIDIA-specific** | |
| 30 | Single-worker mutation owner + fixed boundary ordering | `engine-architecture.md` §5.1 | **SYCL-portable** | pure design |
| 31 | FIFO head / one-staged-prefill / exact-`B` decode-round gates | §5.2–5.3 | **SYCL-portable** | pure design |
| 32 | Persistent-safe backfill proof (no ETA assumptions) | §5.2; `resource-scheduling-and-context-cache.md` §6.3 | **SYCL-portable** | pure design |
| 33 | `AttentionPairs = B·S + S(S+1)/2` and `Transfer = max(...)` cost model | §8.2 | **SYCL-portable** | coefficients re-measured per hardware |
| 34 | Affine `SequenceCapacityCurve` auto-capacity with 1 GiB headroom | `paged-kv-cache.md` §3.3 | **design-only** | the affine form transfers; the constants do not |
| 35 | MTP predictor (stem/block/final-norm; `mtp_pack`, `mtp_round`) | `qwen3_5-model.md`, `include/ninfer/ops/mtp_*.h` | **NVIDIA-specific in practice** | needs the trained head, which the published GGUF lacks |
| 36 | DFlash / DFlash2 drafters (masked block, conv, selector, codebooks) | `dflash.md`, `candidate_selector.h` | **NVIDIA-specific in practice** | needs trained companion weights |
| 37 | `speculative_prepare_verify_inputs` / `speculative_accept_*` (head-agnostic) | `speculative_round.h` | **SYCL-portable** (algorithm) | the reusable speculation core |
| 38 | ReplaySSM bitwise-clone acceptance criterion | `op-development.md` §6.1; `replayssm-gdn.md` | **design-only** | an acceptance policy, not code |
| 39 | Op contract discipline (7-field contract; wrapper/launcher/kernel chain; one-way dependencies; independent oracle) | `op-development.md` §2–6 | **SYCL-portable** | a process rule; the most durable takeaway |
| 40 | Vision tower + media decode | `qwen3_5-model.md`, `src/media/` | **NVIDIA-specific / out of scope** | bongo has no multimodal requirement |
| 41 | Frontend compiled chat templates (not arbitrary Jinja) | `qwen3_5-model.md` §Vocabulary | **design-only** | bongo should keep Jinja for llama.cpp compatibility |
| 42 | OpenAI / Anthropic HTTP Gateway | `src/serve/` | **SYCL-portable** | host C++ |

---

## 7. What bongo should take, and what it should not

### 7.1 Take

1. **`LinearPolicy`-style per-use activation permission, with weights that stay encoded.** This is the
   identified gap. Declare `A16 | A8 | A4` per use, and implement at least one route that consumes
   `Q2_0`/`IQ2_XS`/`IQ3_XXS` codes directly instead of dequantizing to FP16.
2. **The Op contract discipline.** A 7-field contract per op, a wrapper/launcher/kernel split with one-way
   dependencies, and an independent FP32/FP64 oracle per op. Cheap process, and it is what makes the
   fusion in (3) safe to change.
3. **Epilogue and projection fusion.** `linear_swiglu`/`linear_add`/`linear_pair`/`linear_topk`,
   `attn_input_proj`/`gdn_input_proj`/`gdn_gating_proj`, `causal_conv1d_silu`, `rmsnorm_pack_tail`. On a
   bandwidth-bound box these are direct byte reductions and none of them need new math.
4. **A per-`T` mechanism set with a measured crossover** (GEMV/SIMT for decode, MMA for prefill, sliced-K
   in between), selected per `(quant type, N, K, T)`.
5. **The KV/state tier design** — complete-continuation identity, the five checkpoint kinds, `Rebuild` vs
   `Recovery` saving, retention priors, ordered-stage peaks, and the root-fallback guarantee. Reason: these
   are host-side algorithms with no CUDA dependency, and they convert idle VRAM into prefix reuse without
   touching the expert-bytes bottleneck.
6. **Pin state, tier KV, stream experts.** NInfer's StateImage (60–144 MiB) is much smaller than an expert
   bank, and bongo's 128K KV is ~1.7–3.3 GB (inference, §3.5) versus tens of GB of expert bytes. The cheap
   thing to keep resident is state; the expensive thing is experts.
7. **One mutation owner + the fixed boundary order**, exact-`B` decode rounds, the one-staged-prefill gate,
   and the persistent-safe backfill proof.
8. **The speculation *verify* core, not the drafters.** `prepare_verify_inputs → target forward →
   accept-longest-greedy-prefix → atomic prefix commit`, with a point-mass proposal from bongo's
   suffix/ngram drafter, plus ReplaySSM raw-input record/Fold for the 36 GDN layers. Report speedup per
   workload class with acceptance and tokens/round.
9. **The `B·S + S(S+1)/2` prefill cost model** and the `Transfer = max(...)` transfer model as the
   planner's cost vocabulary.
10. **Startup-fixed capacity derived from the real wired layout**, with an explicit readiness enum
    (`PermanentlyInfeasible` / `TemporarilyBlocked` / `NeedsTransfer` / `Ready`) so admission can tell
    "never" from "not now".

### 7.2 Do not take

1. **A bongo weight container.** Re-encoding ~80 GB of published GGUF tiers is out of scope and would
   break compatibility with llama.cpp and with the tier workflow. A manifest sidecar is the cheap slice.
2. **The closed shape registry and the fixed sparse-MoE geometry.** bongo's expert banks are streamed,
   not resident, and its dimensions differ. Copy the capability-declaration *pattern*.
3. **NVFP4 / FP8 / TMA / mbarrier / PDL device code and the `sm_120a` build gate.** Not portable.
4. **CUDA Graph capture as the only launch-overhead answer.** Take the design (stable addresses, exact-`B`
   topology, scalar-only per-replay updates) and find the Arc mechanism; do not port the boundaries.
5. **The MTP and DFlash/DFlash2 drafters.** The trained heads are absent from the published GGUF.
6. **The "no weight offload" product boundary.** It is NInfer's ceiling definition, not bongo's design.
   NInfer's planner never has to answer the question that is bongo's entire premise.
7. **Compiled chat templates.** bongo should stay on llama.cpp-compatible Jinja rendering.
8. **The Vision tower and media pipeline.**
9. **The 1 GiB headroom constant, the 1..8 concurrency cap, the 1..5 / 1..15 draft caps, and the
   `{127, 511, 2047, ...}` graph boundaries.** These are 5090- and product-specific numbers. The
   *formulas and enum shapes* transfer; the constants must be re-measured on the reference box.

### 7.3 Verdict

**The measured gap matters, and NInfer's contribution to closing it is mostly *shape*, not code.** The
portable, high-value items are (a) the per-use activation policy with weights consumed in their stored
encoding, (b) the epilogue/projection fusion set, (c) the complete-continuation checkpoint identity plus
the `Rebuild`-vs-`Recovery` eviction rule, and (d) the head-agnostic speculative verify/commit structure
with ReplaySSM for recurrent state. The NVIDIA-specific parts — NVFP4, TMA, PDL, the 5090-tuned kernels
and graph boundaries — are boundary markers for how far the Arc port can be pushed, not a plan.

---

## 8. Residual uncertainty and next experiments

### 8.1 Residual uncertainty (what this document does not settle)

- **Kernel-level performance is not verified here.** The design of each Op is documented and its contract
  is explicit; the achieved fraction of Arc's roofline is unknown and must be measured.
- **bongo's exact geometry is assumed, not read.** `head_dim`, `H_kv` for the 12 full-attention layers,
  and GDN value-head counts come from the batch family (2/4 KV heads, 256 head dim) and were not verified
  against the Swift-1.5 GGUF. The 128K KV estimate in §3.5 is an inference with that assumption stated.
- **Whether Xe2/Battlemage exposes a usable FP8 or int8 MMA path for the A8/A4 policy** is unverified.
  This is the single question that decides how much of the `LinearPolicy` design bongo can realize.
- **The publication's acceptance figures are for Qwen3.8-27B on a 5090** and are not bongo's numbers.
  They establish that acceptance is workload-dominated, which is the transferable part.
- **NInfer's own expert-cache analogue does not exist.** NInfer never offloads; there is no evidence in
  this repo about hot-expert residency, per-token streaming, or SSD-resident tables. That evidence must
  come from R4/R5 and from Strata.

### 8.2 Next experiments (with owners)

1. **R6 (Intel/SYCL feasibility, existing task): test the A8/A4 question directly.** Does Xe2 expose
   int8/FP8 matmul for a `Q2_0`/`IQ2_XS`-weight contraction without expanding weights, and what is the
   measured `T=1` and `T=1024` throughput versus the A16 path? That answer decides the kernel plan.
2. **R6: test the graph substitute.** Does a SYCL command-graph / Level Zero command-list path give
   per-replay scalar-only updates for a fixed-`B` decode, and how many profiles are needed before the
   launch overhead is hidden? Re-derive boundaries; do not reuse `{127, 511, ...}`.
3. **R5 (SSD/PLE shard, existing task): confirm the overlap budget** for the 16-row-per-token PLE read
   against the measured decode window, because NInfer's `next_weight_prefetch` hint is the only
   prefetch mechanism in this repo and it is an L2 hint, not an SSD scheduler.
4. **New (this doc's recommendation, for the synthesis task S1 to sequence): a bongo checkpoint-identity
   prototype.** Implement only the identity predicate (complete continuation + exact token/position/media
   identity + `rewrite_execution_frontiers`) and measure its false-hit rate against a naive
   longest-common-prefix cache on the reference box, using the existing harness. Rationale: every other
   reuse mechanism is worthless if a "hit" is not a real hit, and this is pure host-side work with no
   kernel dependency. If this shows no benefit, the rest of the tier work should be dropped.

No new child issues are filed from this task: R4/R5/R6 already exist as siblings under
[BAS-62](/BAS/issues/BAS-62), and the two new proposals above are inputs to the synthesis task S1, not
independent parallel work.

---

## 9. Reproduction

```sh
# read-only upstream survey; no weights, no multi-GB download
export S="${PAPERCLIP_RUN_SCRATCH_DIR:-/tmp}/ninfer-survey"
mkdir -p "$S" && cd "$S"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Neroued/ninfer.git ninfer
cd ninfer && git rev-parse HEAD
# expect e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d

# the inventory file committed with this doc:
#   bench/results/2026-09-28-ninfer-source-survey/op-inventory.txt
```

Source-of-record files for each section: §1 `docs/maintainer/artifact-container.md`,
`docs/weight-conversion.md`, `src/artifact/`; §2 `docs/maintainer/dflash.md`,
`docs/maintainer/qwen3_5-model.md`, `docs/maintainer/replayssm-gdn.md`,
`include/ninfer/ops/speculative_round.h`, `docs/performance/qwen3.8-27b.md`; §3
`docs/maintainer/resource-scheduling-and-context-cache.md`, `docs/maintainer/paged-kv-cache.md`,
`docs/maintainer/engine-architecture.md`; §4 `include/ninfer/ops/`, `src/ops/`,
`docs/maintainer/op-development.md`; §5 `docs/maintainer/engine-architecture.md`,
`src/models/qwen3_5/program/planning/graph_profiles.cpp`, `include/ninfer/types.h`.
