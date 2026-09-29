# M3.2b — n-gram self-speculation A/B on the shipped Vulkan config

Issue: [BAS-131](/BAS/issues/BAS-131) (child of [BAS-62](/BAS/issues/BAS-62)).
Raw files: the `*.json` beside this README. Runner: `bench/run-speculation-m32b.sh`.

## Verdict

**Negative. Do not enable n-gram self-speculation in the shipped configuration.**
It never reaches the 1.3x decode gate, and it *regresses* decode when acceptance
is low. No `bongo.sh` change is made.

## Identity (both legs)

| field | value |
| --- | --- |
| engine | llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`) |
| backend | Vulkan, `--device Vulkan1` |
| tier | `iq2_xs` |
| KV | `--cache-type-k q8_0 --cache-type-v q8_0` |
| Stage 0 baseline | `--n-gpu-layers 99 --n-cpu-moe 16 --parallel 1 --flash-attn on --ctx-size 131072` |
| spec leg only | `--spec-type ngram-map-k4v` (engine defaults: `size-n`, `size-m`, `min-hits`) |
| host | Intel Arc Pro B70 32 GiB, one GPU, shared flock held for each leg (BAS-80) |

`baseline` = `--spec-type none`. Every request is greedy (`temperature=0`,
`ignore_eos=true`), `max_tokens=128`, two reps; the decode prompt is fed with
`cache_prompt=true`.

## Decode A/B

Workload classes: `generic` = the shipped 5-sentence corpus repeated to fill the
context (pathological repetition, upper bound on acceptance); `docs` = this
repository's markdown concatenated in path order (largely non-repeating, lower
bound). Each row is the median of two decode reps at 128 output tokens.

| workload | context | baseline tok/s | spec tok/s | ratio | acceptance | tokens/round | gate (>=1.3x at acc>=0.5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| generic | 4 096 | 14.40 | 17.52 | **1.217x** | 0.873 | 72.0 | **fail** |
| generic | 131 072 | 8.21 | 8.26 | **1.006x** | ~0 (no drafts) | n/a | fail |
| docs | 4 096 | 17.79 | 16.29 | **0.916x** | 0.660 | 12.0 | **fail (regression)** |
| docs | 131 072 | 9.23 | 8.82 | **0.955x** | 0.167 | 2.0 | **fail (regression)** |

Readings:

- The only positive case is the 4K repeated corpus, where the model literally
  echoes the prompt and the n-gram drafter accepts 87% of a very long draft.
  Even there the win is **1.22x, below the 1.3x gate**.
- At 128K the model stops echoing and writes novel reasoning; the drafter finds
  nothing to propose. `generic` 128K produced *no* draft-acceptance line at all,
  and `docs` 128K accepted 2 of 12 drafted tokens (0.167). Speculation is then
  pure overhead: 1.006x and 0.955x.
- `docs` 4K has 0.66 acceptance and still runs **0.92x** — the draft build and
  the verify forward cost more than the accepted tokens save. The shipped engine
  has no acceptance-adaptive window (the reference `AcceptancePolicy` in
  `bench/spec_verify_core.py` does, but the engine's `ngram-*` drafter does not),
  so a low-acceptance workload pays for drafts it rejects.

Mechanism: decode is band-limited by the CPU-resident MoE expert FFN
(`--n-cpu-moe 16`). A speculative verify batch does not share that per-token
expert work, so verifying *S* drafts costs about *S* token-forwards. Speculation
only wins when the accepted run is long enough to amortise the CPU draft build
(the 4K echo case) — and even then not by the gate margin.

## Equivalence at temperature 0

`bench/measure-speculation.py compare` requires the spec-on and spec-off greedy
texts to match byte for byte.

| workload | context | identical? |
| --- | ---: | --- |
| generic | 4 096 | **yes** (325 chars) |
| docs | 4 096 | **yes** (168 chars) |
| generic | 131 072 | **no** — diverges at char 23 |
| docs | 131 072 | **no** — diverges at char 8 |

The 128K divergence is **not** a rejected-draft leak and **not** prefix-cache
contamination. The reference verify core proves the accept-longest-greedy-prefix
contract exactly (`bench/spec_verify_core.py`, 40 traces), and a clean-slate
re-test (`bench/run-speculation-equiv.sh`, `cache_prompt=false`, full prefill on
both legs, no prior speculative decode) **still diverges at char 23** —
`bench/results/2026-09-28-speculation/equiv/equiv-compare-generic.json`.

The cause is engine-level floating-point non-determinism across cache and
batching states, not a verify-core fault. The non-speculative leg alone is not
bit-stable: the same prompt and `--spec-type none` produced

- `cache_prompt=true` (reusing a prefix built by the 4K decode): `"The user has sent a massive block of repeated text ..."`
- `cache_prompt=false` (fresh full prefill): `"The user has sent a very long text that consists ..."`

Batched prompt processing and batched verify forwards accumulate in a different
order from a sequential decode, so a near-tie can flip. At 128K it flips; at 4K
it does not. Practical reading: **do not rely on bit-exact greedy equality at
long context on this engine.** The mathematical contract still holds — every
committed token is the target's own greedy token given the prefix that produced
it — but bit-equality across configurations is not guaranteed.

## Reproduce

```sh
# full matrix (queues on the shared GPU flock, ~1 h of GPU)
BONGO_GPU_LOCK_TIMEOUT=-1 ./bench/run-speculation-m32b.sh

# verify-core + compare plumbing (GPU-free)
./bench/speculation-selftest.sh
```

## Raw files

- `baseline-generic.json`, `baseline-docs.json` — spec-off (`--spec-type none`)
- `spec-generic.json`, `spec-docs.json` — spec-on (`--spec-type ngram-map-k4v`)
- `compare-generic.json`, `compare-docs.json` — per-workload verdict
- `equiv/equiv-baseline-generic.json`, `equiv/equiv-spec-generic.json`, `equiv/equiv-compare-generic.json` — clean-slate (`cache_prompt=false`) equivalence re-test
