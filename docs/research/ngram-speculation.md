# Suffix / n-gram speculation (M3.2)

Status: **measurement in progress** — engine path wired, equivalence proven in the reference core;
live A/B numbers land in `bench/results/2026-09-28-speculation/`.

- Issue: [BAS-75](/BAS/issues/BAS-75) (child of [BAS-62](/BAS/issues/BAS-62)).
- Engine: llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), **Vulkan**, IQ2_XS,
  `--n-cpu-moe 16`.
- Tools: [`bench/spec_verify_core.py`](../bench/spec_verify_core.py) (reference core),
  [`bench/measure-speculation.py`](../bench/measure-speculation.py) (engine A/B),
  [`bench/run-speculation-ab.sh`](../bench/run-speculation-ab.sh) (one-command runner).

## 1. The finding that de-scopes the engine patch

The published GGUF has no MTP head, so the plan said "implement the head-agnostic verify core". The
pinned engine already ships it. `llama-server --help` lists self-speculation types:

```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,
            ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
```

`ngram-*` are **drafter-free** (self-speculation): the drafter is a suffix/n-gram lookup over the
context, exactly the "point-mass proposal from a suffix/ngram drafter" in the brief. The verify loop
(`common/speculative.cpp`) is `prepare_verify_inputs -> target forward -> accept-longest-greedy-prefix
-> atomic prefix commit`. So M3.2 is a **configuration + measurement** task, not a new engine path —
the risk is that the numbers are bad, not that the code is missing.

The engine also exposes synthetic acceptance for benchmarking:

```
--spec-synth-len L          target mean acceptance length (including the target token)
--spec-synth-rates P0,P1,... unconditional per-position acceptance (non-increasing)
```

Synthetic acceptance makes the drafter's proposals random, so it cannot be used for correctness (the
engine warns "generated output is not valid") — only for the speedup-vs-acceptance curve.

## 2. Baseline flags (pinned and selectable)

Stage 0 stays the default and is unchanged:

```
--model <iq2_xs shard 1> --ctx-size 131072 --jinja --flash-attn on
--cache-type-k q8_0 --cache-type-v q8_0
--n-gpu-layers 99 --n-cpu-moe 16 --parallel 1 --metrics --device Vulkan1
```

Speculation is additive and reversible; `--spec-type none` is the baseline:

```
--spec-type ngram-map-k4v          # suffix/n-gram drafter (recommended default)
--spec-ngram-map-k4v-size-n 12     # lookup n-gram length
--spec-ngram-map-k4v-size-m 48     # draft m-gram length
--spec-ngram-map-k4v-min-hits 1
```

To make an agentic turn path the product path, combine with the prefix-cache serving config from
[BAS-73](/BAS/issues/BAS-73): `--cache-prompt`, `--slot-save-path`, and a warm-up request at start.

## 3. Reference verify core and equivalence

`bench/spec_verify_core.py` fixes the contract in one testable place:

- `NgramDrafter` — longest-suffix point-mass proposal;
- `prepare_verify_inputs` — prefix + whole draft;
- `target_forward` + `accept_longest_greedy_prefix` — accept the longest run matching the target's own
  greedy tokens and take the target's token at the first mismatch as the bonus;
- `AcceptancePolicy` — EWMA acceptance sizes the window, shrinking to 1 when acceptance falls below a
  threshold so a low-acceptance workload never pays for drafts.

`python3 bench/spec_verify_core.py` proves, with no GPU:

1. speculative output is **byte-identical** to plain greedy across 40 synthetic traces
   (orders 1–6, 8 seeds, 96 tokens each);
2. the core never commits a rejected draft token;
3. the window shrinks to 1 for low acceptance and grows to the cap for high acceptance;
4. a perfect drafter reaches 6.0 tokens/round;
5. a never-matching drafter leaves the output identical (no regression).

The engine-level equivalence gate is `measure-speculation.py compare`: the same prompt is decoded
greedily (`temperature=0`) once with `--spec-type none` and once with the n-gram type; the two texts
must match.

## 4. Engine A/B

`bench/measure-speculation.py` records, per context: the greedy equivalence generation, streaming
decode timings and TTFT with `cache_prompt=true`, and the draft counters (from `timings` or parsed from
the server's `draft acceptance = ...` log line). Raw files:
`bench/results/2026-09-28-speculation/{baseline,spec,synth-*}.json`.

Results are added below once the run completes (see that directory for the raw JSON).

## 5. Limits

- n-gram acceptance depends on context repetition; the generic corpus prompt is not the agentic coding
  distribution, so the real-`ngram` row is a lower bound.
- Synthetic acceptance isolates the verify machinery's speedup but produces invalid text.
- One tier, one backend, one box.
