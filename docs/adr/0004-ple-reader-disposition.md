# ADR-0004 — M3.4b PLE reader stays opt-in; close the engine prefill line

- Status: Accepted
- Date: 2026-09-28
- Deciders: CTO
- Related: [BAS-79](/BAS/issues/BAS-79), [BAS-77](/BAS/issues/BAS-77), [BAS-76](/BAS/issues/BAS-76),
  [BAS-72](/BAS/issues/BAS-72), [ADR-0003](0003-engine-direction.md),
  [`docs/research/ple-reader.md`](../research/ple-reader.md),
  [`bench/results/2026-09-28-ple-reader-engine/`](../../bench/results/2026-09-28-ple-reader-engine/summary.md)

## Context

[BAS-77](/BAS/issues/BAS-77) delivered a standalone reader for the 26.82 GiB PLE / n-gram second shard:
`O_DIRECT` reads through a bounded worker pool plus a bounded row cache, with mmap + `MADV_RANDOM` page
faults as the fallback. It measured **4.3x per token on decode** and **6.7x on a de-duplicated 2048-token
prefill chunk** against the mmap path, with zero major faults and ~34 MiB RSS growth — but that baseline was
**forced cold** (`posix_fadvise(DONTNEED)` before reads), so it measured the isolated read path, not the
engine.

[BAS-79](/BAS/issues/BAS-79) wired the reader into llama.cpp at `4da633776` (`b11223`) behind
`--ple-reader on|off|auto` (default `off`) and ran the engine A/B at 4K/128K against the pinned Stage-0
Vulkan `--n-cpu-moe 16` baseline and the M3.3 ([BAS-76](/BAS/issues/BAS-76)) byte-budget placement. Raw
files and `summary.md` are committed under
[`bench/results/2026-09-28-ple-reader-engine/`](../../bench/results/2026-09-28-ple-reader-engine/summary.md)
at `b479c2c`. The measured result:

| config | ctx | reader off | reader on | delta |
| --- | ---: | ---: | ---: | ---: |
| baseline (`--n-cpu-moe 16`) prefill | 131072 | 132.4 tok/s | 133.1 tok/s | **+0.5%** |
| m33 (`--n-cpu-moe 0` + `-ot`) prefill | 131072 | 136.5 tok/s | 135.6 tok/s | **-0.7%** |
| baseline decode | 131072 | 7.69 tok/s | 7.76 tok/s | +0.86% |
| m33 decode | 131072 | 7.57 tok/s | 7.64 tok/s | +1.03% |

- The 128K prefill (126,767 tokens) is the only real prefill, and it is **neutral**. Verdict:
  `fail-no-gain` for the "measured prefill gain" bound.
- No >5% regression vs either baseline; RSS grows at most +0.39 GiB against the 26.82 GiB table, so the
  table is never forced resident.
- The reader **is** engaged: `per_layer_token_embd.weight` is IQ4_NL (90 B/row), the
  `ggml_compute_forward_get_rows` interception fires, and `reader-process.json` shows the O_DIRECT fd for
  `on` and `[]` for `off`.
- The 4K rows are prefix-cache hits (`cache_n=4094`, `prompt_n=3`); the summariser marks them `cache-hit`
  and excludes them from the prefill bound. The one 4K movement (baseline-off prompt eval 3137 ms -> 313 ms)
  is a first-request cold-start artifact, not a reader effect: the same-context m33-**off** run (also mmap)
  is 295 ms, ~10x faster than the baseline-off 3137 ms.

The A/B carries an open scope call: accept the measured-negative and close M3.4b, or spend 1-2 more GPU-hours
on a decode-path measurement ([BAS-115](/BAS/issues/BAS-115)).

## Decision

**Accept the measured negative. Close M3.4b ([BAS-79](/BAS/issues/BAS-79)).**

1. Keep the reader merged, correct and **opt-in** behind `--ple-reader off`; do not change the default.
2. Do not authorise the proposed decode-path A/B.
3. Stop engine work on this line; the standalone read-path result is documented, not a shipping claim.

## Why

1. **The engine is not PLE-gather-bound.** The 128K prefill is MoE-compute-bound (~7.5 ms/token); the PLE
   gather is a small fraction of it. Making the read path faster does not move a compute-bound total. This
   is the same conclusion ADR-0003 R4/R7 reached for residency and RAM: the bottleneck is kernel maturity,
   not second-shard I/O.
2. **The decode path is already measured and also neutral.** Decode throughput per token is independent of
   whether the prefix was reused, so the proposed "decode A/B with no prefix reuse" would re-measure the
   quantity already in the matrix. All four decode deltas (+0.86% to +1.75%) sit at or below the
   summariser's 1% gain floor and inside single-run noise.
3. **The only apparent movement is a cold-start artifact.** The baseline 4K `off` TTFT spike is on a
   cache-hit continuation and is contradicted by the m33-`off` 4K number on the same run. It cannot carry a
   decode-gain claim.
4. **The 6.7x/4.3x is an isolated read-path number.** It was measured against a forced-cold mmap baseline
   with eviction; the engine does not force eviction and overlaps the read with compute, so the isolated
   speedup does not transfer.
5. **Cost per unit of work.** 1-2 GPU-hours on a contended single GPU (the [BAS-80](/BAS/issues/BAS-80)
   serialisation) to resolve a sub-1% effect on a default-off flag is poor value; the outcome would not
   change the default.

## Alternatives considered

- **(a) Authorise the decode-path A/B (BAS-115 option 2).** Rejected: the quantity is already measured, the
  expected effect is ~1% and below the gain floor, and the single GPU is contended. Cost does not buy a
  decision that changes.
- **(b) Remove the reader patch.** Rejected: it is safe (no >5% regressions, RSS bounded), additive, and a
  real capability if a future profile is second-shard I/O-bound. A default-off flag costs nothing at
  runtime.
- **(c) Promote `--ple-reader on` as the default.** Rejected: no measured engine gain; it would add
  `O_DIRECT` I/O and a ~90 MB row cache for no benefit.
- **(d) Keep the prefill line open with more repeats.** Rejected: two independent configs both show neutral
  prefill; the mechanism (compute-bound) explains it and a third config would not change the disposition.

## Consequences

- **Positive.** M3.4b closes on an honest measured-negative; no more GPU time on this line; the capability
  is retained and reversible; the default stays the pinned, shippable path.
- **Negative / risks.** The reader stays unproven at the engine level; the standalone 6.7x/4.3x must be
  cited as an isolated read-path result, not a throughput win. The A/B is one run per config, so sub-1%
  decode effects are not resolved — acceptable, because the default does not depend on them.
- **Revisit criterion.** Reopen only if a measured profile shows the second-shard gather on the critical
  path — e.g. page-cache pressure that evicts the table, a larger context/table, or a config where decode is
  memory-bound. Then re-run the A/B with >=3 repeats and grade the reverse of `POSIX_FADV_DONTNEED` pressure
  explicitly.

## Rollback path

The reader is a patch plus a flag on a pinned engine revision. Reverting is a pin/flag change:
`--ple-reader off` reproduces the pinned baseline bit-for-bit. No production infrastructure, DNS, billing,
secrets, or destructive operation is touched by this decision.
