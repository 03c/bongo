# bongo — final overview: the goal, the approach, and the speed

Status: final review for [BAS-174](/BAS/issues/BAS-174). Date 2026-09-29.
Author: CTO. This is the single-page answer to three questions: did we get where we
wanted, is it all checked in, and can you run it.

## Verdict

- **The product goal is met and verified.** On the reference box, `./bongo.sh` takes a
  checkout to a live OpenAI-compatible endpoint serving `iq2_xs` of
  Swift-1.5-Qwen3.8-Flash-Next at a **131072-token context**, with no manual steps.
- **The one-command run works today.** Verified end-to-end in this review (see
  [Verification performed](#verification-performed)). `/v1/models` returned HTTP 200 with
  `n_ctx = 131072`, and a chat completion returned HTTP 200.
- **The speed work is real and measured, but the original ambition is partially met.**
  The cached agentic turn is ~32% faster at 16K and ~20% faster at 128K versus the
  pre-M4 baseline. **4K decode is ~19.5 tok/s**, below the original `>=25 tok/s`
  ambition. That ambition is retired to the unfunded GPU milestone
  [BAS-166](/BAS/issues/BAS-166) (backlog), and the current decode target is
  `>=19.0 tok/s` median on the CEO decision [BAS-171](/BAS/issues/BAS-171).
- **"Is it all checked in?" — no, and this review fixes it.** Every M3/M4 commit was
  pushed to the branch `BAS-62-improve-speed-architecture`, but GitHub `main` was stale
  (only M0/M1). This review merges the speed work onto `main` through
  [the review branch](#check-in-and-github-state).

## The goal

The initial asks are in [BAS-48](/BAS/issues/BAS-48) and [BAS-62](/BAS/issues/BAS-62):

- A local runtime for the **Intel Arc Pro B70** (Battlemage, 32 GB), the Intel-Arc
  counterpart to [Strata](https://github.com/Niko1221/Strata).
- **One command** that pulls the model and serves it behind an **OpenAI-compatible
  endpoint**.
- **128K context minimum**, on a box with **32 GB VRAM + 32 GB RAM + SSD**.
- One model only: **Qwen3.8-Flash-Next** (the Swift 1.5 fine-tune), starting at
  **IQ2_XS** and reaching for higher quants if they fit.
- Be "clever" about experts, KV, and the SSD-resident n-gram shard, and get **better
  performance than plain llama.cpp** by taking the Strata/NInfer approach.

Explicit non-goals at the start: not a multi-model server, not a CUDA port, not a
general-purpose inference engine.

## Can you run it?

Yes. On the reference box:

```sh
git clone https://github.com/03c/bongo && cd bongo && ./bongo.sh
```

`bongo.sh` detects the Arc B70, provisions the compute runtime, fetches the pinned
llama.cpp build (and the M4.2-patched Vulkan engine), downloads the GGUF with resume,
and launches `llama-server` at `http://127.0.0.1:8080/v1`. The default resolves to:

```
--ctx-size 131072 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
--n-gpu-layers 99 --n-cpu-moe 12 --load-mode none --cache-prompt
--device Vulkan1
env GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1
```

Useful switches: `--engine stage0` is the no-rebuild opt-out to the Stage 0 baseline;
`--placement tier` pins the fixed per-tier expert split; `--ctx 262144` runs 256K;
`--gguf-dir DIR` reuses an existing download; `--detach` runs in the background.
Full reference: [`docs/bongo-sh.md`](bongo-sh.md).

## The approach

The work was staged by ADRs and measured at every gate. The plan changed twice, both
times because a measurement contradicted the assumption.

| Stage | Direction | Outcome |
| --- | --- | --- |
| **M0 foundation** | Research the hardware, the model, and the runtime options. | [ADR-0001](adr/0001-runtime-architecture.md), [ADR-0002](adr/0002-baseline-engine.md): llama.cpp is the baseline; 1,224-tensor GGUF inventory; the published GGUF has **no usable MTP head**, so speculation is re-scoped. |
| **M1 baseline** | Ship the one-command setup and a reproducible benchmark. | `bongo.sh` + `bench/harness.py`; Stage 0 numbers below; QA-verified ([BAS-54](/BAS/issues/BAS-54)); CX fixes in [BAS-58](/BAS/issues/BAS-58). |
| **M2 placement** | Can a custom adaptive VRAM expert cache close the gap at 128K? | **Go/no-go: no-go** ([BAS-53](/BAS/issues/BAS-53)). At 128K decode is flat across the feasible static range; the split is already at the VRAM edge. |
| **M3 engine** | Research Strata/NInfer, then patch llama.cpp where it pays. | [ADR-0003](adr/0003-engine-direction.md): **patch, do not rebuild**. Backend A/B keeps **Vulkan** ([BAS-72](/BAS/issues/BAS-72)); prefix cache and the slot sidecar land; integer MMVQ ~1.05x, n-gram speculation negative, PLE reader neutral, 256K context shipped; the warm-prefix profile shows the turn is **80.1% host/CPU** ([BAS-130](/BAS/issues/BAS-130)). |
| **M4 critical path** | Attack the dominant host/CPU term, not the GPU kernel. | [ADR-0005](adr/0005-host-cpu-critical-path.md): `--load-mode none` (-10.4%/-7.1%), then the **M4.2 root-cause fix** ships as the default (M4.3), decode is profiled (M4.4), and `--placement auto` becomes the default (M4.5). |

The decisive finding: the planned GPU lever (expert matmul) was **3.4% of the turn**,
while the main-thread host→VRAM expert upload was **~38%**. The M4.2 fix — pinning the
host buffer to the *compute* device instead of `devices[0]` (the AMD iGPU) — collapsed
that branch from 1136 ms to 45 ms per turn. That, not a kernel rewrite, is what met the
turn targets. Details and rollback: [ADR-0005](adr/0005-host-cpu-critical-path.md#rollback-path).

Two lines were stopped on measurement, and that is a feature of the approach:
[ADR-0006](adr/0006-moe-expert-lru-disposition.md) stops the dynamic VRAM expert LRU
(-49% to -85%), and [ADR-0004](adr/0004-ple-reader-disposition.md) keeps the PLE reader
opt-in (neutral engine A/B).

## The speed

### Stage 0 baseline to shipped default

Stage 0: IQ2_XS, pinned llama.cpp `b11223`, stock Vulkan, `--n-cpu-moe 16`, warm,
median. Shipped default: `--placement auto` (`--n-cpu-moe 12` at <= 131072), the
M4.2-patched engine, `--load-mode none`.

| metric | Stage 0 baseline | Shipped default | change | target |
| --- | ---: | ---: | ---: | --- |
| 512-token cached turn, 16K | 3983 ms | **2721 ms** | **-31.7%** | <= 3000 ms — met |
| 512-token cached turn, 128K | 5900 ms | **4746 ms** | **-19.6%** | <= 5000 ms — met |
| 4K decode | 17.7 tok/s | **19.56 tok/s** | +10.5% | >= 19.0 — met (re-baselined) |
| 4K decode vs the tier split | 16.96 tok/s | **19.56 tok/s** | **+15.3%** | placement lever |

Stage 0 raw matrix (prompt tok/s / decode tok/s / TTFT ms), for reference:

| context | prompt | decode | TTFT |
| ---: | ---: | ---: | ---: |
| 1024 | 234.2 | 19.9 | 4318 ms |
| 4096 | 231.5 | 17.7 | 17706 ms |
| 32768 | 174.5 | 11.7 | 187760 ms |
| 131072 | 133.2 | 8.0 | 982856 ms |

### What else ships

- **256K context** is available and needle-verified; the default loads at a ~30.65 GiB
  VRAM peak ([BAS-78](/BAS/issues/BAS-78), [BAS-158](/BAS/issues/BAS-158)).
- **Prefix caching / agentic turns**: a 512-token growing turn is 4.51 s at 31K and
  0.29 s on a full prefix hit ([BAS-73](/BAS/issues/BAS-73)); a restored slot reuses its
  prefix via the checkpoint sidecar ([BAS-86](/BAS/issues/BAS-86)).
- **Cheaper in RAM than assumed**: the shipped default's server VmRSS is ~2.4 GiB (this
  review measured 2.56 GiB), not the ~10 GiB of the `mmap` opt-out.

### The ceiling, honestly

- Placement is at its edge: `--n-cpu-moe 12` is the lowest loadable split at 131072
  (`10` and `8` OOM), giving a **19.59 tok/s** host-side ceiling.
- The **GPU-busy floor is 38.7 ms/step (~25.8 tok/s)** with zero host time. So `>=25`
  needs a ~25-30% GPU-side cut in flash attention + dense matmuls, not a config lever.
  That is the unfunded, backlog milestone [BAS-166](/BAS/issues/BAS-166).
- The decode target was therefore re-baselined twice by the CEO:
  [BAS-164](/BAS/issues/BAS-164) set `>=19.5`, then [BAS-171](/BAS/issues/BAS-171)
  corrected it to `>=19.0` because the shipped default straddles 19.5 at the ceiling
  (session medians 19.556 / 19.421 / 19.444; pooled 9-run median 19.444).

## What is not done

- **A Strata/NInfer-class custom engine was not built.** [ADR-0003](adr/0003-engine-direction.md)
  chose "patch llama.cpp" over a rewrite, and the measurement justified it: the gap was
  host scheduling, not op coverage. The remaining GPU-side decode work is
  [BAS-166](/BAS/issues/BAS-166) (backlog, low, unfunded).
- **SYCL is not the default.** The M3.0 A/B found SYCL slower on every 128K gate, so
  Vulkan stays the default ([BAS-72](/BAS/issues/BAS-72)). The later SYCL runtime
  enumeration fix and the Q2_0 tier trial sit in the unmerged PR #4 (see below).
- **Higher-quant tiers are not the shipped path.** IQ3_XXS does not fit this hardware;
  Q2_0 was trialed ([BAS-59](/BAS/issues/BAS-59), in PR #4) but is not the default.
- **No clean-room install has been verified** without the existing `$BONGO_HOME` and
  cache. [BAS-54](/BAS/issues/BAS-54) recorded a container/VM run as UNVERIFIED.

## Check-in and GitHub state

Before this review:

- `origin/main` was at the M0/M1 merge (`aeae090`), with only PRs #1-#3 merged.
- All M3/M4 work (113 commits) was committed and pushed to
  `origin/BAS-62-improve-speed-architecture`, but had **no pull request** and was **not
  on `main`**.
- **PR #4** ([BAS-57](/BAS/issues/BAS-57) SYCL enumeration + [BAS-59](/BAS/issues/BAS-59)
  Q2_0) was open and unmerged.

This review lands the speed work on `main` through one review branch:
`BAS-174-review-and-updates` merges `BAS-62-improve-speed-architecture` (fast-forward
content plus a merge commit) and adds this document plus the README status update. The
pull request is the `pull_request` work product on [BAS-174](/BAS/issues/BAS-174).

PR #4 is left as a **separate, deliberate follow-up**: it touches
`setup_runtime_env()` in `bongo.sh`, which the M3/M4 work rewrote, so it needs a rebase
and a test pass, not a blind merge. It is filed as a child task with a Coder owner
(see the [BAS-174](/BAS/issues/BAS-174) thread).

## Verification performed

Smallest checks that prove the claims in this document:

1. **Syntax + unit tests.** `bash -n bongo.sh` clean; `bash tests/bongo-sh.test.sh`
   **70 passed, 0 failed** (includes the shipped `--placement auto` default, the
   `--placement tier` opt-out, and the forced-OOM fallback).
2. **Plan resolution.** `./bongo.sh --dry-run --gguf-dir <iq2_xs>` printed
   `--n-cpu-moe 12 (auto)`, Vulkan, engine `m42`, `--load-mode none`, with the two
   `GGML_VK_*` levers.
3. **End-to-end serve on the reference box** (Arc Pro B70, Fedora 44, `xe`): started the
   shipped default; `/v1/models` returned HTTP 200 with `n_ctx = 131072` and
   `ftype = IQ2_XS`; a chat completion returned HTTP 200 with a correct answer and
   `predicted_per_second = 22.05` on a short prompt; `VmRSS = 2.56 GiB`. The server was
   then stopped and `pgrep llama-server` confirmed clear.
4. **Regression evidence** is committed under `bench/results/`, and the target status is
   recorded in [ADR-0005](adr/0005-host-cpu-critical-path.md) and
   [`docs/roadmap.md`](roadmap.md).

## Residual risk

- The end-to-end check reused the existing `$BONGO_HOME` (runtime, engine, model). A
  **cold, clean-room install is still unverified**; treat the first fresh clone as
  [BAS-57](/BAS/issues/BAS-57)/PR #4 territory for SYCL and as an unverified path for
  provisioning.
- The decode number is **at the placement ceiling** and noisy: run0 is a cold outlier
  (~19.0), warm runs ~19.4-19.6. The `>=19.0` target is a median of 3, not a best run.
- `main` will show **PR #4 as behind/conflicting** after this merge until the follow-up
  rebases it.

## Rollback

Every change is config- or pin-reversible; no data migrations and no production
infrastructure. `--engine stage0` returns `bongo.sh` to the Stage 0 Vulkan baseline
without a rebuild. Reverting the merge commit on `main` restores the M0/M1 line. See
[ADR-0005 § Rollback path](adr/0005-host-cpu-critical-path.md#rollback-path).

## Where the evidence lives

- [`docs/roadmap.md`](roadmap.md) — the milestone ledger and target status.
- [`docs/adr/`](adr/) — ADR-0001..0006, including every decision and its alternatives.
- [`docs/bongo-sh.md`](bongo-sh.md) — the one-command reference, backends, and flags.
- [`bench/results/`](../bench/results/) — raw JSON/markdown for every number above.
- [`docs/research/`](research/) — the 24 research notes, including the Strata/NInfer
  surveys and the engine gap analysis.
