# MoE4All / INFR on the Arc Pro B70 — first-pass evaluation

> **Superseded by the rigorous matrix.** The full quant × context × MTP matrix
> with 3 reps, depth decode and the paired MTP A/B is in
> [`moe4all-b70-matrix.md`](moe4all-b70-matrix.md) ([BAS-181](/BAS/issues/BAS-181)).
> The depth-0 decode here is a cold-routing number; at real depth INFR is at
> parity (`q2_0` 4K: 13.68 vs bongo 13.24) and ahead at 32K
> (12.54/15.77 vs 11.67/10.36). Prefill is ~5–6× slower only for short
> prompts; a real 32K prompt is ~1.6–1.7× slower (102/128 vs 175/208 tok/s).
> MTP does gain 1.3–2×. The engine verdict (do not switch) is unchanged, but
> the decode gap the first pass reported was wrong.

Status: first pass for [BAS-179](/BAS/issues/BAS-179). Date 2026-09-29. Author: CTO.
Verification level: **reproducible single-run measurements**, not a rigorous matrix.
The full quant × context × MTP matrix is delegated (see "Follow-up").

## Question

The board asked us to test [Headmaster218/MoE4All](https://github.com/Headmaster218/MoE4All)
("INFR") as a possible replacement/alternative to bongo for running
**Qwen3.8-Flash-Next** on the Arc Pro B70 — including **MTP** (multi-token
prediction / speculative decode), which bongo does not have — and to produce a
speed overview across a few quants, with and without MTP.

## Verdict (short)

- **It runs the model on the B70.** INFR builds from source on Fedora 44 and
  loads the Swift-1.5-Qwen3.8-Flash-Next GSQ-RCO quants, paging experts across
  the 32 GB VRAM / 30 GiB RAM / SSD tiers.
- **It is ~5x slower than bongo's llama.cpp Vulkan path** on this box:
  **~3.4–4.4 tok/s decode** and **~37 tok/s prefill** (INFR) versus
  **19.56 tok/s decode** and **~230 tok/s prefill** (bongo shipped default).
  MTP is functional but **does not close the gap** (~3.3 tok/s).
- **There is an Intel-specific blocker.** INFR's default host DMA
  (dedicated transfer queue) **hangs the Vulkan device** on the B70. All usable
  numbers here required `INFR_NO_HOST_DMA=1`. The slow path is then the expert
  pager / host upload on Intel, not the model math.
- **Recommendation:** do **not** switch bongo to MoE4All for the B70 today.
  Keep it as a reference engine; the MTP sidecar idea is worth tracking for the
  retired GPU milestone ([BAS-166](/BAS/issues/BAS-166)), because INFR shows the
  MTP head can be loaded from a sidecar for `qwen4exp`.

## What we built and how (reproducible)

No Linux binary is published (the GitHub release is a 13 MB Windows `.exe` zip),
so INFR was built from source. The toolchain is user-local (no root):

- Rust **1.97.1** via rustup → `~/.cargo`, `~/.rustup`.
- **zig 0.14.1** at `~/.local/share/zig-0.14.1` used as the C/C++ toolchain
  (`~/.local/bin/cc`, `~/.local/bin/c++` wrappers; the wrappers translate
  `--target=x86_64-unknown-linux-gnu` → `x86_64-linux-gnu`, which zig requires).
- **glslc** (`shaderc 2026.1`) extracted from the Fedora RPM to
  `~/.local/bin/glslc` — the Vulkan build compiles its compute shaders at build
  time and hard-fails without it.
- INFR source: `~/.local/share/MoE4All` at
  `ed62393068679573afe94a1472454efe7eae0f15` (tag `release-0.9.0`).

```sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export CC=cc CXX=c++ CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=cc
cd ~/.local/share/MoE4All
cargo build --release -p infr-cli --locked   # finished in ~2.5 min
# binary: target/release/infr
```

Model and MTP head used:

- Target: `~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/…-00001-of-00002.gguf`
  and `…/q2_0/…00001-of-00002.gguf` (arch `qwen4exp`, 48 blocks, 512 experts,
  hidden 2560 — so this is the same Qwen3.8-Flash-Next as bongo's target).
- MTP head: `unsloth/Qwen3.8-Flash-Next-GGUF` →
  `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (1.8 GiB), cached at
  `~/.local/share/MoE4All-models/mtp/mtp-shared-Q4_K_M.gguf`. The sidecar is
  `qwen4exp`, `nextn_predict_layers=1`, `nextn_shared_target_tensors=1`; it
  pairs with the GSQ-RCO target (hidden 2560 / 512 experts match) even though
  the GSQ-RCO GGUF itself drops the head. **This is the first time an MTP head
  has been run against the bongo target.**

## Measurements

All runs: Arc Pro B70 via `--dev Vulkan1`, `ctx 4096`, Q8 K/V, greedy,
`INFR_NO_HOST_DMA=1`, `ubatch 512`, `paging.cache=16GiB`, 1 rep. Single run per
cell — treat small differences as noise. Device: Intel BMG G31, 31.9 GiB.

| quant | mode | prefill | decode |
| --- | --- | ---: | ---: |
| IQ2_XS | ordinary (`infr bench`) | pp512 **37.0 t/s** | tg16 **3.4–3.7 t/s** |
| Q2_0 | ordinary (`infr bench`) | — | tg16 **4.4 t/s** |
| IQ2_XS | **MTP on** (`infr run`, sidecar) | 24-tok prime ~4 t/s | **3.3 t/s** |

For reference, the same model on the same box under bongo's shipped llama.cpp
Vulkan default (`docs/final-overview.md`):

| metric | bongo (llama.cpp Vulkan) | INFR first pass |
| --- | ---: | ---: |
| prefill (pp512 / 1K) | ~230–234 t/s | 37 t/s |
| 4K decode | 19.56 t/s | 3.4–4.4 t/s |
| MTP | not available | 3.3 t/s (no gain) |

INFR's own RX 7900 XTX numbers for Qwen3.8 Q2_K_XL are 155 t/s prefill / 29.45
t/s decode at depth 0 (24 GiB VRAM, 40 GiB RAM, `docs/perf/`). The B70 result is
well below that, consistent with the per-vendor gap and the pager behaviour
below.

## Findings

1. **Host DMA hangs Intel.** Default (`paging.host_dma` on) gives
   `The logical device has been lost` on the dedicated transfer queue during the
   first prefill (reproduced at `cache=6GiB/ub256` and at the auto
   `cache≈22GiB/ub2048`). With `INFR_NO_HOST_DMA=1` the same configurations run.
   This is an upstream INFR/ANV defect worth reporting; it also means the
   measured numbers use a **degraded upload path**, so they are a lower bound.
2. **The bottleneck is the pager/SSD path, not the GPU.** Even with
   `paging.cache=22GiB` (≈73% of expert blocks resident in VRAM) plus a ~20 GiB
   host cache, decode is ~3.4 t/s. The decode step waits on per-layer expert
   residency checks and host→device staging; the same wall in INFR's own docs is
   the "router→CPU→expert" structural boundary.
3. **MTP works but is masked by the same bottleneck.** The sidecar loads, the
   greedy MTP path runs (output coherent), and decode is 3.3 t/s — no better
   than ordinary. With acceptance-gated decode, MTP can only pay off once the
   per-cycle cost is dominated by useful work (INFR measures a 9–60% MTP gain on
   the 7900 XTX at 4K/20K, i.e. exactly where this box is not).
4. **The GSQ-RCO GGUFs have no MTP head** (confirmed by inspecting the GGUF
   tensor index: zero `nextn.*` tensors), matching bongo's
   [BAS-56](/BAS/issues/BAS-56) finding. MTP therefore requires either the
   unsloth/sidecar head or a different quant family — it cannot be bolted onto
   the existing GSQ-RCO file by conversion.
5. **Prefill is also far behind** (37 vs ~230 t/s). Intel Xe2 is a secondary
   INFR target: no cooperative-matrix XMX path is enabled by default
   (`VK_NV_cooperative_matrix2` is advertised but unusable, and `coopmat_8x8` is
   opt-in), and `docs/perf/vulkan-review.md` states the backend is RDNA3-tuned.

## Residual risk / caveats

- Single rep, single run, one prompt shape; no warm/cold separation. The
  `tg` numbers moved 2.8–4.4 t/s across runs, so treat them as an order of
  magnitude, not a target.
- Measurements are with the host-DMA workaround, i.e. INFR's slow path.
- No 32K/128K context runs yet; the pager is expected to degrade further with
  depth (more KV pressure, smaller expert cache).
- The MTP head is the **base** Qwen3.8 head, not a Swift-specific one; accept
  rate on the Swift fine-tune was not instrumented.

## Follow-up

Delegated as a child task with a Coder owner:

- full quant × context matrix (IQ2_XS, Q2_0, and at least one higher tier such
  as `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` IQ3_XS/IQ4_NL), 3 reps, with
  cold/warm separation, at `ctx 4096` and `ctx 32768`;
- MTP A/B with acceptance instrumentation on >= 2 prompts, ordinary and MTP
  paired on the same prompt;
- record whether INFR's host-DMA hang reproduces on a fresh kernel cache, and
  file it upstream.

The MTP sidecar pairing (base head + Swift/GSQ target) is the one result worth
carrying forward regardless of the engine decision.

## Reproduction summary

```sh
# ordinary
INFR_NO_HOST_DMA=1 infr bench <target-00001.gguf> -p 512 -n 16 -r 1 \
  --ctx 4096 --dev Vulkan1 -u 512 --set paging.cache=16GiB
# MTP
INFR_MTP=1 INFR_SPEC_DRAFT=<mtp-shared-Q4_K_M.gguf> INFR_NO_HOST_DMA=1 \
  infr run <target-00001.gguf> "What is 2+2? Answer with one word." \
  --max-new 16 --temp 0 --ctx 4096 --dev Vulkan1 -u 512 \
  --set paging.cache=16GiB
```
