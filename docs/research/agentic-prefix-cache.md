# Agentic serving profile — prefix caching, long context, and a throughput envelope

Measured 2026-09-28 on the reference box for [BAS-62](/BAS/issues/BAS-62), answering the plan-review
comment: the workload is **agentic coding** (a small first prompt that grows), so per-turn
prompt-processing must come from **prefix reuse**, and the context target is **256K** (the model's
262144 native limit) with **quantized KV**.

- Method: `bench/measure-prefix-cache.py` (new). Raw: `bench/results/2026-09-28-prefix-cache/` and
  `bench/results/2026-09-28-prefix-cache-31k/`.
- Server: llama.cpp `b11223` (`4da633776`), **Vulkan** (`--device Vulkan1`), IQ2_XS, `--n-cpu-moe 16`,
  `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`, `--parallel 1`, `--ctx-size 32768`.
- Reference box: Arc Pro B70 32 GiB VRAM, ~30 GiB RAM, Fedora 44.

## 1. The finding that changes the target

**All published Stage 0/Stage 1 numbers are cold prefill, because `bench/harness.py` sends
`cache_prompt: false`.** llama-server's `--cache-prompt` is enabled by default, and an agentic turn re-sends
the whole history, so the cached path is the one that matters and it had never been measured.

It works, and it is the dominant lever for the CEO's workload.

| cached prefix | cold prefill | cold prompt tok/s | full-hit TTFT | +512-token turn: prefill | +512 turn tok/s | +512 turn **TTFT** | decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4097 | 24.78 s | 165.4 | **190 ms** | 5.116 s | 100.5 | **5.12 s** | 14.1 |
| 16384 | 83.64 s | 195.9 | **251 ms** | 3.867 s | 133.2 | **3.88 s** | 12.5 |
| 24575 | 132.85 s | 185.0 | **258 ms** | 3.988 s | 129.1 | **4.01 s** | 12.1 |
| 31743 | 180.01 s | 176.3 | **300 ms** | 4.292 s | 120.0 | **4.32 s** | 11.2 |

Reading the table:

- A **full prefix hit is ~0.2–0.3 s** against a 25–180 s cold prefill — a 100–600x TTFT cut.
- A **512-token continuation** (the normal agentic turn delta) is **~3.9–5.1 s**, roughly flat from 4K to
  31K of cached context. The delta path runs at **100–133 tok/s**, below the cold rate (165–196 tok/s),
  because it attends over the cached KV and uses smaller batches.
- Repeating an unchanged prompt is a full hit (~0.3 s) — useful for a retry/diff turn.
- The first request after server start is slow (255 tokens took 27 s; a second server start took 3.6 s) —
  shader/kernel warmup. **Warm the server once at startup** so the user's first turn is not the compile.

Extrapolation to the target context (model, not measured): a 512-token delta at 256K stays in the
~4–8 s band if the delta rate holds at ~90–120 tok/s and the attention-over-cache term grows slowly;
the cold re-prefill of 256K is **~30+ min** at the measured 128K rate (133 tok/s). Losing the cache on a
256K session is therefore catastrophic for latency; keeping/persisting it is the product feature.

## 2. Envelope the CEO asked for

**Measured today** (Vulkan, IQ2_XS, n=16):

| Turn shape | TTFT | Decode tok/s |
| --- | ---: | ---: |
| first prompt 4K, cold | 24.8 s | 14.1 |
| first prompt 16K, cold | 83.6 s | 12.5 |
| next turn +512 tok, 4K cached | 5.1 s | 14.1 |
| next turn +512 tok, 16K cached | 3.9 s | 12.5 |
| next turn +512 tok, 31K cached | 4.3 s | 11.2 |
| unchanged prompt, 31K cached | 0.3 s | 11.5 |
| 128K, cold (prior baseline) | 983 s | 8.0 |

**Modelled after the planned levers** (targets to be measured, not promises — no Arc number exists yet for
MMQ or speculation on this model):

| Lever (milestone) | Expected on prompt processing | Expected on decode |
| --- | --- | --- |
| M3.1 integer MMQ/MMVQ (IQ2_XS, no FP16 expansion) | ~1.3–1.8x | ~1.2–1.5x |
| M3.4 PLE direct reads + prefetch | up to ~2x prefill (R3/R5) | neutral |
| M3.2 suffix/n-gram speculation | neutral | ~1.3–1.8x where acceptance ≥0.5 |
| M3.3 placement (`-ot`, dynamic LRU) | +18–27% prefill | +25–37% 4K decode |

A reasonable *working target* to hold the build to: **512-token turn TTFT ≤ 3 s at 16K context and ≤ 5 s at
256K**, and **4K decode ≥ 25 tok/s**. These are gates, not forecasts; M3.0/M3.1 measure the real multipliers.

## 3. Long context and KV budget

Geometry (12 full-attention layers, 2 KV heads, head_dim 256): `12 × 2 × 256 × 2(K,V)` = **24 KiB/token
f16 / 12 KiB/token q8**, plus the linear-attention state (constant ~0.15–0.2 GiB).

| Context | KV f16 | KV q8_0 | KV q4_0 (est.) |
| ---: | ---: | ---: | ---: |
| 128K | 3.0 GiB | ~1.5–2.0 GiB | ~0.9 GiB |
| 256K (262144) | 6.0 GiB | ~3.0–3.8 GiB | ~1.5–1.9 GiB |

Numbers include the estimated indexer KV; the 128K q8 row was cross-checked against
[gguf-inventory §6](gguf-inventory.md) and the measured VRAM after load.

Budget consequence: the shipped `--n-cpu-moe 16` peaks at **29.26 GiB of 31.92 GiB usable** and is the
128K-safe ceiling (`n=12` device-loses at 31.85 GiB). The 256K fit was then **measured directly**
(`bench/results/2026-09-28-ctx256-fit/`):

| 256K config | VRAM after load | Verdict |
| --- | ---: | --- |
| q8 KV, `n=16` | **31.79–31.82 GiB** | **unsafe** — within ~0.03 GiB of the 31.85 GiB device-loss point; loaded and served a tiny request, but any real prefill would likely lose the device |
| q8 KV, `n=18` | **30.35 GiB** | **safe**, ~1.5 GiB margin; keeps KV precision; costs 2 more layers of CPU experts |
| q4 KV, `n=16` | **30.10 GiB** | safe, ~1.75 GiB margin; loses KV precision |

So **256K is reachable on this box, but not at the shipped placement and not with the naive KV choice.**
The recommended default is **q8 KV with `--n-cpu-moe 18`**; q4 KV is the alternative if expert residency is
worth more than KV precision. A full 256K prefill was not run (~30+ min); the fit test loads the full KV
budget and is the risk it retires.

## 4. Serving model for the workload (implemented in M3.0a)

llama-server `b11223` already has the necessary machinery; bongo now exposes and records it
(see [`docs/bongo-sh.md`](../bongo-sh.md) and `bench/`):

- `--cache-prompt` is **on by default**; in-session turns reuse the slot KV (measured above).
  `bongo.sh` now passes it (or `--no-cache-prompt`) explicitly and records it, and
  `bench/harness.py` sends `cache_prompt: true` by default (`profile: agentic`), so the cached
  path is the measured default instead of the cold-prefill proxy.
- `--slot-save-path PATH` + `/slots/{id}?action=save|restore|erase` persists a slot KV to disk.
  `bongo.sh` enables the endpoint by default under `$BONGO_HOME/run/slots` and records the path;
  `bench/measure-prefix-cache.py` times save -> erase -> restore and re-sends the prompt to
  prove the restored KV is reused. Saving is explicit; no KV file is written unless a client
  asks for it.
- `--cache-idle-slots` saves idle slots to the **in-RAM** prompt cache on a new task, bounded by
  the engine's prompt-cache budget (`--cache-ram`, default 8192 MiB); `--ctx-checkpoints N`
  bounds the context checkpoints a slot may keep. Both default to the engine value and are
  exposed only as recorded overrides.
- The server is warmed at startup. The first request after a model load paid ~27 s of
  shader/kernel compile; `bongo.sh` now sends one tiny request after health so that cost lands
  at startup, not on the user's first turn (`--no-warmup` restores a cold start for A/B).

## 5. Limits of this study

- One user turn pattern (append a suffix), one model tier (IQ2_XS), one backend (Vulkan), one box.
- Prefix sizes measured to 31K; 256K is extrapolated from the 4K→31K trend plus the model geometry.
- The diffusion/attention-over-cache cost at 256K is not measured; a 128K-class prefix test is the
  confirmation experiment and needs a `--ctx-size 163840` server (~20 min per cold prefill).
- Slot save/restore was not timed here; it is standard llama.cpp behaviour and is gated as a measurement.
- The lever multipliers are the research's estimates; the whole point of M3.1/M3.2 is to replace them with
  Arc measurements.
