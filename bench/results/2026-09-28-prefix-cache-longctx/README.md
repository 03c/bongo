# 2026-09-28 — cached 128K/256K 512-token delta turn (BAS-132)

The M3.5b product-path metric: the time to first token for a **512-token agentic
turn** on a **warm 128K/256K prefix**, with the delta prefill rate and peak VRAM.
Raw output of `bench/run-prefix-longctx.sh`, which runs
`bench/measure-prefix-cache.py --cached-only --no-slot` at the two contexts under
one single-GPU flock (BAS-80).

```sh
./bench/run-prefix-longctx.sh          # ~1.5 h of GPU, both contexts under one lock
```

## How it was run

- Engine: llama.cpp **`b11223`** (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), **Vulkan** (`--device Vulkan1`).
- Model: IQ2_XS `Swift-1.5-Qwen3.8-Flash-Next` (`qwen4exp`, 36 Gated-DeltaNet + 12 full-attention layers).
- KV: `--cache-type-k q8_0 --cache-type-v q8_0`; `--cache-prompt` on; `--flash-attn on`; `--parallel 1`.
- Per context: `prime` (first request on the empty slot — the one full prefill),
  `hit` (identical prompt, must be a full KV reuse), `grow` (prefix + 512 tokens —
  the product metric), `repeat` (the grown prompt, must be a full hit).
- `--cached-only`: the product path never pays an explicit cold pass. The prime
  request pays the same full prefill, so the runtime is unchanged; the labels are honest.
- `--no-slot`: a restored slot is not reused on this hybrid model (BAS-86), so the
  post-restore re-prefill would add ~25/50 min per context for no product number.
- Peak VRAM is `MemorySampler` (fdinfo `drm-resident-vram0`) over each case window.
- Exact argv and `/props` are in each `bongo-config.json` / the run memory block.

## Results

### ctx128k — ctx 131072, q8 KV, `--n-cpu-moe 16` (pinned Stage 0 placement)

| case | prompt_n | cache_n | prompt tok/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `prime_p128000` | 127748 | 251 | 131.5 | 971,635.9 |
| `hit_p128000` (full hit) | 4 | 127995 | — | 527.9 |
| `grow_p128000_d512` (**512-token turn**) | 515 | 127995 | **42.2** | **12,307.4** |
| `grow_p128000_repeat` | 4 | 128506 | — | 432.8 |

Peak VRAM **29.26 GiB** (whole run; 29.05 GiB in the delta window). Delta decode 6.35 tok/s.

### ctx256k — ctx 262144, q8 KV, `--n-cpu-moe 18` (shipped 256K default)

| case | prompt_n | cache_n | prompt tok/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `prime_p256000` | 255751 | 251 | 111.9 | 2,285,284.9 |
| `hit_p256000` (full hit) | 4 | 255998 | — | 658.8 |
| `grow_p256000_d512` (**512-token turn**) | 515 | 255998 | **48.2** | **10,869.7** |
| `grow_p256000_repeat` | 4 | 256509 | — | 598.7 |

Peak VRAM **30.86 GiB**. Delta decode 5.47 tok/s.

## Finding: the `<=5 s` delta-turn target is **not met** at either context

- 128K: **12.31 s** (2.5x the target). 256K: **10.87 s** (2.2x the target).
- The full hit is cheap and stable: **0.53 s / 0.66 s**, so **prefix reuse works** —
  the cost is entirely the 512-token delta prefill.
- The delta prefill rate is **42–48 tok/s** against 96–133 tok/s at 4K–31K
  (`bench/results/2026-09-28-prefix-cache-m3.0a/`). Per new token the delta costs
  ~9–10 ms at 4K–31K and ~21–24 ms at 128K–256K; the rate roughly halves by 128K.
- The 256K point is *slightly faster per token* than 128K (48.2 vs 42.2 tok/s)
  although it offloads two more expert layers and attends over 2x the KV. One
  sample per point, so the bump may be noise, but it shows the attention-over-KV
  term is not the only driver and the 31K→256K trend is not clean.
- The gap is **prompt-processing throughput**, not cache reuse. The levers that
  move it are the planned M3.1 integer MMQ/MMVQ (IQ2_XS without FP16 expansion)
  and M3.2 suffix/n-gram speculation; a cached, unchanged-prompt retry is already
  sub-second.

## Caveats

- Single box, single pass; no variance repeats.
- The 128K run used `--n-cpu-moe 16` (the pinned baseline); the 256K run used
  `--n-cpu-moe 18` (the only safe q8 placement at 256K, `bench/results/2026-09-28-ctx256-fit/`).
  So the two points differ in placement as well as context; both are the shipped
  setting for their context.
- `grow` reuses 127,995 / 255,998 of the prefix tokens (`cache_n`), i.e. the cache
  reuse is complete; the remaining cost is the 515-token delta.
