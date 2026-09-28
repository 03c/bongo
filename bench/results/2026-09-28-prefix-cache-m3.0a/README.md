# 2026-09-28 — prefix-cache M3.0a (cached path + slot save/restore)

Raw result of `bench/measure-prefix-cache.py`, run through
`bench/run-prefix-cache.sh` **after** the M3.0a changes (explicit `--cache-prompt`,
`--slot-save-path`, startup warmup). See
[`docs/research/agentic-prefix-cache.md`](../../../docs/research/agentic-prefix-cache.md)
for the analysis and [`docs/bongo-sh.md`](../../../docs/bongo-sh.md) for the flags.

## How it was run

```sh
BONGO_PREFIXES=4096,31744 BONGO_CTX=131072 bench/run-prefix-cache.sh
```

`run-prefix-cache.sh` acquires the shared single-GPU flock (BAS-80), starts the pinned
Stage 0 server through `bongo.sh`, runs the measurement, and stops only that server.
`--prefix-cache.json` is the raw record.

- Engine: llama.cpp **`b11223`** (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), **Vulkan**
  (`--device Vulkan1`).
- Model: IQ2_XS `Swift-1.5-Qwen3.8-Flash-Next` (`qwen4exp`, 36 Gated-DeltaNet + 12
  full-attention layers).
- Flags: `--ctx-size 131072 --jinja --flash-attn on --cache-type-k q8_0
  --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 16 --parallel 1 --metrics
  --cache-prompt --slot-save-path <dir>`.
- Exact argv and `/props` are recorded under `server.flags` / `server.props` in the JSON.

## Results (31K cached context, 512-token turn)

| case | prompt_n | cache_n | prompt tok/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `cold_p4096` | 4097 | 0 | 157.9 | 25,993.7 |
| `hit_p4096` | 4 | 4093 | — | 192.6 |
| `grow_p4096_d512` | 514 | 4093 | 96.4 | 5,338.2 |
| `cold_p31744` | 31743 | 0 | 175.3 | 181,095.0 |
| `hit_p31744` | 4 | 31739 | — | **294.2** |
| `grow_p31744_d512` | 515 | 31739 | 114.9 | **4,505.5** |
| `grow_p31744_repeat` | 4 | 32250 | — | 282.8 |

Acceptance: a 512-token turn at 31K cached context is **4.51 s (≤ 5 s)** and a full hit is
**0.29 s (< 1 s)**.

## Slot save/restore (31K point, q8 KV)

| action | bytes | ms |
| --- | ---: | ---: |
| `save` (`/slots/0?action=save`) | 585,931,732 | 127.9 |
| `erase` (`/slots/0?action=erase`) | — | 9.1 |
| `restore` (`/slots/0?action=restore`) | 585,931,732 | 264.4 |

`restore_verified` is **false**: the request after the restore re-prefilled all 31,743
tokens (179,685 ms) instead of reusing the restored KV. This model is a hybrid
`qwen4exp` GGUF; the server saves the slot tokens + sequence state but not its context
checkpoints, and the engine needs a checkpoint to resume a prefix on hybrid/recurrent
memory. `--ctx-checkpoints` / `--cache-idle-slots` do not change this and `--swa-full`
is disabled (no SWA layers). In-session prefix reuse is unaffected. See
[`docs/bongo-sh.md` § Slot KV persistence](../../../docs/bongo-sh.md#slot-kv-persistence).

## Caveats

- Single box, single pass; no variance repeats.
- The 256K q8 slot save/restore was not run (cold 256K prefill ~50 min; GPU held by the
  BAS-72 backend A/B); delegated to a follow-up.
- `server.flags` contains the ephemeral scratch slot path, which no longer exists.
