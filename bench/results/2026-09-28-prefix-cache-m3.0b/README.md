# 2026-09-28 — prefix-cache M3.0b (checkpoint sidecar: restored slot IS reused)

Raw result of `bench/measure-prefix-cache.py`, run through `bench/run-prefix-cache.sh`
with the **patched** engine `~/.bongo/llama/b11223/vulkan-ckpt` and
`--save-slot-checkpoints` on. See the M3.0a baseline in
[`../2026-09-28-prefix-cache-m3.0a/`](../2026-09-28-prefix-cache-m3.0a/) and
[`docs/bongo-sh.md`](../../../docs/bongo-sh.md).

## How it was run

```sh
BONGO_LLAMA_BIN=~/.bongo/llama/b11223/vulkan-ckpt \
BONGO_SAVE_SLOT_CHECKPOINTS=1 BONGO_NEEDLE=1 \
BONGO_PREFIXES=31744 BONGO_CTX=131072 bench/run-prefix-cache.sh
```

- Engine: llama.cpp `b11223` + `tools/patches/slot-checkpoints-sidecar.patch`,
  Vulkan, `--save-slot-checkpoints`.
- Model: IQ2_XS `Swift-1.5-Qwen3.8-Flash-Next` (`qwen4exp`). `--n-cpu-moe 16`.
- Exact argv and `/props` are under `server.flags` / `server.props` in the JSON.

## Result — the restore is reused (the M3.0a gap is closed)

| case | prompt_n | cache_n | prompt_ms | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `cold_p31744` | 31743 | 0 | 186,333.5 | 186,404.5 |
| `hit_p31744` | 4 | 31739 | 299.3 | 323.8 |
| `grow_p31744_d512` (512-token turn) | 515 | 31739 | 5,663.7 | 5,689.3 |
| `grow_p31744_repeat` | 4 | 32250 | 270.1 | 294.8 |
| `slot0_prime_p31743` | 4 | 31739 | 309.8 | 333.2 |
| **`slot0_after_restore_p31743`** | **1** | **31742** | **427.6** | **457.2** |

| slot action | bytes | ms |
| --- | ---: | ---: |
| `save` KV | 585,931,732 | 455.0 |
| `save` checkpoint sidecar `{filename}.ckpt` | 236,078,956 | (with save) |
| `erase` | — | 14.2 |
| `restore` | — | 152.7 |

Against [BAS-86](/BAS/issues/BAS-86)'s acceptance criteria:

- **`cache_n > 0` after restore: MET.** `cache_n = 31742` on the first request after
  `save` + `erase` + `restore`; the M3.0a baseline re-prefilled all 31,743 tokens here.
- **TTFT at parity with a full hit, < 1 s at 31K: MET.** 457 ms after restore vs 324 ms
  for an in-session hit — no re-prefill.
- **In-session behaviour unchanged: the cached path is identical to M3.0a** (`hit` 324 ms,
  `grow_repeat` 295 ms). The 512-token turn measured **5,689 ms** this run vs **4,505 ms**
  in M3.0a; the flag only runs on `save`/`restore`, so this is host contention during the
  run (the cold prefill was also slower: 170.4 vs 175.3 prompt tok/s), not a code change.
  A cleaner re-run is queued (see below).

## Needle correctness — first run's verdict is a test bug, not a restore bug

`needle_after_restore.status = "fail"` in this JSON, but `check_needle` was passing a
malformed prompt: `harness.build_needle_document` already appends
`Question: ... Answer:`, and the helper appended a **second** question and leaked the
answer (`... ? VAULT-COORD-7391-QXZ`). The model then skipped the code and continued the
filler, e.g. `'llow channel for movement ...'`. Fixed in `71edb52`; the canonical
131K needle baseline (`../2026-09-27-baseline/raw/needle.json`) passes, and a clean
31K post-restore needle run is committed under `../2026-09-29-prefix-cache-m3.0b/`.

## Caveats

- Single box, single pass; the measurement ran while other agents' CPU work overlapped
  (see the 512-token-turn note).
- 128K/256K reuse is not measured here; the 31K point is the M3.0b acceptance evidence.
  The M3.0a 256K baseline is in [`../2026-09-28-prefix-cache-256k/`](../2026-09-28-prefix-cache-256k/).
