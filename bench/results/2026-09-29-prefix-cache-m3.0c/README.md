# 2026-09-29 — prefix-cache M3.0c (checkpoint sidecar reuse at ~128K)

Raw result of the BAS-145 follow-up: confirm the BAS-86 checkpoint-sidecar slot
restore is **reused at long context**, on the same box and in the same GPU
window as its stock control.

Run by [`bench/run-prefix-cache-m3.0c.sh`](../../run-prefix-cache-m3.0c.sh)
(detached behind the shared single-GPU flock, [BAS-80](/BAS/issues/BAS-80); the
run held the lock for 4,140 s after queueing 1,297 s).

## How it was run

```sh
mkdir -p bench/results/2026-09-29-prefix-cache-m3.0c
BONGO_GPU_LOCK_TIMEOUT=-1 setsid nohup bench/run-prefix-cache-m3.0c.sh \
    > bench/results/2026-09-29-prefix-cache-m3.0c/runner.log 2>&1 &
```

Both legs run `bench/run-prefix-cache.sh` with `ctx=131072`, `prefix=128000`:

| leg | engine | binary sha256 | flag |
| --- | --- | --- | --- |
| patched | `~/.bongo/engine/llama.cpp-pin/build-vulkan/bin` (rebuilt from `tools/build-llama-vulkan.sh` + `tools/patches/slot-checkpoints-sidecar.patch`) | `1920d777…482` | `--save-slot-checkpoints` **on** |
| stock | `~/.bongo/llama/b11223/vulkan` | `73d076b9…3a76` | off |

Base flags for both: `--n-cpu-moe 16`, q8_0 KV, `--flash-attn on`, Vulkan,
`--parallel 1`, `--cache-prompt`, `--slot-save-path …`. Exact argv is in each
`prefix-cache.json` under `server.flags`; provenance is `runner-start.json`.

## Result — acceptance criteria

Patched engine, `--save-slot-checkpoints` on:

| case | prompt_n | cache_n | prompt_ms | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `cold_p128000` | 127 999 | 0 | 957,073.9 | 957,217.7 |
| `hit_p128000` | 4 | 127 995 | 432.7 | 528.9 |
| `grow_p128000_d512` (512-token turn) | 515 | 127 995 | 6,613.4 | 6,710.4 |
| `grow_p128000_repeat` | 4 | 128 506 | 359.3 | 457.0 |
| `slot0_prime_p127999` | 4 | 127 995 | 456.8 | 551.4 |
| **`slot0_after_restore_p127999`** | **1** | **127 998** | **761.7** | **871.7** (wall) |

| slot action | bytes | ms |
| --- | ---: | ---: |
| `save` KV (`bongo-prefix-cache-slot0.bin`) | 2,004,745,172 | 1,260.2 |
| `save` checkpoint sidecar (`.bin.ckpt`) | 236,078,956 | (written with the save) |
| `erase` | — | 26.8 |
| `restore` | 2,004,745,172 read | 414.4 |

Stock engine, same window, flag off:

| case | prompt_n | cache_n | prompt_ms | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `cold_p128000` | 127 999 | 0 | 971,668.9 | 971,808.5 |
| `hit_p128000` | 4 | 127 995 | 413.0 | 505.1 |
| `grow_p128000_d512` (512-token turn) | 515 | 127 995 | 6,121.2 | 6,214.2 |
| `slot0_prime_p127999` | 4 | 127 995 | 520.8 | 610.8 |
| `slot0_after_restore_p127999` | 127 999 | **0** | **963,720.1** | 963,812.7 |

- **`cache_n > 0` after `save` + `erase` + `restore`: MET at ~128K.**
  `cache_n = 127 998` with `prompt_n = 1`; the turn costs **761.7 ms** of
  server prompt processing instead of the **963.7 s** full re-prefill the stock
  engine pays in the same window. That is the bounded follow-up BAS-86 asked
  for: it does **not** re-prefill.
- **TTFT / sidecar size recorded: MET.** Post-restore wall **871.7 ms**
  (server `prompt_ms` 761.7 ms) versus 963,812.7 ms for the stock re-prefill.
  Sidecar is **225 MiB** on top of the ~1.9 GiB KV slot; the `save` costs
  1.26 s and the `restore` 414 ms, both far below the re-prefill.
- **Stock (flag-off) control in the same window: MET** (table above).
- **Correctness after restore: MET.** `needle_after_restore.needle_present = true`
  — the sentinel `VAULT-COORD-7391-QXZ` is in the answer. The needle prompt is
  not the cached prefix, so it re-prefills (the safe fallback); the restored KV
  still drives the correct answer.
- **Shipped default unchanged: MET.** `--save-slot-checkpoints` stays opt-in;
  the stock binary does not accept it.

### Note on `restore_verified = false`

`bench/measure-prefix-cache.py` sets `restore_verified = restore_reuse AND
ttft_ok`, where `ttft_ok` needs a client `ttft_ms < 1000`. The post-restore
request generated an empty first content chunk (`text: ""`, `output_tokens: 1`),
so the streaming client reported no TTFT and the composite flag is false. The
reuse itself is `restore_reuse = true` (`cache_n = 127 998`), and the
server-reported `prompt_ms = 761.7` and client wall `871.7 ms` are both < 1 s.
This is the same flag behaviour as the 31K M3.0b run.

## Caveats

- Single box, single pass. The box was shared by other agents' measurement runs
  before the lock was taken; both engines are measured back-to-back under the
  same hold, which is the point of the A/B.
- The stock control's post-restore turn is itself a second full 128K prefill, so
  the stock leg is ~34 min of GPU time.
- 256K is not measured here; the acceptance names it optional and the 128K point
  passes. The 256K slot-restore baseline is
  [`../2026-09-28-prefix-cache-256k/`](../2026-09-28-prefix-cache-256k/).
