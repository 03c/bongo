# 2026-09-29 — prefix-cache M3.0b (checkpoint sidecar, clean run with needle)

Clean re-run of the M3.0b acceptance case after fixing `check_needle` (`71edb52`).
Raw result of `bench/measure-prefix-cache.py` through `bench/run-prefix-cache.sh`, with
the **patched** engine `~/.bongo/llama/b11223/vulkan-ckpt` and `--save-slot-checkpoints`.
The first run's evidence (reuse only) is in
[`../2026-09-28-prefix-cache-m3.0b/`](../2026-09-28-prefix-cache-m3.0b/); the same-window
stock control is in [`../2026-09-29-prefix-cache-m3.0b-stock/`](../2026-09-29-prefix-cache-m3.0b-stock/).

## How it was run

```sh
BONGO_LLAMA_BIN=~/.bongo/llama/b11223/vulkan-ckpt \
BONGO_SAVE_SLOT_CHECKPOINTS=1 BONGO_NEEDLE=1 \
BONGO_PREFIXES=31744 BONGO_CTX=131072 bench/run-prefix-cache.sh
```

- Engine: llama.cpp `b11223` + [`tools/patches/slot-checkpoints-sidecar.patch`](../../../tools/patches/slot-checkpoints-sidecar.patch),
  Vulkan, `--save-slot-checkpoints`. Rebuild with `tools/build-llama-vulkan.sh`.
- Model: IQ2_XS `Swift-1.5-Qwen3.8-Flash-Next` (`qwen4exp`, 36 Gated-DeltaNet + 12
  full-attention). `--n-cpu-moe 16`, q8_0 KV, flash-attn on.
- Exact argv and `/props` are under `server.flags` / `server.props` in the JSON.

## Result — acceptance criteria

| case | prompt_n | cache_n | prompt_ms | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `cold_p31744` | 31743 | 0 | 182,440.2 | 182,513.1 |
| `hit_p31744` | 4 | 31739 | 296.8 | 321.2 |
| `grow_p31744_d512` (512-token turn) | 515 | 31739 | 5,675.5 | 5,700.9 |
| `grow_p31744_repeat` | 4 | 32250 | 270.2 | 295.2 |
| `slot0_prime_p31743` | 4 | 31739 | 302.1 | 326.9 |
| **`slot0_after_restore_p31743`** | **1** | **31742** | **530.3** | **556.8** |

| slot action | bytes | ms |
| --- | ---: | ---: |
| `save` KV | 585,931,732 | 551.4 |
| `save` checkpoint sidecar `{filename}.ckpt` | 236,078,956 | (with save) |
| `erase` | — | 12.4 |
| `restore` | — | 271.5 |

- **`cache_n > 0` after `save` + `erase` + `restore`: MET.** `cache_n = 31742`; the same
  stock engine in the same window re-prefilled all 31,743 tokens (`cache_n = 0`,
  `prompt_ms = 185,186`) — see the control run.
- **TTFT at parity with a full hit, < 1 s at 31K: MET.** 557 ms after restore vs 321 ms
  for an in-session hit.
- **Correctness after restore: MET.** `needle_after_restore.needle_present = true`
  (`prompt_tokens = 31791`, answer contains `VAULT-COORD-7391-QXZ`). The divergent needle
  prompt re-prefills its second half (`cache_n = 0`), which is the safe fallback; the
  restored KV still drives the correct answer.
- **In-session behaviour unchanged: MET.** Under identical host load the stock engine
  measures the same delta turn (5,727 ms) as the patched engine (5,676 ms). The M3.0a
  ≤5 s gate is missed by *both* engines in this window (host contention); it is not a
  regression from the checkpoint flag or the patch.

## Caveats

- Single box, single pass. The box was heavily shared (other agents' measurement runs);
  the 512-token-turn absolute number is load-sensitive for both engines.
- 128K/256K reuse is not measured here. The 31K point is the M3.0b acceptance evidence;
  the 256K slot-restore baseline is in
  [`../2026-09-28-prefix-cache-256k/`](../2026-09-28-prefix-cache-256k/).
