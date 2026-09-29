# 2026-09-29 — prefix-cache M3.0b stock control (checkpoint flag off)

Same-window control for [`../2026-09-29-prefix-cache-m3.0b/`](../2026-09-29-prefix-cache-m3.0b/):
the **stock** engine with `--save-slot-checkpoints` off, so the restored slot has no
checkpoint list. It proves (a) the restore is still not reused without the sidecar and
(b) the delta-turn time is a host property, not a checkpoint regression.

## How it was run

```sh
BONGO_LLAMA_BIN=~/.bongo/llama/b11223/vulkan \
BONGO_SAVE_SLOT_CHECKPOINTS=0 \
BONGO_PREFIXES=31744 BONGO_CTX=131072 bench/run-prefix-cache.sh
```

## Result

| case | prompt_n | cache_n | prompt_ms | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `cold_p31744` | 31743 | 0 | 189,755.8 | 189,825.3 |
| `hit_p31744` | 4 | 31739 | 276.4 | 299.7 |
| `grow_p31744_d512` (512-token turn) | 515 | 31739 | **5,727.4** | 5,751.5 |
| `grow_p31744_repeat` | 4 | 32250 | 260.5 | 284.9 |
| `slot0_prime_p31743` | 4 | 31739 | 285.7 | 309.4 |
| `slot0_after_restore_p31743` | 31743 | **0** | **185,186.0** | 185,210.4 |

- Without the checkpoint sidecar the restored slot is **not reused**: the request after
  `restore` re-prefills all 31,743 tokens (185.2 s), reproducing the M3.0a finding.
- The 512-token turn is **5,727 ms** here vs **5,676 ms** for the patched engine in the
  same window — a 0.9% difference. The M3.0a ≤5 s gate is a property of the current host
  load, not of the checkpoint change.
