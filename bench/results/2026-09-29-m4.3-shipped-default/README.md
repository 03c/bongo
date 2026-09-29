# M4.3 (BAS-158) — the shipped `bongo.sh` default

Raw measurements for [BAS-158](/BAS/issues/BAS-158). Findings and the analysis
table: [`docs/research/m4.3-shipped-default.md`](../../../docs/research/m4.3-shipped-default.md).

Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`) + the M4.2
patch ([`tools/patches/m4.2-vulkan-host-expert-upload.patch`](../../../tools/patches/m4.2-vulkan-host-expert-upload.patch)),
Vulkan `Vulkan1` (Intel Arc B70), tier `iq2_xs`, q8 KV, `--n-cpu-moe 16`, q8 KV,
`--parallel 1`, `--flash-attn on`, one GPU-lock hold ([BAS-80](/BAS/issues/BAS-80)).

## Configs (one server restart each)

| directory | config | 512-token delta `prompt_ms` | notes |
| --- | --- | ---: | --- |
| `turns/shipped_default` | `bongo.sh` default: `--load-mode none` + both upload env vars | **2 716.1** | 16K, target ≤3 000 |
| `turns/shipped_default_128k` | same, prefix 127 999 | **4 669.3** | 128K, target ≤5 000 |
| `turns/stage0_optout` | `bongo.sh --engine stage0` | 5 637.6 | 16K, 566 MB in-turn re-read |
| `needle/shipped_default` | shipped default | — | correctness check |
| `ctx256/ctx256-fit.json` | `--ctx 262144 --n-cpu-moe 18` + levers | — | fit/load spot check |

Cold prefill: 16K 57 038.5 ms; 128K 751 959.7 ms — within 0.05% of M4.2
(57 059.0 / 751 552.0).

## Reproduce

```sh
# the whole measured session, resumable, under the shared GPU flock
bench/run-m4.3-shipped-default.sh

# or the pieces
bash tests/bongo-sh.test.sh
BONGO_HOST_SPLIT=1 \
  BONGO_LLAMA_SERVER="$HOME/.bongo/engine/llama.cpp-pin/build-vulkan/bin/llama-server" \
  BONGO_PROFILE_OUT=bench/results/2026-09-29-m4.3-shipped-default/turns \
  BONGO_PROFILE_CONFIGS=shipped_default,shipped_default_128k,stage0_optout \
  BONGO_PROFILE_DELTAS=512 bench/run-warm-prefix-profile.sh
BONGO_NEEDLE_CONFIGS=shipped_default \
  BONGO_NEEDLE_OUT=bench/results/2026-09-29-m4.3-shipped-default/needle \
  bench/run-needle-check.sh
```

The harness configs mirror `bongo.sh`'s default and opt-out; `--load-mode none` is
the M4.1 lever and the `server-flags.json` / `env` record the two `GGML_VK` vars.
