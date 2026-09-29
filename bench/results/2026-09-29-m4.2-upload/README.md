# M4.2 (BAS-155) — host→VRAM MoE expert upload

Raw measurements for [BAS-155](/BAS/issues/BAS-155). Findings and the analysis
table: [`docs/research/m4.2-host-expert-upload.md`](../../../docs/research/m4.2-host-expert-upload.md).

Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`), Vulkan
`Vulkan1` (Intel Arc B70), tier `iq2_xs`, q8 KV, `--load-mode none
--n-cpu-moe 16 --flash-attn on --ctx-size 131072 --parallel 1`.

The engine patch is `tools/patches/m4.2-vulkan-host-expert-upload.patch`
(built with `BONGO_APPLY_M42_PATCH=1 tools/build-llama-vulkan.sh`). It adds an
env-gated per-device host buffer type (`GGML_VK_HOST_BUFT_PER_DEVICE=1`) plus two
debug probes (`GGML_SCHED_UPLOAD_STATS=1`, `GGML_VK_TRACE_HOST_COPY=1`).

## Sessions

| directory | config | delta `prompt_ms` | notes |
| --- | --- | ---: | --- |
| `ctx16k/lm_none` | `--load-mode none`, host buffer on device 0 | 3 467.8 | same-session control (stock path) |
| `ctx16k-lever/lm_none` | + device-local host buffer | 3 047.0 | |
| `ctx16k-lever-tq/lm_none` | + device-local host buffer + transfer queue | **2 721.8** | winner |
| `ctx128k-lever/lm_none_128k` | + device-local host buffer + transfer queue, prefix 127 999 | **4 661.0** | |
| `needle/lm_none` | lever correctness check | — | pass (`VAULT-COORD-7391-QXZ`) |
| `decode4k/{baseline,lever}` | 4 096-token prompt, 128 tokens, 3 reps | — | ~flat |

The M4.1 frozen `--load-mode none` baseline to beat: **3 569.4 ms** (16K) /
**5 480.6 ms** (128K) ([`docs/research/host-cpu-decomposition.md`](../../../docs/research/host-cpu-decomposition.md)).

## Reproduce

```sh
# build the lever engine
BONGO_APPLY_M42_PATCH=1 tools/build-llama-vulkan.sh "$HOME/.bongo/engine/llama.cpp-pin" llama-server

# 16K / 128K delta turns (new named configs in run-warm-prefix-profile.sh)
BONGO_HOST_SPLIT=1 BONGO_LLAMA_SERVER="$HOME/.bongo/engine/llama.cpp-pin/build-vulkan/bin/llama-server" \
  BONGO_PROFILE_OUT=bench/results/2026-09-29-m4.2-upload/ctx16k-lever-tq \
  BONGO_PROFILE_CONFIGS=lm_none_devhost_tq \
  BONGO_PROFILE_DELTAS=512 bench/run-warm-prefix-profile.sh

BONGO_HOST_SPLIT=1 BONGO_LLAMA_SERVER="$HOME/.bongo/engine/llama.cpp-pin/build-vulkan/bin/llama-server" \
  BONGO_PROFILE_OUT=bench/results/2026-09-29-m4.2-upload/ctx128k-lever \
  BONGO_PROFILE_CONFIGS=lm_none_devhost_tq_128k \
  BONGO_PROFILE_DELTAS=512 bench/run-warm-prefix-profile.sh

# 4K decode A/B
BONGO_LLAMA_SERVER="$HOME/.bongo/engine/llama.cpp-pin/build-vulkan/bin/llama-server" \
  bench/run-m4.2-decode4k.sh
```

Both runners take the shared single-GPU flock ([BAS-80](/BAS/issues/BAS-80)) for
the whole measured session.

## Instrumentation snippets

`llama-server.log` lines of the form

```
GGML_SCHED_UPLOAD_STATS layers=... copies=... bytes=... branch_ms=... id_ms=... wait_ms=...
```

are printed once per graph compute (ubatch). `layers=` counts **host-resident
expert tensors** (16 `--n-cpu-moe` layers x 3 `ffn_{gate,up,down}_exps` matrices
= 48), not MoE layers. `GGML_VK_HOST_COPY` lines come from the Vulkan pinned-copy
path; zero in the stock configuration, active with the lever.
