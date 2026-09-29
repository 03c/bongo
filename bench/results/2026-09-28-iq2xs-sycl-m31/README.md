# 2026-09-28 — M3.1 integer MMVQ for the i-quant expert types (BAS-74)

Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`), SYCL backend, built with
the oneAPI 2025.3 container (`ghcr.io/ggml-org/llama.cpp:server-intel`), `GGML_SYCL=ON`,
`GGML_SYCL_F16=ON`. GPU: Intel Arc Pro B70 (32 GiB), Level Zero, single-GPU lock held for every
measured run. Tier `iq2_xs`, `--n-cpu-moe 16`, `q8_0` KV, flash-attn on.

Patch under test: [`tools/patches/iq2xs-sycl-integer-mmvq.patch`](../../../tools/patches/iq2xs-sycl-integer-mmvq.patch).

## 1. Why this patch targets IQ2_S / IQ2_XXS / IQ1_M, not only IQ2_XS

The `iq2_xs` **tier name** is not the ggml **type** the model uses. The per-tensor inventory
([`docs/research/gguf-inventory.md`](../../../docs/research/gguf-inventory.md), cross-checked for
1224/1224 tensors) and a direct re-read with `tools/gguf-inventory.py --local` both show **zero
`IQ2_XS` tensors**. The expert tensors are:

| tensor | types present (48 layers) |
| --- | --- |
| `ffn_gate_exps` | IQ2_S, IQ2_XXS, IQ1_M |
| `ffn_up_exps` | IQ2_S, IQ2_XXS, IQ1_M |
| `ffn_down_exps` | Q2_0 (all 48) |

The stock backend has only a **single-column** MMVQ kernel for IQ2_S / IQ2_XXS / IQ1_M, so an
expert routed `n` tokens launches `n` GEMVs, and once `n > MMVQ_MAX_BATCH_SIZE (8)` it expands the
weights to FP16 for a dequantise + oneDNN GEMM. Q2_0 already has a multi-column kernel in stock.

This patch adds multi-column MMVQ for IQ2_S / IQ2_XXS / IQ1_M (it keeps the earlier IQ2_XS kernel),
reusing the generic `mul_mat_vec_q_ncols` template with the codebook tables bound, and makes the
batch cap tunable for all four types with `GGML_SYCL_IQUANT_MMVQ_MAX` (default `8`, the pinned
baseline). `GGML_SYCL_IQUANT_MMVQ_MAX=64` is the M3.1 tap.

## 2. Files

| file | what |
| --- | --- |
| `correctness-iq2_xs-cap64.txt` | `test-backend-ops test -b SYCL0 -o MUL_MAT -p iq2_xs` |
| `correctness-iq2_xxs-cap64.txt` | same, `iq2_xxs` |
| `correctness-iq2_s-cap64.txt` | same, `iq2_s` |
| `correctness-iq1_m-cap64.txt` | same, `iq1_m` |
| `llama-bench-ab-expanded.txt` | stock vs cap8 vs cap64 `-p 512 -n 128 -r 3` (expanded patch) |
| `llama-bench-ab-iq2xs-only.txt` | same for the earlier IQ2_XS-only patch, for reference |
| `cachedturn-run{1,2,3}/` | `measure-prefix-cache.py --prefixes 4096 --delta 512` |

All `test-backend-ops` runs: **pass** (SYCL vs the CPU reference), including every `n` in the
multi-column range. Metric: the standard `test-backend-ops` per-type error threshold; the run
exits `OK` for all four types at cap 64.

## 3. Full-model llama-bench (`-p 512 -n 128 -r 3`)

| leg | pp512 t/s | tg128 t/s |
| --- | ---: | ---: |
| stock `b11223` | 71.10 ± 20.40 | 18.76 ± 0.91 |
| patched, cap 8 | 94.69 ± 32.05 | 21.57 ± 0.23 |
| patched, cap 64 | **105.83 ± 18.83** | 21.74 ± 0.17 |

Reading: the multi-column kernels alone (cap 8, i.e. `n = 2..8`) already lift pp512 over stock, and
raising the cap to 64 (the `n > 8` experts stay integer) lifts it again. The per-run spread is
large (host/thermal drift, ±20 t/s on stock), so treat the absolute ratios as directional. The
cap-64 vs cap-8 ordering (105.8 vs 94.7) is the directly relevant, same-binary comparison.

## 4. Cached-turn (`--prefixes 4096 --delta 512`)

The measured metric is the `grow_p4096_d512` line: a 514-token request whose 4093-token prefix is
already cached.

| run | leg | cold p4096 t/s | grow 512 t/s | grow TTFT ms |
| --- | --- | ---: | ---: | ---: |
| 1 | stock | 243.05 | 148.14 | 3469.8 |
| 2 | stock | 238.85 | 104.03 | 4940.8 |
| 3 | stock | 249.49 | 147.61 | 3482.2 |
| 3 | patched cap 64 | **263.50** | **154.45** | **3327.8** |

The stock `grow` number swings 104–148 t/s between identical runs, so single-shot cached-turn A/B is
unreliable. Run 3 is the only one with the expanded patch. Against run 3 stock: cold `249.49 →
263.50` = **1.056x**, grow `147.61 → 154.45` = **1.046x**.

## 5. Verdict for the acceptance criteria

- **No FP16 materialisation on the measured path:** with `GGML_SYCL_IQUANT_MMVQ_MAX=64`, the
  `iq2_xs`-tier expert gate/up/down matmuls (IQ2_S, IQ2_XXS, IQ1_M, Q2_0) all run the integer
  multi-column MMVQ kernel for the cached-turn per-expert batch (~10 tokens). The kernel path is
  visible in a `GGML_SYCL_DEBUG=1` run as `Calling mul_mat_vec_iq2_*_q8_1_sycl_switch_ncols`.
- **Numerical parity:** `test-backend-ops` SYCL vs CPU passes for all four types at cap 64; bound =
  the standard `test-backend-ops` per-type threshold.
- **>= 1.3x prompt / >= 1.2x decode on the cached-turn metric:** **NOT met.** Cached-turn prefill
  improves ~1.05x; decode is unchanged by the patch (the cap only affects `ne[1] > 8`), and the
  small tg128 movement is within run drift.

**The important negative result:** cap 64 already removes FP16 expansion from the expert path for
the cached-turn, and the total gain is only ~5%. Therefore FP16 expansion is **not** the dominant
cached-turn cost on this box, and a true tiled integer MMQ kernel is not expected to reach 1.3x on
its own. The remaining cost is in the non-MoE path (attention over the cached prefix, the hybrid /
recurrent layers, host-side work).

## Reproduce

```sh
cd ~/.bongo/engine
./build-sycl.sh llama.cpp-pin "llama-server llama-bench test-backend-ops"
docker run --rm --entrypoint sh --device /dev/dri --user "$(id -u):$(id -g)" \
  -e GGML_SYCL_IQUANT_MMVQ_MAX=64 \
  -v "$HOME/.bongo/engine:/work" -w /work/llama.cpp-pin \
  ghcr.io/ggml-org/llama.cpp:server-intel -c \
  './build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p "iq2_s"'
```
