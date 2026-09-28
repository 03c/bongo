# M3.1 — integer IQ2_XS matmul on the SYCL backend: findings and the remaining port

Work for [BAS-74](/BAS/issues/BAS-74) (M3.1), part of [BAS-62](/BAS/issues/BAS-62).
Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`).
Author: Coder (Paperclip). Date: 2026-09-28.

## CORRECTION (M3.1b, 2026-09-28): the IQ2_XS tier contains no IQ2_XS tensors

The sections below were written on the assumption that the `iq2_xs` **tier name** is the ggml
**type** the model uses. It is not. The per-tensor inventory
([`gguf-inventory.md`](gguf-inventory.md), cross-checked against the authors' allocation and
capsule files for 1224/1224 tensors) and a direct re-read with `tools/gguf-inventory.py --local`
both show **zero `IQ2_XS` tensors in the `IQ2_XS` tier**. The expert tensors are:

| tensor | types actually present (48 layers) |
| --- | --- |
| `ffn_gate_exps` | IQ2_S, IQ2_XXS, IQ1_M |
| `ffn_up_exps` | IQ2_S, IQ2_XXS, IQ1_M |
| `ffn_down_exps` | **Q2_0** (all 48) |
| attention / shared | IQ4_XS, IQ3_S, Q6_K, IQ4_NL, Q8_0, BF16, F32 |

The IQ2_XS MMVQ kernel added in the first heartbeat is therefore inert on this model. The real
prefill lever is the i-quant **expert** types: the stock SYCL backend ships only a **single-column**
MMVQ kernel for IQ2_S / IQ2_XXS / IQ1_M, so an expert routed `n` tokens launches `n` GEMVs, and
once `n > MMVQ_MAX_BATCH_SIZE (8)` the weights are expanded to FP16 for a dequantise + oneDNN
GEMM. Q2_0 already has a multi-column kernel in stock; the other three do not.

The M3.1b patch adds multi-column MMVQ for IQ2_S / IQ2_XXS / IQ1_M (IQ2_XS kept because the task
named it), reusing the generic `mul_mat_vec_q_ncols` template with the codebook tables bound, and
makes the batch cap tunable for all four types with `GGML_SYCL_IQUANT_MMVQ_MAX` (default 8 =
pinned baseline). Correctness (`test-backend-ops` vs CPU) passes for all four types at cap 64. The
full-model A/B and the honest limits are recorded in
[`../../bench/results/2026-09-28-iq2xs-sycl-m31/README.md`](../../bench/results/2026-09-28-iq2xs-sycl-m31/README.md).

**Consequence for the milestone:** the 1.3x prompt / 1.2x decode target is not met by the MMVQ
route. A true tiled integer MMQ (or a fixed multi-column MMVQ for the remaining types) is still
required to move the cached-turn metric materially. See the results README for the numbers.

## TL;DR

1. **The integer IQ2_XS expert/dense matmul kernel does not exist in llama.cpp's SYCL backend
   — at the pinned revision or on current upstream `master`.** `ggml_sycl_supports_mmq()` is hard
   `return false` for *all* types, and the SYCL MMQ `switch (src0->type)` only implements
   `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K..Q6_K`. There is no `IQ*` MMQ tile kernel to "turn on":
   the 2-bit i-quants use a codebook (`iq2xs_grid`) and need a new `load_tiles` + `vec_dot` pair.
2. **A first, safe step is implemented and correctness-verified:** multi-column MMVQ for IQ2_XS
   (`mul_mat_vec_q_ncols<QK_K, QI2_XS/2, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1_mmvq, N>`), wired
   into `ggml_sycl_op_mul_mat_vec_q` for activation batches of 2..8. It reuses the already-shipped
   decode-side dp4a dot product and does not touch the prefill (`MUL_MAT_ID`) path.
   `test-backend-ops` for `MUL_MAT type_a=iq2_xs`: **14/14 pass** (SYCL vs CPU).
3. **Chunking the prefill batch through MMVQ is not the answer.** Extending the batch-size gate
   and running the batch as MMVQ groups removes the FP16 expansion but is far slower than the
   existing oneDNN dequant GEMM at prefill batch sizes (measured under contention: ~0.4 TFLOPS
   chunked MMVQ vs ~2.5 TFLOPS dequant GEMM at `m=4096, n=512, k=14336`). The prefill criterion
   needs a **true tiled integer MMQ**, not an MMVQ loop.
4. **Performance measurement for the milestone is not yet valid.** The reference box was held by a
   parallel M-series run (a 256K Vulkan server at ~26.5 GiB VRAM), so every GPU number below is
   contended and indicative only. A quiet-window A/B is required before any default changes.

## 1. The code path, and why prefill expands to FP16

`qwen4exp` MoE uses `GGML_OP_MUL_MAT_ID` for `ffn_{gate,up,down}_exps` (512 experts, 10 active,
48 layers). In `ggml_sycl_mul_mat_id` (`ggml/src/ggml-sycl/ggml-sycl.cpp`):

- **decode** (`src1->ne[2] == 1`): the fused path `ggml_sycl_mul_mat_id_mmvq_fused` runs
  `ggml_sycl_mul_mat_vec_q_id` → `launch_mul_mat_vec_q_moe` → `vec_dot_iq2_xs_q8_1_moe`. This is
  already integer dp4a (`mul_mat_vec_q_moe`).
- **prefill** (batch > 1): the generic counting-sort branch per expert calls
  `ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row)` with `src1_row.ne[1] = tokens-for-expert`
  (tens to hundreds). There,

  ```cpp
  bool use_mul_mat_q = ggml_sycl_supports_mmq(src0->type) && ...;   // false: returns false for all types
  ...
  } else if (use_mul_mat_vec_q) { ... }   // can_use_mul_mat_vec_q caps ne[1] <= MMVQ_MAX_BATCH_SIZE (8)
  } else if (use_mul_mat_q)     { ... }   // MMQ, disabled
  } else { ggml_sycl_op_mul_mat<no_quantize_q8_1>(..., ggml_sycl_op_mul_mat_sycl); }  // oneDNN dequant
  ```

  So experts with more than 8 routed tokens fall to `ggml_sycl_op_mul_mat_sycl`, which dequantises
  IQ2_XS to FP16 and runs an FP16 GEMM. That is the mechanism the gap analysis (R6 §2) named.

`ggml_sycl_supports_mmq` (`ggml-sycl.cpp:4037`):

```cpp
inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    return false;
}
```

Upstream `master` (fetched 2026-09-28, `4364bf723`) still has the same `return false` and the same
10-type switch — so there is no upstream commit to backport.

## 2. What is implemented (`tools/patches/iq2xs-sycl-integer-mmvq.patch`)

A ~70-line change in one file, no new header:

- `ggml/src/ggml-sycl/mmvq.cpp`
  - `vec_dot_iq2_xs_q8_1_mmvq(...)`: binds `iq2xs_grid` / `ksigns64` to the raw
    `vec_dot_iq2_xs_q8_1` so it matches `vec_dot_q_sycl_t`.
  - `mul_mat_vec_iq2_xs_q8_1_sycl_ncols<N>`: instantiates the existing generic
    `mul_mat_vec_q_ncols` template for IQ2_XS, `N = 2..8`.
  - `mul_mat_vec_iq2_xs_q8_1_sycl_switch_ncols(...)`.
  - `ggml_sycl_op_mul_mat_vec_q`: a leading hand-off for IQ2_XS that walks the activation batch in
    `MMVQ_MAX_BATCH_SIZE`-sized groups (one group at `N<=8`, so the prefill gate is unaffected).
- No change to `ggml_sycl_supports_mmq`, so the prefill path is **not** rerouted.

The patch is flag-free and does not touch the Vulkan baseline or any config default. It can be
removed by reverting the two hunks.

## 3. Correctness

`test-backend-ops test -b SYCL0 -o MUL_MAT -p iq2_xs` (SYCL vs the CPU reference):

```
14/14 tests passed
Backend SYCL0: OK
```

Raw: [`bench/results/2026-09-28-iq2xs-sycl-mmvq/correctness-iq2-xs.txt`](../../bench/results/2026-09-28-iq2xs-sycl-mmvq/correctness-iq2-xs.txt).
Cases include `n=2..8` (the new multi-column kernel) and `n=9,10,64` (the existing per-column loop,
unchanged). Metric: max relative error over the CPU reference, tested at the backend-default
tolerance; bound = the standard `test-backend-ops` per-type threshold (all cases passed, so the
error is inside it).

## 4. Performance — indicative only (contended box)

`test-backend-ops perf -b SYCL0 -o MUL_MAT -p iq2_xs`, `m=4096, k=14336` (raw files in
`bench/results/2026-09-28-iq2xs-sycl-mmvq/`):

| n (activation columns) | stock GFLOPS | patched GFLOPS | note |
| ---: | ---: | ---: | --- |
| 1 | 13.0 | 20.0 | same single-column kernel; difference is contention |
| 3 | 20.0 | 58.8 | multi-column kernel |
| 5 | 40.1 | 126.0 | multi-column kernel |
| 8 | 19.4 | 86.4 | multi-column kernel |
| 512 | 2490 | 418 | **same path in both**; difference is contention (quiet patched / busy stock) |

The `n=512` row proves the run is not a clean A/B: that case is untouched by the patch, so the
6x spread is host contention, not the kernel. Read the small-`n` rows as directional only.

A full-model `llama-bench` on this box did not complete: the parallel 256K run held ~26.5 GiB of
the 32 GiB VRAM, so the SYCL server could not fit and stalled. No end-to-end prompt/decode number
is claimed for this heartbeat.

## 5. The remaining work (the actual prefill lever)

To move the cached-turn prefill metric, add an **integer MMQ tile kernel for IQ2_XS**:

1. `load_tiles_iq2_xs` + `vec_dot_iq2_xs_q8_1_mul_mat` in `ggml/src/ggml-sycl/mmq.cpp`, following
   the CUDA dp4a design (`ggml_cuda_mmq_load_tiles_iq2_xs`, `VDR_IQ2_XS_Q8_1_MMQ` in
   `mmq-load-tiles.cuh` / `vecdotq.cuh`) but in the SYCL tile convention used by `load_tiles_q2_K`
   and `vec_dot_q2_K_q8_1_mul_mat`.
2. Add `ggml_mul_mat_iq2_xs_q8_1_sycl` + a `case GGML_TYPE_IQ2_XS` in `ggml_sycl_op_mul_mat_q`.
3. Make `ggml_sycl_supports_mmq` return true for `GGML_TYPE_IQ2_XS` (or behind
   `GGML_SYCL_IQ2XS_MMQ`), and drop the `src0->type != GGML_TYPE_IQ2_XXS` workaround for this type.
4. Re-verify with `test-backend-ops` (IQ2_XS, all shapes) and, in a **quiet window**, A/B
   `llama-bench -p 512 -n 128` plus `bench/measure-prefix-cache.py` against the M3.0 backend.

The multi-column MMVQ work in §2 is still worth keeping: it is correct, it removes the FP16
expansion for activation batches of 2..8, and it is the natural building block for the MMQ path.

## Reproduce

Container with the SYCL toolchain and GPU access (oneAPI 2025.3, already present on the box):

```sh
# build (configure + compile) inside the container
tools/build-llama-sycl.sh /path/to/llama.cpp <targets>

# correctness + micro-perf
docker run --rm --entrypoint sh --device /dev/dri --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work/llama.cpp-pin ghcr.io/ggml-org/llama.cpp:server-intel -c \
  './build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p "iq2_xs"'
```

A quiet-window end-to-end run (not done here) is:

```sh
./bench/run.sh --tier iq2_xs --repeats 3 --repeats-deep 1 --deep-threshold 131072
python3 bench/measure-prefix-cache.py ...
```

## Limitations

- No valid end-to-end prompt/decode ratio yet: the host was GPU-saturated by a parallel M-series
  run. Every performance figure here is marked contended.
- `test-backend-ops` exercises `MUL_MAT`, not `MUL_MAT_ID`; the MoE path shares
  `ggml_sycl_op_mul_mat_vec_q`, but a full-model run is the real proof.
- The MMQ port in §5 is not started.
