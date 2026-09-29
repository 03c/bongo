# R6 — Intel Arc Pro B70 / SYCL kernel feasibility and device-loss root cause

Research task [BAS-68](/BAS/issues/BAS-68), part of the [BAS-62](/BAS/issues/BAS-62) recon graph.
Status: **first pass, measured on the reference box, 2026-09-28.** Author: Coder (Paperclip).

Deliverable for the "what is feasible for a bongo-owned Intel-native engine" question. It resolves
the device-enumeration premise, inventories SYCL op coverage against the ops `qwen4exp` needs, gives a
SYCL-vs-Vulkan capability table, and names the decisive experiment.

Primary sources: llama.cpp `b11223` source tree (`4da6337767f973e2b4d0797e5b323d77d8565e4a`) read directly;
the reference box; and the sibling docs [`expert-placement.md`](expert-placement.md),
[`intel-arc-b70.md`](intel-arc-b70.md), [`strata-architecture.md`](strata-architecture.md) (R1, BAS-63 —
supersedes the recon on mechanism), [`strata-ninfer-recon.md`](strata-ninfer-recon.md),
[ADR-0002](../adr/0002-baseline-engine.md).

## TL;DR

1. **The "NEO abort / B70 not enumerated" premise is stale.** It was never a driver, kernel, or build
   defect. It was a **user-space oneAPI runtime provisioning gap** in `bongo.sh`, and it is fixed by
   [BAS-57](/BAS/issues/BAS-57). With the pinned 2025.3 runtime, `llama-ls-sycl-device` enumerates
   `[level_zero:gpu:0] Intel Arc Pro B70 Graphics`. Both original failure modes are reproduced below.
2. **SYCL op coverage is complete for this model.** Every op used by the `qwen4exp` graph (MoE
   `MUL_MAT_ID`, `FLASH_ATTN_EXT`, `GATED_DELTA_NET`/`SSM_CONV`, `ROPE`, `GET_ROWS`, `TOP_K`) is
   implemented in the SYCL backend — the same set Vulkan implements. Coverage is **not** the gap.
3. **The gap is kernel maturity, not missing kernels.** On the shipped `IQ2_XS` tier both backends
   dequantize in prefill: SYCL's integer `MMQ` is globally disabled in `b11223`
   (`ggml_sycl_supports_mmq()` returns `false`), and Vulkan has no integer MMQ kernel for `IQ2_XS`.
   SYCL *does* have a quantized dp4a MoE kernel for decode (`mul_mat_vec_q_moe` + `vec_dot_iq2_xs_q8_1`).
4. **Measured on the reference box (contended host, see caveat): SYCL prefill is ~1.4x faster than
   Vulkan, but SYCL decode is ~1.6x *slower*.** The SYCL graphs path is disabled for this model because
   `MUL_MAT_ID` forces a blocking host wait, a plausible mechanism for the decode deficit.
5. **Verdict: No-go on a from-scratch Intel-native engine as the Stage 1/2 answer.** Make the
   already-implemented llama.cpp SYCL backend the baseline (it is now viable), and spend the budget on
   schedule/speculation and backend selection, not a new engine. The decisive experiment is the full
   warm 4K/**128K server harness on SYCL vs Vulkan** — TTFT is the product's binding cost.

---

## 1. Device enumeration — root cause and reproduction

### 1.1 What was actually wrong

The stage-0 note in `docs/roadmap.md` says "the Intel compute stack (Level Zero / SYCL) does not
enumerate the B70 (NEO abort), so these numbers are on the Vulkan fallback." That premise is now
resolved. Three independent, purely user-space provisioning gaps could each produce a
"no device" or "NEO abort" symptom, and `bongo.sh` had all three in sequence (fixed by BAS-57):

| # | Gap | Symptom |
| - | --- | ------- |
| 1 | The prefix had no `libsycl.so.8` (the `intel-oneapi-runtime-dpcpp-cpp` meta-package does not depend on `intel-oneapi-runtime-dpcpp-sycl-core`). | `error while loading shared libraries: libsycl.so.8: cannot open shared object file` — looks like "no device". |
| 2 | The Fedora `intel-igc-libs` RPM ships `libigc.so.2.36.3+0` with SONAME `libigc.so.2` but no symlink (a real RPM install relies on `ldconfig`; a plain `cpio` extraction does not create it). NEO cannot `dlopen` IGC during device init. | `Abort was called ... gmm_helper/resource_info.cpp` (SIGABRT, exit 134). This is the reported "NEO abort". |
| 3 | `libur_adapter_level_zero.so.1` needs `libumf.so.1`, shipped separately in `intel-oneapi-umf-1.0` and not pulled by the other meta-packages. Without it the Level Zero UR adapter silently fails to load. | `terminate called ... sycl::_V1::exception: No device of requested type available`, thrown from `dpct::dev_mgr` → `sycl::detail::select_device`. This is the "SYCL sees zero devices" case. |

It is a **runtime provisioning issue**, not a driver, firmware, kernel, or build issue. No driver or
firmware state was changed to fix it, and none needs to be. The `xe` driver was already correct.

### 1.2 Exact reproduction (verified on the reference box, 2026-09-28)

Reference box: Intel Battlemage G31 `[Arc Pro B70]` (`8086:e223`, Sparkle subsystem `172f:0105`),
`xe` driver, Fedora 44, kernel `7.0.13-200.fc44.x86_64`, 32 GiB VRAM, 30 GiB RAM.
Engine: llama.cpp `b11223` `4da6337767f973e2b4d0797e5b323d77d8565e4a`, prebuilt asset
`llama-b11223-bin-ubuntu-sycl-fp16-x64` at `~/.bongo/llama/b11223/sycl`.

The pinned user-local runtime was provisioned into a scratch prefix (no root, no system change):

```sh
# repo = https://yum.repos.intel.com/oneapi ; extract RPMs with rpm2cpio | cpio
# pinned ABI first, then the pinned packages extracted LAST so they win:
#   intel-oneapi-runtime-dpcpp-sycl-core-2025.3.3-30   -> libsycl.so.8
#   intel-oneapi-runtime-mkl-2025.3.1-8
#   intel-oneapi-runtime-dnnl-2025.3.0-409
#   intel-oneapi-umf-1.0-1.0.3-17                       -> libumf.so.1
#   + Level Zero / IGC / gmmlib / opencl from the Fedora repo:
#   intel-level-zero-26.22.38646.6-4.fc44, oneapi-level-zero-1.28.6-1.fc44,
#   intel-igc-libs-2.36.3-3.fc44, intel-gmmlib-22.10.2-1.fc44, intel-opencl-26.22.38646.6-4.fc44
# recreate SONAME symlinks (the cpio gap):
#   $RUNTIME/usr/bin/ldconfig -n $RUNTIME/usr/lib64
```

Working environment (this is the whole "smallest change"):

```sh
R=~/.bongo/runtime
export LD_LIBRARY_PATH="$R/opt/intel/oneapi/redist/lib:$R/opt/intel/oneapi/umf/1.0/lib:$R/usr/lib64/llvm15/lib:$R/usr/lib64"
export ZEL_LIBRARY_PATH="$R/usr/lib64"
```

Probe and result:

```sh
$ ~/.bongo/llama/b11223/sycl/llama-ls-sycl-device
Found 1 SYCL devices:
|ID|        Device Type|                                   Name|Version|compute units|...|Global mem| Driver version|
| 0| [level_zero:gpu:0]|             Intel Arc Pro B70 Graphics|   20.2|          256|...|   34242M | 1.15.38646+6|
```

`llama-server --list-devices` reports `SYCL0: Intel(R) Arc(TM) Pro B70 Graphics (32656 MiB, 32585 MiB free)`.

Failure mode #2 (IGC SONAME symlink removed — exactly what a raw `cpio` extract produces):

```
$ mv $R/usr/lib64/libigc.so.2{,.hidden}; llama-ls-sycl-device
Abort was called at 15 line in file:
/builddir/build/BUILD/intel-compute-runtime-26.22.38646.6-build/.../shared/source/gmm_helper/resource_info.cpp
# exit 134 (SIGABRT), core dumped
```

Failure mode #3 (`libumf.so.1` removed):

```
$ terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  No device of requested type available.
  ...
  libsycl.so.8 ... sycl::detail::select_device(...)
  libggml-sycl.so.0 ... dpct::dev_mgr::dev_mgr()
```

Failure mode #1 (ABI mismatch — the state the repo was in before BAS-57, and the state of the shared
`~/.bongo/runtime` observed at the start of this task):

```
$ ~/.bongo/llama/b11223/sycl/llama-ls-sycl-device
error while loading shared libraries: libsycl.so.8: cannot open shared object file: No such file or directory
```

### 1.3 Version matrix that enumerates the B70

| Component | Version |
| --- | --- |
| OS / kernel | Fedora 44 / `7.0.13-200.fc44.x86_64` |
| GPU / driver | Arc Pro B70 (BMG G31) / `xe` |
| Level Zero loader | `oneapi-level-zero` 1.28.6-1.fc44 (`libze_loader.so.1.28.6`) |
| Level Zero Intel GPU driver (NEO) | `intel-level-zero` 26.22.38646.6-4.fc44 (`libze_intel_gpu.so.1.15.38646`) |
| IGC | `intel-igc-libs` 2.36.3-3.fc44 (`libigc.so.2`, `libigdfcl.so.2`) |
| gmmlib | `intel-gmmlib` 22.10.2-1.fc44 |
| SYCL core (DPC++ runtime) | `intel-oneapi-runtime-dpcpp-sycl-core` **2025.3.3-30** (`libsycl.so.8`) |
| oneMKL / oneDNN | `intel-oneapi-runtime-mkl` 2025.3.1-8 / `intel-oneapi-runtime-dnnl` 2025.3.0-409 |
| UMF | `intel-oneapi-umf-1.0` 1.0.3-17 (`libumf.so.1`) |
| TBB | `intel-oneapi-runtime-tbb` 2023.1.0-151 |

The SYCL asset for `b11223` is built against the **2025.3 ABI**. The oneAPI repo now also serves
2026.1 (`libsycl.so.9`, MKL `.so.6`); if a newer transitive copy is extracted in glob order it shadows
the 2025.3 libraries and the helper again fails as "no device". Pin the three ABI packages and extract
them last.

### 1.4 Next diagnostic step if enumeration regresses

If SYCL again reports no device, this ordered checklist isolates the layer:

1. `ldd ~/.bongo/llama/b11223/sycl/llama-ls-sycl-device | grep 'not found'` — a missing
   `libsycl.so.8` means the ABI/provision gap (#1). A missing `libumf.so.1` is #3.
2. `ZEL_ENABLE_SYSMAN=1` and `ZES_ENABLE_SYSMAN=1` are sometimes suggested for Level Zero
   introspection; they did **not** matter here. The real reference is `ONEAPI_DEVICE_SELECTOR`;
   `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` is the useful filter once the adapter loads.
   `SYCL_CACHE_*` only affects JIT cache location, not enumeration.
3. Run `llama-ls-sycl-device` with `SYCL_PI_TRACE=1` to see whether the Level Zero UR adapter
   loads; an adapter load failure points at UMF/SONAME, while a load with zero devices points at
   IGC/NEO.
4. Check that `~/.bongo/runtime/.bongo-provisioned` records the pinned package set, and that
   `libsycl.so.8` exists (BAS-57 added an ABI guard that warns when it does not).

**Note for the reader:** the BAS-57 fix lives on the `BAS-48-project-setup` branch
(`0d70f00`, `b830d45`, `1119984`) and is **not yet merged** into `BAS-62-improve-speed-architecture`.
Until it is, the research branch's `bongo.sh` still contains the older provisioning and will fall back
to Vulkan on a machine with the shared runtime in its current state.

---

## 2. SYCL op coverage for the ops bongo needs

Derived from the `qwen4exp` graph (`src/models/qwen4exp.cpp`, `src/llama-graph.cpp`) and the SYCL
dispatch switch in `ggml/src/ggml-sycl/ggml-sycl.cpp`.

| bongo op | Where qwen4exp uses it | SYCL backend | Notes (SYCL) |
| --- | --- | --- | --- |
| **MoE expert matmul** (`MUL_MAT_ID`) | `build_moe_ffn` → `ggml_mul_mat_id` for `ffn_{gate,up,down}_exps`, 512 experts × 10 active × 48 layers | ✅ | Dedicated quantized MoE kernels: `launch_mul_mat_vec_q_moe<... IQ2_XS ... vec_dot_iq2_xs_q8_1_moe>` (`mmvq.cpp`), plus `ggml_sycl_mul_mat_id_mmvq_fused`. dp4a-based. **But** it forces a blocking host wait and disables SYCL graphs — see §3. |
| **i-quant dequant GEMV / MMVQ** | all routed-expert and attention weights at `IQ2_XS`/`IQ3_XXS` | ✅ | MMVQ supports `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`. `IQ2_XXS` is deliberately excluded from a path in the dispatch. |
| **i-quant mat-mat (prefill, integer MMQ)** | same weights, batch > 1 | ⚠️ disabled | `ggml_sycl_supports_mmq()` is hard-coded to `return false` in `b11223` ("TODO: accuracy issues in MMQ"). Prefill dequantizes / uses the generic SYCL + oneDNN path. |
| **flash attention** | 12 full-attention layers, `GGML_OP_FLASH_ATTN_EXT` | ✅ | `fattn.cpp` + `fattn-tile/vec/onednn/mkl`. `GGML_SYCL_ENABLE_FLASH_ATTN=1`, `GGML_SYCL_FA_ONEDNN=1`, `GGML_SYCL_ENABLE_MKL_FA=1` in the prebuilt. |
| **GDN / linear attention** | 36 `linear_attention` layers: `llm_build_delta_net_base` → `GGML_OP_GATED_DELTA_NET`; `ggml_ssm_conv` → `SSM_CONV` | ✅ | `gated_delta_net.hpp`, `ssm_conv.cpp`, `ssm_scan.cpp`; graph fusion for `SSM_CONV + ADD + SILU` and `GATED_DELTA_NET + CPY` state snapshots exists. |
| **QSA / lightning indexer** | `build_attn_qsa`, `build_qsa_top_k`, indexer path | ✅ | `GGML_OP_LIGHTNING_INDEXER` is in the SYCL dispatch list. |
| **rope** | `ggml_rope_multi` (partial rotary 0.25) | ✅ | `rope.cpp`. |
| **router top-k** | `ggml_top_k` (10 of 512) | ✅ | `topk-moe.cpp`, `topk-radix.cpp`. |
| **PLE / n-gram rows** | 7 × `ggml_get_rows` (`per_layer_token_embd`, lazy) | ✅ | `GGML_OP_GET_ROWS`. Lazy tensor rows are read on the CPU host buffer type by the loader, not a GPU op. |
| **Sampling** | token sampling | ✅ (CPU) | Sampling lives in `llama-sampling` on the CPU in all backends; not a GPU coverage question. |

Reference — Vulkan implements the same op set (`MUL_MAT_ID`, `FLASH_ATTN_EXT`, `GATED_DELTA_NET`,
`SSM_CONV`, `SSM_SCAN`, `LIGHTNING_INDEXER`, `ROPE`, `TOP_K`, `GET_ROWS`, `CUMSUM`). **No op bongo
needs is missing on either backend.**

### 2.1 Cross-check with Strata's kernel list

Strata (`include/strata/kernels/`, `src/kernels/`) is organised as: `native_moe`, `native_mmvq`,
`native_flash_attn`, `native_gdn`, `native_gdn_preprocess`, `native_gr_norm`, `native_gr_postops`,
`native_qsa`, `native_qsa_indexer`, `native_qsa_score`, `native_rope`, `native_router`,
`router_top10`, `iq_kernels`, `fused_gdn`, `fused_gr`, `mrope`, `kv_q4`, `kv_q8`, `kv_stream`, `ple`,
`ngram`, `s2_gemv`, `s2_gemv_q8`, `s2_expert_grouped`, `shared_expert`, `quantize_act`, `sampler`,
`verify_kernels`.

Every functional family maps to an op llama.cpp already implements on both SYCL and Vulkan. Strata's
edge is **not** a kernel that llama.cpp lacks; it is scheduling and fusion quality
(`strata-ninfer-recon.md` §2.5; [`strata-architecture.md`](strata-architecture.md) §TL;DR quantifies it:
per-layer graph replay 1.51x, quantized-weight expert matmul 1052→1130 prompt tok/s, VRAM expert tier
−2.7 ms/token, PCIe expert ring 1130→1290, speculation 1.6-1.8x). That is exactly the class of work §7
scopes. In particular, Strata's #1 and #2 multipliers both map onto **llama.cpp backend work**, not
to a new engine: graph replay is `ggml_sycl_graph` (which `MUL_MAT_ID` currently disables, §4), and
quantized-weight matmul is `mmq.cpp`/`mmvq.cpp` (whose integer mat-mat path is currently off).

---

## 3. Vulkan vs SYCL — capability and the gap

The published tiers are `IQ2_XS` (recommended) and `Q2_0`; expert weights dominate both. The
task brief asks whether SYCL/oneAPI exposes int8 / dp4a dot-product for quantized MoE on
Battlemage. **Yes, and llama.cpp's SYCL backend uses it — for decode.** But it is not the whole
story.

### 3.1 Capability table (for the ops in §2)

| Capability | Vulkan (Mesa ANV) | SYCL (Level Zero / oneAPI 2025.3) |
| --- | --- | --- |
| `MUL_MAT_ID` for `IQ2_XS` | ✅ DMMV dequant path (`mul_mat_vec_iq2_xs_*`) | ✅ MMVQ + dp4a (`mul_mat_vec_q_moe` / `vec_dot_iq2_xs_q8_1_moe`), fused variant |
| integer mat-mat (MMQ) for `IQ2_XS` | ❌ no `matmul_iq2_xs` shader | ❌ `ggml_sycl_supports_mmq()` returns `false` for **all** types in `b11223` |
| integer MMQ for k-quants / Q2_0 / Q4-8 / IQ4 | ✅ (`matmul_*_q8_1`, incl. `MUL_MAT_ID` variants) | ❌ disabled globally (reorder MMVQ exists for Q1_0/Q4_0/Q8_0/Q2_K..Q6_K) |
| flash attention | ✅ `flash_attn_*` shaders | ✅ SYCL FA, oneDNN and MKL FA variants |
| GDN / `GATED_DELTA_NET`, `SSM_CONV` | ✅ | ✅ (+ fusion) |
| bf16 / fp16 / coopmat | ✅ (`KHR_coopmat` on BMG G31) | ✅ (XMX / oneDNN / MKL) |
| graph capture / reduced dispatch overhead | not applicable (no graph replay in the Vulkan backend) | **disabled for this model**: `MUL_MAT_ID` does a blocking host wait, so `ggml_sycl_graph` refuses the graph (`ggml-sycl.cpp` ~L6183-6198). Strata measures graph replay at **1.51x** (`strata-architecture.md`). |
| dp4a (int8 dot) | ✅ `int dot: 1` (ANV reports int8 dot) | ✅ `dpct::dp4a` throughout `vecdotq.hpp` |

Two structural facts matter more than any single number:

- **Prefill does not use integer tensor cores for `IQ2_XS` on either backend.** SYCL's MMQ is off
  globally; Vulkan has no `IQ2_XS` matmul shader. So Strata's headline "MMQ instead of
  dequantize-to-FP16" win (`strata-ninfer-recon.md` §2.5) is *available in principle on Battlemage
  via SYCL*, but **not wired up in `b11223`**. That is the single most concrete engine-level gap.
- **`Q2_0` (the BAS-59 tier) has a Vulkan integer MMQ kernel but no SYCL one.** If the tier choice
  moves to `Q2_0`, Vulkan gains an integer prefill path immediately; SYCL does not. Tier and backend
  should be chosen together.

### 3.2 Why the Vulkan path was thought slow

The historical "Vulkan is slow" reading came from the `n=12` device-loss and from comparing against
Strata's CUDA numbers, not from a clean per-backend measurement. The honest decomposition is:

- The 6-8x gap vs Strata (`strata-ninfer-recon.md` §1) is a **CUDA vs generic-backend** gap (MMQ,
  overlap, speculation), not a Vulkan-specific defect.
- For `IQ2_XS` specifically, neither Vulkan nor SYCL avoids dequantize-to-FP16 in prefill.
- At 128K the binding cost is long-context attention/KV plus CPU-expert traffic, which both backends
  pay (`expert-placement.md`).

---

## 4. Measurement: SYCL vs Vulkan on this box

Method: `llama-bench` from the same `b11223` build, identical flags on both backends —
`-m <IQ2_XS shard 1> -ngl 99 -ncmoe 16 -fa on -ctk q8_0 -ctv q8_0`, Vulkan pinned with `-dev Vulkan1`.

| test | SYCL | Vulkan | ratio |
| --- | ---: | ---: | ---: |
| `pp512` (prefill), `-r 3` | **74.94 ± 13.15** tok/s | 54.03 ± 24.76 tok/s | SYCL 1.39x |
| `tg128` (decode), `-r 3` | 5.34 ± 0.16 tok/s | **9.51 ± 0.94** tok/s | Vulkan 1.78x |
| `tg256` (decode-only), `-r 3` | 4.86 ± 0.83 tok/s | 7.49 / 8.29 tok/s (two runs) | Vulkan ~1.6x |

**Direction is consistent across three independent pairs: SYCL is ~1.4x faster in prefill and ~1.6x
slower in decode.** The decode gap has low variance (SYCL ±0.16), so it is not pure noise; the prefill
gap has high variance (±13 / ±25) because the host was under memory/IO pressure.

### Caveat — read this before quoting the numbers

The reference box is shared with other agents. Two long, I/O-heavy `route_capture` jobs (each loading
the 63 GiB model) ran concurrently with most of these measurements, thrashing the page cache on a
30 GiB host. **The absolute t/s are depressed well below the warm server numbers** (Stage 0 recorded
231 prompt / 17.7 output tok/s at 4K on Vulkan with the same flags). The *ratios* are the usable
signal; the absolutes are not. A clean, quiet-window harness run is required to replace them (see §7).

### Plausible mechanism for the decode deficit

`ggml_sycl_mul_mat_id()` performs a blocking `stream->wait()` after copying the expert `ids` to host.
Because the MoE uses `MUL_MAT_ID` on every layer, `ggml_sycl_graph` disables SYCL graph capture for the
whole graph (`ggml-sycl.cpp` logs "disabling SYCL graphs due to unsupported node type MUL_MAT_ID").
Without graph capture, every op is an individual Level Zero submission with host-side overhead. This
is a concrete, fixable engine-level cost — but fixing it means changing llama.cpp's SYCL backend, not
writing a new engine from scratch. The companion R-task micro-probe
`bench/micro/levelzero_probe.py` (Level Zero command-list submission cost vs N separate submissions,
plus the USM doorbell handoff) is the right instrument to size that overhead before any patch.

---

## 5. Vulkan device-loss root cause

The `n=12` 128K failure from `expert-placement.md`:

```
decode() failed: vk::Queue::submit: ErrorDeviceLost
```

is **VRAM exhaustion under a live workload**, not a driver bug:

- After load, `n=12` sits at 31.37 GiB; at 128K the KV cache and compute buffers push peak VRAM to
  **31.85 GiB** against **31.92 GiB usable** (32.00 GiB physical minus stolen), i.e. 0.07 GiB of
  margin. The Xe KMD then resets the context and the Vulkan queue returns `ErrorDeviceLost`.
- The all-GPU config (`n=0`) fails earlier, at model load, on a 0.78 GiB allocation — the same ceiling.
- The shipped default (`--n-cpu-moe 16`) leaves ~2.6 GiB of margin and completes 128K with a passing
  needle, so the baseline is correct; only the aggressive placement is unsafe.

This is a **capacity-boundary** device loss. It is not the SYCL/NEO enumeration problem, and it is not
reproducible on a correct config. The guardrail is the measured feasibility edge (between `n=12` and
`n=16`), already recorded as a BAS-53 follow-up.

---

## 6. Build path on Fedora 44 (reproducible)

The shipping path uses the pinned prebuilt (`llama-b11223-bin-ubuntu-sycl-fp16-x64`, oneAPI 2025.3.3
ABI) plus a user-local runtime — no compiler install needed. A from-source build additionally needs
the DPC++ compiler:

```sh
# 1. Install oneAPI DPC++/C++ (Base toolkit or Deep Learning Essentials) and source setvars.
#    On Fedora there is no distro package for the compiler; use the Intel oneAPI repo
#    (https://yum.repos.intel.com/oneapi). OneAPI 2025.3.3 is the verified release.
source /opt/intel/oneapi/setvars.sh

# 2. Configure and build (upstream docs/backend/SYCL.md "II. Build llama.cpp"):
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j

# 3. Pin the source revision (bongo pins the commit, not a tag):
#    git -C llama.cpp checkout 4da6337767f973e2b4d0797e5b323d77d8565e4a
```

Key CMake flags and defaults for this backend:

- `-DGGML_SYCL=ON`, `-DGGML_SYCL_F16=ON` (FP16 is the recommended/default performance path).
- `-DGGML_SYCL_TARGET=INTEL` (default), `-DGGML_SYCL_DNNL=ON` uses oneDNN GEMM.
- Runtime env knobs seen in the prebuilt banner: `GGML_SYCL_ENABLE_FLASH_ATTN=1`,
  `GGML_SYCL_ENABLE_DNN=1`, `GGML_SYCL_FA_ONEDNN=1`, `GGML_SYCL_ENABLE_MKL_FA=1`,
  `GGML_SYCL_ENABLE_GRAPH=0` (graphs are off by default), `GGML_SYCL_ENABLE_VMM=1`.

**Cost/effort:** a from-source DPC++ build is a ~several-GB toolchain install and a long first `icpx`
compile. It buys the ability to change kernels; for the baseline it buys nothing over the pinned
prebuilt. Recommendation: keep the pinned prebuilt for the product path, and only install DPC++ on a
dedicated build/bench host when someone actually intends to modify the SYCL backend.

**Realism for a small team:** adding *one* kernel to llama.cpp's SYCL backend (e.g. an integer
`IQ2_XS` MMQ, or removing the `MUL_MAT_ID` host wait) is a bounded, reviewable change in one file,
using patterns that already exist (`mmq.cpp`, `mmvq.cpp`, the Q2_K/Q4_K reorder MMVQ). Writing a
*whole engine* in SYCL from scratch is the multi-month project `strata-ninfer-recon.md` warns about,
and it re-implements ops that are already correct.

---

## 7. Verdict and the decisive experiment

**Go, conditionally, on the llama.cpp SYCL backend as the primary baseline** — enumeration is fixed,
op coverage is complete, and SYCL prefill is measurably faster on the tested config.

**No-go on a from-scratch, bongo-owned Intel-native (SYCL) engine** as the Stage 1/2 answer. Rationale:

1. There is **no coverage gap** to justify a new engine; every op is implemented. The measured gap is
   kernel/schedule quality, which is reachable by patching llama.cpp's SYCL backend.
2. The un-realised win (integer MMQ for `IQ2_XS`, dp4a MoE prefill) is a **bounded backend patch**, not
   an engine.
3. On the measured config, SYCL decode is *slower* than Vulkan, so "Intel-native" is not
   automatically faster; the current bottleneck at 128K is attention/KV and CPU-expert traffic, which a
   new engine would pay too.
4. Budget is better spent on the levers `strata-ninfer-recon.md` §4 identifies: schedule/speculation
   (suffix/ngram drafter), PLE prefetch, and backend selection per tier — not on re-implementing GDN
   and flash attention in SYCL.

**Decisive experiment (run this in a quiet window before any further engine decision):**

> Start the bongo server on each backend with the shipped flags (`--n-cpu-moe 16 --flash-attn on
> --cache-type-k q8_0 --cache-type-v q8_0`, `IQ2_XS`) and run `bench/harness.py` warm at 4K and
> **128K**. Compare **128K prefill tok/s / TTFT** and **128K decode tok/s** between `SYCL0` and
> `Vulkan1`, three repeats each.

Decision rule: if SYCL holds ≥1.3x on 128K TTFT and stays within ~10% on 128K decode, make SYCL the
default and close the engine question. If SYCL is slower on TTFT or materially slower on decode,
keep Vulkan and spend the budget on speculation/schedule work. The `--n-cpu-moe 16` config is the one
to run because it is the shipped default and the only config that is simultaneously 128K-safe
(`expert-placement.md` §"What fails, and how").

Today's data predicts a **split decision** — SYCL wins TTFT, Vulkan wins decode — so the result
depends on which metric the product values more (TTFT is the known pain: ~16 minutes at 128K). That
is exactly why this must be measured, not assumed.

---

## Reproduce

```sh
# Runtime (user-local, no root): see §1.2. Then:
R=~/.bongo/runtime
export LD_LIBRARY_PATH="$R/opt/intel/oneapi/redist/lib:$R/opt/intel/oneapi/umf/1.0/lib:$R/usr/lib64/llvm15/lib:$R/usr/lib64"
export ZEL_LIBRARY_PATH="$R/usr/lib64"
~/.bongo/llama/b11223/sycl/llama-ls-sycl-device        # must list the B70

M=~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
cd ~/.bongo/llama/b11223/sycl
./llama-bench -m "$M" -ngl 99 -ncmoe 16 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3
cd ~/.bongo/llama/b11223/vulkan
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d \
  ./llama-bench -m "$M" -ngl 99 -ncmoe 16 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3 -dev Vulkan1
```

Failure-mode repro (on a scratch runtime copy; restore afterwards):

```sh
# NEO abort (missing IGC SONAME symlink, i.e. raw cpio extract):
mv $R/usr/lib64/libigc.so.2{,.hidden} && llama-ls-sycl-device   # -> Abort ... resource_info.cpp
# zero devices (missing UMF):
mv $R/opt/intel/oneapi/umf/1.0/lib/libumf.so.1{,.hidden} && llama-ls-sycl-device
#   -> sycl::_V1::exception: No device of requested type available
```

## Limitations

- **Contended host.** The A/B numbers in §4 were taken while other agents ran model-loading jobs;
  absolute t/s are not product numbers. Ratios only.
- **`llama-bench` is not the product path.** It excludes the server's batching and warm-up. The
  decisive run is the server harness (§7).
- **One tier.** Measurements use `IQ2_XS`. `Q2_0` has different MMQ coverage (Vulkan has it, SYCL
  does not) and should be measured separately.
- **One config.** `--n-cpu-moe 16` only. A different CPU/GPU split changes the CPU/GPU balance and may
  change which backend wins.
- **No from-source SYCL build was attempted.** §6 documents the path; it was not executed on this box
  (that would install a multi-GB DPC++ toolchain, which the task constrained to a noted, user-local
  change).
- **No 128K SYCL run was completed**, because the shared GPU was busy for the whole window. The 128K
  TTFT/decode comparison remains the open decisive measurement.
