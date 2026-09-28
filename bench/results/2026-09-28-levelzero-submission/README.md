# Level Zero submission cost and USM doorbell on the Arc Pro B70 (BAS-70)

Raw evidence for [BAS-70](/BAS/issues/BAS-70) — the two Arc-side facts that decide whether
Strata's per-layer graph capture and its device-wait doorbell are worth porting (U4 and U5 in
[`docs/research/strata-architecture.md`](../../../docs/research/strata-architecture.md) §7).
Analysis and verdict: [`docs/research/levelzero-submission.md`](../../../docs/research/levelzero-submission.md).

- Box: Intel Arc Pro B70 (Battlemage G31, PCI `8086:e223`), Fedora 44 Server, kernel
  `7.0.13-200.fc44.x86_64`, `xe` driver.
- Driver stack: NEO/Level Zero `libze_intel_gpu.so.1.15.38646`, loader
  `libze_loader.so.1.28.6`, Level Zero API `1.15`, IGC `libigc.so.2.36.3+0`.
  **No SYCL runtime or DPC++ compiler is installed on the box** (`libsycl.so` absent), so the
  probe drives `ze_api.h` through Python ctypes and assembles its two micro kernels as SPIR-V
  words ([`bench/micro/spirv_kernels.py`](../../../bench/micro/spirv_kernels.py)).
- Reproduce with one command:

  ```sh
  python3 bench/micro/levelzero_probe.py \
      --out bench/results/2026-09-28-levelzero-submission \
      --reps 1000 --warmup 50 --nodes 0,1,4,16,43,200 \
      --u5-reps 30 --u5-timeout-ms 300 --hold-ms 2 --spin-budget 1000000
  ```

  The script re-execs itself once to put the user-local runtime on `LD_LIBRARY_PATH`
  (`$BONGO_HOME/runtime/usr/lib64`, `.../llvm15/lib`, `.../opt/intel/oneapi/redist/lib`) and to
  set `ZEL_LIBRARY_PATH`; NEO dlopens IGC, and IGC dlopens LLVM, so all three are required.

## Files

| file | contents |
| --- | --- |
| `raw/levelzero-probe.json` | every raw sample (ns) plus the environment record and method description |
| `probe-run.log` | stdout/stderr of the recorded run |
| `raw/watchdog-spin.txt` | bounded device-spin wall times for 25M / 100M iterations |
| `raw/xe-journal.txt` | `xe` kernel messages for the session (engine resets, GPU faults) |

## U4 — what one submission costs (reps=1000, warmup=50)

One submission = one `zeCommandQueueExecuteCommandLists`. The "graph" analogue is a *closed*
command list holding N launch nodes, so the per-node append cost is paid once at capture time,
exactly as `cudaGraphExec_t` replay does.

Warm medians, µs:

| N nodes | batched submit | batched submit+complete | per-node submissions | per-submission submit+complete | capture: append µs/node |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.343 | 6.742 | — | — | — |
| 1 | 1.964 | 9.998 | 1.994 | 9.227 | 1.362 |
| 4 | 2.044 | 6.453 | 1.413 | 5.330 | 1.010 |
| 16 | 1.372 | 6.251 | 1.402 | 5.109 | 0.902 |
| 43 | 1.383 | 7.765 | 1.392 | 5.099 | 0.886 |
| 200 | 1.392 | 17.192 | 1.413 | 5.160 | 0.881 |

- **One submission costs ≈1.4 µs** and is flat in N. Cold and warm are the same
  (N=200: cold median 1.393 µs / warm median 1.392 µs).
- **Submitting N nodes one at a time costs N × ≈1.4 µs**, so there is **no crossover**: the
  per-submission cost is what batching removes, not a per-node cost.
- Appending a node into a closed list (the capture cost) is 0.88–1.36 µs/node and is paid once.
- An immediate command list — where every append *is* a submission — measures 1.54–1.66 µs per
  appended node.

Strata's motive for the whole-token graph was 96 submissions/token at ~0.3–0.4 ms each
(29–38 ms/token, a WDDM property). On this stack 96 submissions cost **0.13 ms/token** host-side
(0.48 ms/token if each is also synchronised), i.e. ~250x cheaper. Synchronisation is not required
per layer anyway: order within one queue is implicit.

## U5 — USM flag visibility in both directions (reps=30)

`k_store`/`k_doorbell` write an `int` in a `zeMemAllocShared` allocation; the host spins on it and
timestamps with `time.perf_counter_ns`. The doorbell kernel is bounded (1,000,000 iterations ≈
34 ms) so a release the device never sees cannot wedge the GPU; it records what it last read.

Device → host (kernel writes, host spins, no Level Zero call in the spin loop):

| measurement | observed | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| pure host spin | 30/30 | 36.7 µs | 35.6 µs | 40.5 µs |
| host spin + `zeFenceQueryStatus` each iteration | 30/30 | 36.6 µs | 32.5 µs | 41.1 µs |
| submission start → completion | — | 40.2 µs | 35.9 µs | 41.7 µs |

The driver entry changes nothing. The decisive measurement is the doorbell, where the kernel keeps
running for 34 ms: the host saw the publish at **33.7 µs** — three orders of magnitude before the
kernel finished — so a device store to USM shared memory **is** host-visible with no submission.

Host → device (host releases a spinning kernel by writing `ack`):

| release path | device saw the release | times |
| --- | ---: | ---: |
| ack set before the launch (control) | **30/30** | — |
| no release | 0/30 | spins full 34 ms budget |
| plain host store, no driver call | **0/30** | spins full budget |
| store + `zeFenceQueryStatus` | **0/30** | spins full budget |
| store + a new queue submission | **0/30** | spins full budget |
| store into a `BIAS_UNCACHED` shared allocation | **0/30** | spins full budget |

The control proves the handshake wiring is correct; the negative is therefore about coherence.
A host store to USM shared memory becomes visible to the device **only at a submission boundary**,
never mid-kernel, and neither a driver entry, an extra submission, nor an uncached allocation
changes that.

## Device-side spin and the `xe` hang detector

A bounded 1,000,000-iteration volatile spin takes 34.4 ms (≈34 ns/iteration — the visible cost of
re-reading host memory from the device). Isolated long spins completed normally in this session
(25M → 890.7 ms, 100M → 3429.4 ms, `sync_rc=0`), but the session journal records **three** `ccs`
engine resets (see `raw/xe-journal.txt`):

- `08:32:54` — `Engine memory CAT error [18]` then `Timedout job ... in python3` and an engine
  reset. This was the run that launched a kernel with an unset (garbage) pointer argument, which
  also returned `ZE_RESULT_ERROR_DEVICE_LOST` (`0x70000001`).
- `08:40:41` — five `Fault response: Unsuccessful -EINVAL` faults, then an engine reset.
- `08:46:38` — five more `-EINVAL` faults and a reset, during a back-to-back
  `1M/10M/50M/100M` spin sequence in one context.

Conclusion to carry forward, stated as fact: long device-side spins are *possible* but the driver
does have a timeout path, and every GPU fault in this session ended in an engine reset and (once)
a lost device. A portable doorbell must be bounded and short, and must never launch a kernel with
an unset pointer argument.

## Toolchain findings (reproduced while building the probe)

These cost most of the session and are worth recording:

1. `zeInit(0)` returns `ZE_RESULT_ERROR_UNINITIALIZED` (0x78000001) on this box; the working entry
   point is `zeInitDrivers()` with `ze_init_driver_type_desc_t.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU`.
2. The loader finds `libze_intel_gpu.so.1` only with `ZEL_LIBRARY_PATH=<runtime>/usr/lib64`; and
   NEO only keeps IGC loaded when `llvm15/lib` is on `LD_LIBRARY_PATH` (otherwise the driver
   aborts in `gmm_helper/resource_info.cpp`).
3. A SPIR-V module whose kernel takes a `__global` pointer must use **storage class 5
   (`CrossWorkgroup`)**. IGC rejects anything else with *"Generic pointers are not allowed as
   kernel argument storage class"*.
4. `OpMemoryBarrier` inside a kernel makes IGC 2.36.3 abort with *"Internal Compiler Error:
   Segmentation violation"* and takes the process with it. `OpAtomicStore` and a volatile
   `OpStore` compile; the probe uses a volatile store.
5. `zeFenceCreate` takes the **command queue**, not the context; passing a context segfaults.
6. OpenCL is unusable on this box for the same reason Level Zero used to be: with
   `OCL_ICD_VENDORS` pointing at `intel-opencl/libigdrcl.so`, enumeration aborts. The probe
   therefore uses Level Zero only.
