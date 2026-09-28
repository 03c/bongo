# Level Zero submission cost and USM flag visibility on the Arc Pro B70

Answers **U4** and **U5** of [`strata-architecture.md`](strata-architecture.md) §7 — the two
Arc-side facts that decide whether Strata's per-layer graph capture and its device-wait doorbell
are worth porting. Companion to [BAS-63](/BAS/issues/BAS-63) (Strata deep-dive) and
[BAS-57](/BAS/issues/BAS-57) (restoring Level Zero on the reference box).

- Date: 2026-09-28 · Issue: [BAS-70](/BAS/issues/BAS-70) · Branch: `BAS-62-improve-speed-architecture`
- Raw data and the exact command: [`bench/results/2026-09-28-levelzero-submission/`](../../bench/results/2026-09-28-levelzero-submission/README.md)
- Box: Arc Pro B70 (BMG G31, `8086:e223`), Fedora 44, kernel `7.0.13-200.fc44.x86_64`, `xe`;
  NEO/Level Zero `1.15.38646`, loader `1.28.6`, API 1.15, IGC `2.36.3+0`.
- **No SYCL runtime and no DPC++ compiler exist on this box** (`libsycl.so` absent). The probe
  therefore drives `ze_api.h` directly through Python ctypes — Level Zero is the layer both SYCL
  command graphs and llama.cpp's SYCL backend sit on — and assembles its two micro kernels as
  SPIR-V words ([`bench/micro/spirv_kernels.py`](../../bench/micro/spirv_kernels.py),
  driver [`bench/micro/levelzero_probe.py`](../../bench/micro/levelzero_probe.py)).

---

## TL;DR

| Question | Measurement on the B70 / Fedora 44 / Level Zero | Verdict |
|---|---|---|
| **U4** — cost of one graph launch / command-list submission | **1.34 µs** for one submission (closed list, N=0…200 nodes), flat in N; **1.4 µs** per node when submitted individually; capture (append) 0.88–1.36 µs/node paid once | Strata's per-layer graph mechanism **does not pay for launch overhead here**. 96 submissions/token = **0.13 ms/token** (0.48 ms/token if each is synchronised), not the 29–38 ms/token the WDDM motive was built on |
| **U5a** — is a device store to USM shared memory host-visible without a submission? | **Yes.** The host observed the doorbell publish at **33.7 µs** while the kernel was still spinning for 34 ms, with no Level Zero call in the spin loop | The device→host half of the doorbell works |
| **U5b** — does a host store to USM shared memory reach a spinning device without a submission? | **No — 0/30 in every variant**, including with an intervening driver entry, an intervening queue submission, and a `BIAS_UNCACHED` allocation. A control that presets the flag reads it 30/30 | The host cannot release the device mid-kernel. **The whole-token-graph device-wait design does not port as written** |

The two facts together cancel the mechanism: the token graph exists only to remove 96 submissions
per token, and those cost 0.13 ms/token here; the device-wait it needs in exchange cannot be
driven from the host.

---

## 1. Method

### 1.1 Why this is not the SYCL command-graph API

`docs/research/strata-architecture.md` §6 proposes `sycl::ext::oneapi::experimental::command_graph`
as the porting target. That API is not available on the reference box: `bongo.sh` provisions the
oneAPI *runtime*, not the DPC++ compiler, and no `libsycl.so` exists anywhere on the filesystem
(recorded in the run's `environment.runtime.libsycl`). The measurements below are taken one layer
lower, on Level Zero, which is what a SYCL command graph compiles down to (a closed
`ze_command_list` replayed by `zeCommandQueueExecuteCommandLists`). The one thing that is
therefore *not* measured is any host-side overhead the SYCL graph wrapper adds on top; see §5.

### 1.2 U4 design

- **Graph analogue** — one closed command list containing N `zeCommandListAppendLaunchKernel`
  nodes, replayed once per iteration with **one** `zeCommandQueueExecuteCommandLists`. Nodes are
  appended once before the loop, so this is a graph replay, not a rebuild.
- **Individual analogue** — N closed lists of one node each, executed once per iteration.
- N ∈ {0, 1, 4, 16, 43, 200}, 1,000 measured reps and 50 warmup reps per cell; the first 50 reps
  are reported separately as *cold*. 43 is Strata's measured block-graph node count; 200 is well
  past it. Kernel: `k_store(__global volatile int*, int)` — a real store to a shared-USM word.
- Timed on the submitting host thread with `time.perf_counter_ns()`: *submit* = return of
  `zeCommandQueueExecuteCommandLists`, *submit+complete* = after `zeCommandQueueSynchronize`.
- Also measured: the cost of appending a node into a not-yet-closed list (the capture cost), and
  the cost of an *immediate* command list, where every append is itself a submission.

Reproduce:

```sh
python3 bench/micro/levelzero_probe.py --out /tmp/ze --reps 1000 --warmup 50 \
    --nodes 0,1,4,16,43,200 --skip-u5
```

### 1.3 U5 design

Two kernels, both hand-assembled SPIR-V because no compiler is present:

- `k_store(flag, value)` — `flag[0] = value`, volatile, into a `zeMemAllocShared` word.
- `k_doorbell(flag, value, ack, result)` — `flag[0] = value`, then a **bounded** volatile spin on
  `ack` (1,000,000 iterations ≈ 34 ms), then `result[0] = ack[0]`. The bound matters: an
  unbounded device wait can wedge the GPU (§4).

Device → host: the host spins on `flag` with `ctypes` reads and no Level Zero call, with a
`zeFenceQueryStatus` per iteration as the CUDA-`cudaEventQuery` analogue, and against a plain
submit-and-synchronise baseline.

Host → device: the kernel spins; the host writes `ack = 1` at ~2 ms and the kernel reports whether
it ever saw it. Six release paths are compared, including one where the host sets `ack` **before**
the launch — the control that proves the wiring works.

Reproduce:

```sh
python3 bench/micro/levelzero_probe.py --out /tmp/ze --skip-u4 \
    --u5-reps 30 --u5-timeout-ms 300 --hold-ms 2 --spin-budget 1000000
```

---

## 2. U4 — submission cost

Warm medians in µs (full table and raw samples in the results directory):

| N nodes | batched submit | batched submit+complete | per-node submissions | per-submission submit+complete | capture: append µs/node |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **1.343** | 6.742 | — | — | — |
| 1 | 1.964 | 9.998 | 1.994 | 9.227 | 1.362 |
| 4 | 2.044 | 6.453 | 1.413 | 5.330 | 1.010 |
| 16 | 1.372 | 6.251 | 1.402 | 5.109 | 0.902 |
| 43 | 1.383 | 7.765 | 1.392 | 5.099 | 0.886 |
| 200 | 1.392 | 17.192 | 1.413 | 5.160 | 0.881 |

Three readings:

1. **The submission is ~1.4 µs and flat in N.** An empty list and a 200-node list cost the same to
   submit. Cold and warm are the same to two decimal places (N=200: 1.393 µs cold, 1.392 µs warm).
2. **There is no crossover node count.** Because the per-submission cost does not depend on N, the
   only thing batching saves is the *number* of submissions: `(N-1) × 1.4 µs`. A 43-node layer
   graph saves ~59 µs against 43 separate submissions; 48 such layers save ~2.8 ms/token.
3. **The capture cost is real but one-time.** Appending a node costs 0.88–1.36 µs and is paid at
   capture; replay pays nothing per node. An immediate list — the path where every append is a
   submission — measures 1.54–1.66 µs per node, which is the same order as a separate submission.

### 2.1 Amdahl, applied to Strata's own motive

Strata's motive is stated verbatim in `session.hpp`: *"Under WDDM every graph launch costs
~0.3-0.4 ms of submission latency, and the per-layer loop above makes 96 of them per token:
measured 48.6 ms/token for 12 ms of GPU kernels and 27 ms of pool."* The whole-token graph exists
to delete those 96 launches.

On this stack the same 96 submissions cost **96 × 1.4 µs = 0.134 ms/token** of host time
(0.48 ms/token if each is also synchronised — and synchronisation is not required, since order
within one queue is implicit). That is **~250x less** than the WDDM figure and ~0.1–0.4% of the
125 ms/token the Vulkan baseline actually spends at 128K. The exposed part of the launch cost on
the serial path is nil.

Note what this does *not* say: it does not say a graph is useless. The capture/append cost of
~1 µs/node is paid every token by an engine that enqueues its kernels per token, so a captured
graph can still recover the capture cost on ~2,000 nodes (a 43-node layer × 48 layers). But that is
a different, smaller claim than Strata's 1.51x, and it can be had with **one command list per
token** rather than 96 submissions, without needing the device-wait machinery at all.
[BAS-71](/BAS/issues/BAS-71) measured it: on 2,064 nodes the saving is **1.76 ms/token**, the
lower end of the 0.88–1.36 µs/node range, not the ~2.8 ms upper bound derived above.

---

## 3. U5 — the USM flag in both directions

### 3.1 Device → host works

| measurement (30 reps) | observed | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| pure host spin, no Level Zero call | 30/30 | 36.7 µs | 35.6 µs | 40.5 µs |
| host spin + `zeFenceQueryStatus` per poll | 30/30 | 36.6 µs | 32.5 µs | 41.1 µs |
| submit start → completion | — | 40.2 µs | 35.9 µs | 41.7 µs |

A driver entry in the poll loop buys nothing (36.57 µs vs 36.72 µs), and neither is needed. The
decisive case is the doorbell, where the kernel keeps running for 34 ms: the host saw the publish
at a median of **33.7 µs**, ~1000x before the kernel finished. That is the exact behaviour CUDA
needed `__threadfence_system()` plus a driver entry to achieve, and on this stack it is free.

### 3.2 Host → device does not work

| release path (30 reps each) | device saw it | kernel wall |
| --- | ---: | ---: |
| `ack` set **before** the launch (control) | **30/30** | 110.7 µs (exits immediately) |
| no release | 0/30 | 34.40 ms (full budget) |
| plain host store, no driver call | **0/30** | 34.40 ms |
| store + `zeFenceQueryStatus` | **0/30** | 34.41 ms |
| store + a new queue submission | **0/30** | 34.40 ms |
| store into `BIAS_UNCACHED` shared memory | **0/30** | 34.40 ms |

The control reads 1 on the first poll in every run, so the kernel, the argument wiring and the
`result` read-back are all correct. The negative is therefore about coherence, not about the test:
**a host store to USM shared memory is visible to the device only at a submission boundary, never
to a kernel that is already running.** Neither a driver entry, a second submission, nor an uncached
allocation changes that. The result is reproducible at 34.4 ms = the full spin budget, i.e. the
spinner genuinely never saw the release.

This is the mirror image of the CUDA finding recorded in `strata-architecture.md` §1.2 — CUDA
needed the driver entry to make *device* writes visible to the *host*; Level Zero gives that for
free but cannot deliver the *host* write into a running kernel by any of the four paths tested.

### 3.3 Consequence for the whole-token graph

Strata's `session_capture_token`/`session_run_token` structure is
`pre[l] → doorbell_wait → parts ← y_miss (H2D) → post[l]`, with a one-thread
`doorbell_wait_kernel` spinning on a host-written flag so the host makes no driver call in the
loop. On this stack that kernel would spin until it exhausted whatever bound it was given: the
host's release never arrives. A port must therefore keep the host in the loop — which is precisely
the `session_loop` shape Strata already has, and which §2 above shows is cheap here.

---

## 4. Device-side spin and the driver hang detector

A bounded volatile spin of 1,000,000 iterations takes **34.4 ms** — about **34 ns per iteration**,
the visible cost of re-reading host memory from the device. That number is also the spin's
resolution: a doorbell polled by the device is at best a 34 µs-scale handoff.

Isolated long spins completed cleanly in this session (25M iterations → 890.7 ms; 100M → 3429.4 ms,
`sync_rc = 0`), so long waits are not automatically fatal. But the session journal
(`raw/xe-journal.txt`) records **three** `ccs` engine resets:

- `08:32:54` — `Engine memory CAT error [18]`, then `Timedout job: seqno=… in python3`, then
  `Engine reset: engine_class=ccs`. This run also returned `ZE_RESULT_ERROR_DEVICE_LOST`
  (`0x70000001`); its trigger was a kernel launched with an unset pointer argument.
- `08:40:41` and `08:46:38` — five `Fault response: Unsuccessful -EINVAL` faults each, then resets.

**Design rule for a port:** every device-side wait must be bounded, every kernel argument must be
initialised before launch, and a device wait must never be the thing that guarantees forward
progress. `docs/research/strata-architecture.md` §6 already flags "must be validated for forward
progress"; this measurement says forward progress is not achievable in the host→device direction
at all.

---

## 5. What remains unmeasured

- **The SYCL `command_graph` wrapper itself.** No SYCL runtime or DPC++ compiler is installed, so
  the measurement is at the Level Zero layer underneath. A SYCL graph implementation could add
  host-side bookkeeping on top of the 1.4 µs; llama.cpp's SYCL backend does not use graphs today.
  Resolving it needs the oneAPI DPC++ compiler provisioned (the runtime alone is not enough —
  `libsycl.so` was absent from the user-local prefix).
- **Alternative host→device release channels.** Only a plain store, a driver entry, an extra
  submission and an uncached allocation were tried. A copy-engine H2D write
  (`zeCommandListAppendMemoryCopy`) or `zeCommandListAppendMemoryFill` into the ack word would be a
  *submission*, so it should work, but it reintroduces the submission the design was avoiding and
  was not measured.
- **Multi-queue and multi-token behaviour.** All measurements are on one compute queue, ordinal 0
  (queue groups: compute `0x7`, copy `0x2`, one queue each). Whether two queues overlap, and
  whether submission cost doubles or hides, is untested.
- **The exact `xe` hang timeout.** `/sys/kernel/debug` is root-only on this box, so the preemption
  timeout behind `Timedout job` was not read; only the symptoms are recorded.
- **The one anomaly seen.** An early 3-rep dry run of the pure host spin recorded 2 observed / 1
  timeout at a 500 ms timeout; the recorded 30-rep run is 30/30 at a 300 ms timeout. Treat the
  single timeout as unexplained noise or as a rare host-side cache-staleness event; it needs a
  dedicated high-rep run to settle, and it does not affect the doorbell conclusion, which is a
  0/30 negative in four independent release paths.

---

## 6. Verdict and next experiment

**Verdict.** Porting Strata's per-layer **graph** mechanism to this stack is not justified by its
stated motive: a Level Zero submission costs 1.4 µs, so 96 of them cost 0.13 ms/token, ~250x less
than the WDDM number the whole-token-graph work exists to avoid. Porting the whole-token **graph**
is worse than unjustified — it is impossible as designed, because its device-wait requires a host
store to reach a running kernel and that does not happen on this stack by any path tested.

What *is* worth keeping from the mechanism is the capture itself, for a different reason: an
engine that enqueues ~2,000 kernels per token pays ~0.9–1.4 µs each at append time, and a captured
list pays that once. That is a per-token argument for "one command list per token", not for
96 per-layer graphs, and it needs no doorbell.

**Measured on the B70 (BAS-71).** One captured command list per token was built and compared with
the same kernels appended per token
([raw](../../bench/results/2026-09-28-per-token-command-list/README.md), [BAS-71](/BAS/issues/BAS-71)).
At 2,064 nodes a captured list costs **0.129 ms/token** (one replayed submission) versus
**1.89 ms/token** rebuilt, so the capture saving is **1.76 ms/token** — the mechanism holds, at the
lower end of the 0.88–1.36 µs/node range rather than the ~1.4 µs upper bound. The per-node delta is
flat at 0.85–0.86 µs/node through 4,128 nodes; cold equals warm except a one-time ~3.6 ms first
launch. Caveat: the replayed list used fixed arguments; a real token needs per-token argument
values, so the saving is realisable only if those can be repointed without re-appending the nodes.
