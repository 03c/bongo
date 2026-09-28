# `bongo.sh` — one-command setup

`bongo.sh` takes a clean checkout to a responding OpenAI-compatible endpoint with a
>= 131072-token context:

```sh
git clone https://github.com/03c/bongo && cd bongo && ./bongo.sh
```

It is the Stage 0 artifact from [ADR-0001](adr/0001-runtime-architecture.md): the pinned
llama.cpp SYCL baseline, plus a documented Vulkan fallback.

## What it does

1. **Detects** the OS and the Intel Arc GPU (`lspci`, `/dev/dri`, the `xe` driver).
2. **Provisions** the Intel compute runtime (Level Zero + oneAPI SYCL runtime) and the
   Vulkan loader/Mesa drivers when needed.
3. **Fetches** the pinned llama.cpp prebuilt binary (`b11223`,
   commit `4da6337767f973e2b4d0797e5b323d77d8565e4a`) for the selected backend.
4. **Downloads** the chosen GGUF tier from Hugging Face with HTTP range resume.
5. **Launches** `llama-server` with a >= 128K context, prints the exact flags, and writes
   the full run configuration to `$BONGO_HOME/run/bongo-config.{json,env}`.

## Requirements

- Intel Arc GPU (the reference target is the Arc Pro B70, PCI `8086:e223`) with the `xe`
  or `i915` kernel driver and a render node under `/dev/dri`.
- `curl`, `lspci`, `df`, and either `dnf` (Fedora) or `apt-get` (Ubuntu/Debian).
- Enough free disk for the tier (see below).
- `HF_TOKEN` if the model repository is gated.

## Options

Run `./bongo.sh --help` for the full list. The most-used options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--tier NAME` | `iq2_xs` | `iq2_xs`, `iq3_xxs`, or `q2_0` |
| `--gguf-dir DIR` | — | Use an existing download instead of downloading |
| `--ctx N` | `131072` | Context size (must be >= 131072) |
| `--backend NAME` | `auto` | `auto`, `sycl`, `vulkan`, or `cpu` (see [Backends](#backends)) |
| `--n-cpu-moe N` | per tier | Explicit number of MoE layers kept on the CPU |
| `--port N` / `--host H` | `8080` / `127.0.0.1` | Bind address |
| `--runtime MODE` | `auto` | `system`, `user`, `dir`, or `auto` |
| `--runtime-dir DIR` | `$BONGO_HOME/runtime` | Use a pre-provisioned runtime prefix |
| `--llama-rev REV` | `b11223` | Pin a llama.cpp release tag |
| `--llama-bin DIR` | — | Use an existing llama.cpp build directory |
| `--check` | — | Provision + verify the runtime, then exit |
| `--dry-run` | — | Print the plan and flags without changing anything |
| `--detach` | fore­ground | Start the server in the background |
| `--force` | — | Re-fetch / re-download even if files look complete |
| `--cache-prompt` / `--no-cache-prompt` | `--cache-prompt` | Prompt (prefix) caching. `--no-cache-prompt` forces cold prefill for A/B |
| `--slot-save-path DIR` | `$BONGO_HOME/run/slots` | Enable `POST /slots/{id}?action=save\|restore\|erase` under `DIR` |
| `--no-slot-save-path` | — | Disable the slot save/restore endpoint (engine default) |
| `--cache-idle-slots` / `--no-cache-idle-slots` | engine (on) | Save idle slots to the in-RAM prompt cache on a new task |
| `--ctx-checkpoints N` | engine (32) | Max context checkpoints per slot |
| `--save-slot-checkpoints` | — | Persist context checkpoints in the slot sidecar so a restored hybrid/recurrent slot reuses its prefix (BAS-86; default: off) |
| `--no-save-slot-checkpoints` | on | Disable checkpoint persistence (baseline) |
| `--no-warmup` | — | Do not send the post-load warmup request |

## Runtime provisioning

`--runtime system` installs distro packages with `sudo`. On Fedora it installs
`intel-level-zero`, `oneapi-level-zero`, `intel-opencl`, `clinfo`, and the oneAPI runtime
packages (`intel-oneapi-runtime-dpcpp-cpp`, `-mkl`, `-dnnl`, `-tbb`, `-compilers`,
`-openmp`, `-opencl`) from the Intel oneAPI repository. Ubuntu/Debian installs the
equivalent `intel-level-zero-gpu` / `level-zero` / `intel-opencl-icd` packages plus the
oneAPI apt repository.

`--runtime user` performs the same provisioning **without root**: it downloads the RPMs
and extracts them under `$BONGO_HOME/runtime`. This keeps the host clean and is what the
sandbox could exercise.

`--runtime dir --runtime-dir DIR` uses a pre-provisioned prefix. The script automatically
adds `DIR/opt/intel/oneapi/redist/lib`, `DIR/usr/lib64`, and `DIR/usr/bin` to the
environment.

## Backends

**Vulkan is the default.** ADR-0002 originally made llama.cpp SYCL the baseline on the expectation
that Intel's own backend would be the fast path; the M3.0 A/B ([BAS-72](/BAS/issues/BAS-72)) measured
the shipped 128K agentic profile on an idle Arc Pro B70 and found SYCL slower on every gate that
matters, so the preference was inverted and the ADR amended with the numbers:

| 128K, IQ2_XS, `--n-cpu-moe 16` | Vulkan1 | SYCL0 | SYCL / Vulkan |
| --- | ---: | ---: | ---: |
| prefill tok/s | 133.46 | 109.01 | 0.82x |
| decode tok/s | 8.00 | 4.69 | 0.59x |
| cold TTFT | 980705 ms | 1200549 ms | 1.22x the time |
| 512-token cached-turn TTFT | 4165 ms | 3925 ms | 0.94x the time |
| steady cached turn TTFT | 282 ms | 403 ms | 1.43x the time |

So `--backend auto` selects **Vulkan** when the llama.cpp build reports an Intel Vulkan device, and
falls back to **SYCL** (Level Zero) when it does not. `--backend sycl` still selects SYCL
unconditionally — you do not need an escape hatch to run it, and SYCL is worth choosing for a
short-context workload, where it is the faster of the two (4K prefill 276.59 vs 231.31 tok/s).
`--backend cpu` is available for debugging.

The discriminator is `llama-server --list-devices` (the same evidence used to pin `--device Vulkan1`),
not `vulkaninfo`, which is not installed on the reference box and cannot confirm a device even when
one is serving.

The selected backend is printed, written to the generated config, and used to pick the pinned
prebuilt asset (`llama-<rev>-bin-ubuntu-{sycl-fp16,vulkan,x64}.tar.gz`).

### The 2026-09-27 NEO/GMM abort is resolved

Earlier revisions of this page reported that the Fedora `intel-level-zero` build aborted in
`gmm_helper/resource_info.cpp` so no SYCL device was enumerated. The GPU was never the problem: the
abort came from two bugs in `bongo.sh`'s own `setup_runtime_env()`.

- `setup_runtime_env()` runs twice per invocation, and each run prepended the prefix's `usr/lib64` to
  `ZEL_LIBRARY_PATH`, producing `.../usr/lib64:.../usr/lib64`. **`ZEL_LIBRARY_PATH` must name exactly
  one directory** — a colon list makes the Level Zero driver enumerate zero devices, which surfaces as
  `sycl::exception: No device of requested type available`. A multi-entry value is now repaired and
  the assignment is idempotent.
- The IGC/LLVM libraries were missing from `LD_LIBRARY_PATH`, so the probe aborted in
  `gmm_helper/resource_info.cpp` *before* enumerating. `usr/lib64/llvm15/lib` is added when the
  prefix has it.

`llama-ls-sycl-device` now reports `[level_zero:gpu:0] Intel Arc Pro B70 Graphics` (Level Zero NEO
`1.15.38646+6`, oneAPI 2025.3.3) from a clean environment, and `./bongo.sh --check --backend sycl`
passes. Fixed in `5cc7df4`.

## Prefix-cache serving (the agentic path)

The product workload is an agentic session: a small first prompt that grows, where each turn
re-sends the whole history and only the new suffix is new. That path is **prefix reuse**, not
the cold prefill the Stage 0 baseline measured.

`llama.cpp` enables `--cache-prompt` by default. `bongo.sh` passes it explicitly (and records it),
because a reader should not have to know the engine default to reproduce a run:

- **`--cache-prompt`** (default) — an agentic turn that appends a suffix prefills only the delta.
  A turn that re-sends an unchanged prompt is a **full KV hit**.
- **`--no-cache-prompt`** — cold prefill every time. This reproduces the historical Stage 0
  baseline and is the A/B control.

### Slot KV persistence

The server is started with a slot-save directory by default (`$BONGO_HOME/run/slots`), so the
KV of a slot can be written to disk and read back:

```sh
# Save slot 0's KV, then restore it later (e.g. after an idle or a restart)
curl -s http://127.0.0.1:8080/slots/0?action=save    -H 'Content-Type: application/json' -d '{"filename":"session.bin"}'
curl -s http://127.0.0.1:8080/slots/0?action=restore -H 'Content-Type: application/json' -d '{"filename":"session.bin"}'
curl -s http://127.0.0.1:8080/slots/0?action=erase   -H 'Content-Type: application/json' -d '{"filename":"session.bin"}'
```

The endpoint is only available when a slot-save path is set; `--no-slot-save-path` restores the
engine default (disabled). The file lives *inside* the directory given to `--slot-save-path`.
Saving is explicit: `bongo.sh` never writes a KV file on its own.

**On this model a restored slot is not reused by default** (baseline). `Swift-1.5-Qwen3.8-Flash-Next` is a hybrid
`qwen4exp` GGUF: 36 Gated-DeltaNet (linear/recurrent) layers and 12 full-attention layers, with no
SWA layers. `save` writes the slot's tokens plus its sequence state (KV + recurrent state), and
`restore` reads them back and reports `n_restored`; the bytes round-trip and are fast (see the
timings below). But the engine also needs a *context checkpoint* to resume a prefix on
hybrid/recurrent memory, and the server does not persist its checkpoint list in the slot file.
The next request therefore re-prefills the whole prompt and the log shows *"forcing full prompt
re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)"*.

`--save-slot-checkpoints` (with `--slot-save-path`) is the attempted fix, and it is
**implemented but not yet measured**: it makes the server write a sidecar file `{filename}.ckpt`
alongside each saved slot, containing the slot's context checkpoint list (anchored position
ranges plus the recurrent/full-attention state bytes). On restore that sidecar is replayed into
the slot, so the engine's `cache_prompt` path can find a usable checkpoint anchor. Whether the
prefix is then actually reused is the open question — **BAS-86 owns the measurement and the AC is
not met until a restored slot shows `cache_n > 0`.** The sidecar is gated behind
`--save-slot-checkpoints`; when it is off (the shipped baseline) nothing changes and the engine
falls back to a full re-prefill on a restored slot exactly as before.

Two preconditions before this flag can be trusted on a normal box:

- It needs an engine built with the patch, not the stock `b11223` binary. Stock rejects the flag
  with `invalid argument: --save-slot-checkpoints`. The patch is pinned at
  [`tools/patches/slot-checkpoints-sidecar.patch`](../tools/patches/slot-checkpoints-sidecar.patch);
  `bongo.sh` accepts `--save-slot-checkpoints` but the flag only takes effect when the selected
  `llama-server` was built with that patch.
- The sidecar's size and the restore-reuse verdict are unmeasured; the 31K/256K numbers below are
  the **baseline (sidecar off)**.

Cost: each checkpoint sidecar is roughly the size of the checkpoint payload for the saved range
(≈390 MiB for the 31K prefix on this model; one sidecar per save/restore cycle).

#### Measured at full size (256K, q8 KV)

With the checkpoint sidecar **off** (the shipped baseline), the 262,144-token point on the
reference box, llama.cpp `b11223` Vulkan, `--n-cpu-moe 18`
([`bench/results/2026-09-28-prefix-cache-256k/`](../bench/results/2026-09-28-prefix-cache-256k/README.md)):

| action | 31K | 256K | rate |
| --- | ---: | ---: | ---: |
| `save` | 127.9 ms | 1,558.6 ms | 2.55 GB/s |
| `restore` | 264.4 ms | 11,521.1 ms | 0.35 GB/s |
| bytes | 585,931,732 | 3,980,332,632 | — |

**The restored KV is not reused at 256K either** — `cache_n: 0`, and the next request
re-prefilled all 261,997 tokens in 2,351 s. The same verdict at 31K, and worse in absolute
terms: at 256K the restore costs 11.5 s and then buys nothing, so losing the KV and
restoring it is a ~204x net loss. The conclusion is size-independent for this hybrid model.

Two things to size for before designing around restore on model swap:

- **Restore degrades with size much faster than save does.** 6.79x the bytes costs 43.6x
  the time, so restore is 6.4x worse than linear (2.22 GB/s → 0.35 GB/s per byte) where save
  only slows 1.8x (4.58 → 2.55 GB/s). A 256K swap pays ~11.5 s of restore before knowing it
  helped.
- **KV size is not simply proportional to token count** across these two points: 18,459
  B/token at 31K vs 15,190 B/token at 256K. One sample each, so treat it as an open question
  rather than a sizing rule.

If `--save-slot-checkpoints` is turned on, budget the sidecar for the whole range rather than
the 31K example above: the 256K slot file is already 3.71 GiB before any sidecar.

`--cache-idle-slots` (engine default: on) saves an idle slot to the **in-RAM** prompt cache when a
new task starts, so a second slot can reuse it without touching the disk. It needs the engine's
prompt-cache RAM budget (`--cache-ram`, default 8192 MiB). `--ctx-checkpoints N` bounds the number
of context checkpoints a slot may keep. Both are exposed for tuning and recorded when set; both
default to the engine value when omitted, so the shipped baseline is unchanged.

### Warmup

After the server reports healthy, `bongo.sh` sends one tiny request. The first request after a
model load pays the shader/kernel compile (measured at ~27 s on the reference box), so warming at
startup keeps that cost off the user's first turn. `--no-warmup` skips it for a cold-start A/B.

See [`docs/research/agentic-prefix-cache.md`](research/agentic-prefix-cache.md) for the measured
turn latency and the slot save/restore timings.

## Tiers and MoE placement

| Tier | Shards (GB) | Total | Default `--n-cpu-moe` |
| --- | --- | ---: | ---: |
| `q2_0` | 39.8 + 26.8 | 66.5 GB | 15 |
| `iq2_xs` | 39.8 + 28.4 | 68.1 GB | 16 |
| `iq3_xxs` | 39.8 + 36.2 | 76.0 GB | 22 |

The default `--n-cpu-moe` keeps roughly a ~24 GB VRAM expert budget over the model's 48
layers; it is an estimate and is meant to be tuned by measurement (see
[`docs/research/gguf-inventory.md`](research/gguf-inventory.md)). `--n-cpu-moe 0` keeps all
experts on the GPU, and `--cpu-moe` (pass all experts to the CPU) is available by passing a
large value.

## Generated config

Every successful run writes:

- `$BONGO_HOME/run/bongo-config.json` — machine-readable: llama.cpp revision **and commit**,
  backend, binary path, model repo/tier/shards, GPU, runtime, exact server flags, and the
  placement settings.
- `$BONGO_HOME/run/bongo-config.env` — the same values as a sourceable shell file
  (`BONGO_SERVER_FLAGS`, `BONGO_LLAMA_COMMIT`, ...).
- `$BONGO_HOME/run/llama-server.log` and `llama-server.pid`.

The benchmark harness (`bench/`) reads `bongo-config.json` to discover the model path and
server settings.

## Idempotency

Re-running `./bongo.sh` with the same arguments:

- skips the runtime install if it is already present. User-local runtimes write the
  sentinel `$BONGO_HOME/runtime/.bongo-provisioned`, so a second `--runtime auto` run
  prints `Existing SYCL runtime detected.` and does not re-download the runtime;
- reuses the downloaded GGUF shards (size-checked against the published total);
- if the endpoint is already healthy, prints the status and exits without starting a second
  server (use `--force` to restart).

## Cleanup

`./bongo.sh --uninstall` prints the exact paths to remove and the system packages that
`--runtime system` would have installed. `./bongo.sh --uninstall --yes` deletes
`$BONGO_HOME` and prints the exact system package removal command; it never removes system
packages on its own.

## Verification performed

Run on the reference box (Fedora 44, Arc Pro B70, `xe`) on 2026-09-27:

- `./bongo.sh --backend vulkan` served the **target** `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
  `IQ2_XS` model (two shards, 68,152,167,168 bytes combined) from `~/.bongo/models`.
- `GET /v1/models` returned 200 with `n_ctx = 131072` and `n_vocab = 248320`.
- `POST /v1/chat/completions` returned `BONGO_TARGET_OK` with `finish_reason = stop` (non-stream) and
  SSE chunks `stream` / `ing` / ` works` terminated by `data: [DONE]` (stream).
- A second `./bongo.sh` run skipped the download, detected the healthy endpoint, and exited without
  starting a second server (idempotent).
- The generated config records llama.cpp `b11223`
  (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), backend `Vulkan`, the two shards, and every flag
  (ctx 131072, `--jinja`, flash-attn `on`, KV `q8_0`, `--n-gpu-layers 99`, `--n-cpu-moe 16`).
- `--check --backend sycl` now passes against a clean environment (it failed before `5cc7df4`), and
  `--backend auto` selects Vulkan. See the [Backends](#backends) section above.

There is no outstanding backend blocker. SYCL is a supported, selectable backend; it is simply not
the measured fast path at 128K, so it is no longer the default.
