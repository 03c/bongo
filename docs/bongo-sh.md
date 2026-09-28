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
| `--backend NAME` | `auto` | `auto`, `sycl`, `vulkan`, or `cpu` |
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

ADR-0002 makes **llama.cpp SYCL** the baseline. `bongo.sh` selects SYCL when a SYCL device
is reported, then falls back to **Vulkan** (Mesa ANV), which ADR-0001/0002 name as the
documented alternative. `--backend cpu` is available for debugging.

The selected backend is printed, written to the generated config, and used to pick the
pinned prebuilt asset (`llama-<rev>-bin-ubuntu-{sycl-fp16,vulkan,x64}.tar.gz`).

### Known issue on the reference box (2026-09-27)

The Fedora `intel-level-zero` / `intel-opencl` build (NEO `26.22.38646.6`) aborts during
GMM initialisation on this host, so no SYCL device is enumerated even though the GPU is
visible and `xe` is loaded:

```
$ LD_LIBRARY_PATH=.../prefix/opt/intel/oneapi/redist/lib:.../prefix/usr/lib64 \
  llama-ls-sycl-device
Abort was called at 15 line in file:
/builddir/build/BUILD/intel-compute-runtime-26.22.38646.6-build/.../gmm_helper/resource_info.cpp
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  No device of requested type available.
```

The same abort occurs with the minimal set of Level Zero/NEO libraries and as root, so it
is not a library-shadowing or permission problem. Vulkan (Mesa `26.2.3`) detects the GPU
correctly:

```
deviceName = Intel(R) Graphics (BMG G31)   (DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
```

Until the NEO/GMM issue is resolved, run with `--backend vulkan` (or leave `--backend auto`
to fall back automatically). Track the driver fix separately from this script.

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

**On this model a restored slot is not reused.** `Swift-1.5-Qwen3.8-Flash-Next` is a hybrid
`qwen4exp` GGUF: 36 Gated-DeltaNet (linear/recurrent) layers and 12 full-attention layers, with no
SWA layers. `save` writes the slot's tokens plus its sequence state (KV + recurrent state), and
`restore` reads them back and reports `n_restored`; the bytes round-trip and are fast (see the
timings below). But the engine also needs a *context checkpoint* to resume a prefix on
hybrid/recurrent memory, and the server does not persist its checkpoint list in the slot file.
The next request therefore re-prefills the whole prompt and the log shows *"forcing full prompt
re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)"*.

Consequences and things that do **not** fix it on this model:

- `--ctx-checkpoints N` only bounds the in-RAM checkpoint list; the checkpoints are still not
  written to the slot file.
- `--cache-idle-slots` keeps an idle slot in the **in-RAM** prompt cache; it does not survive a
  process restart.
- `--swa-full` is rejected by the engine here (`swa_full is not supported by this model`), because
  this GGUF has no sliding-window layers.

So `--slot-save-path` persists the KV *bytes* cheaply, but on this model it does not turn a
restore into a cache hit. **In-session prefix reuse (no restart) is unaffected and remains the
product path.** Making restart survival a latency win needs the engine to persist context
checkpoints (engine work, tracked separately), or a non-hybrid model tier.

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
- `--check --backend sycl` fails with an actionable message (no SYCL device), and `--backend auto`
  falls back to Vulkan. See [BAS-57](/BAS/issues/BAS-57) for the driver fix.

The SYCL backend failure above is the outstanding blocker for the primary path.
