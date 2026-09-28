# Stage 0 baseline — run environment

Reference box, Intel Arc Pro B70, IQ2_XS, 131072-token context.

## How the server was started

The OpenAI-compatible endpoint was started with `bongo.sh` (single command),
using the system Mesa Vulkan driver and a rootless runtime prefix (no
oneAPI/SYCL runtime is available on this host, so the documented Vulkan
fallback from [ADR-0001](../../../docs/adr/0001-runtime-architecture.md) is in
use):

```sh
./bongo.sh --runtime dir --runtime-dir ~/.bongo/runtime-empty \
           --backend vulkan --detach --yes
```

`bongo.sh` pinned the Intel GPU explicitly:

```
--device Vulkan1
```

This pin matters on this host: the CPU's AMD iGPU also exposes a Vulkan
device, and llama.cpp otherwise selects `Vulkan0` (the iGPU), which makes the
server exit during model load. The fix is commit `707f4a5`
(`fix(bongo.sh): pin the Intel Vulkan device when another Vulkan device exists`).
`--device Vulkan1` is recorded in `matrix.json` under `server.flags`, because
the harness reads the running server's `/proc/<pid>/cmdline`.

Backend: **Vulkan** (Mesa ANV), `llama.cpp b11223`
(`4da6337767f973e2b4d0797e5b323d77d8565e4a`). The exact flag list and the
generated `bongo-config.json` are captured in the matrix.

## How the benchmark was run

```sh
./bench/run.sh --tier iq2_xs --repeats 3 --repeats-deep 1 --deep-threshold 32768
```

- 3 repeats at 1024 / 4096 tokens.
- 1 repeat at 32768 / 131072 tokens, because a 128K prefill is ~12-15 minutes
  on this configuration. `matrix.json.config` records both budgets.
- The 128K needle check is folded into the 128K context run: one prefill both
  measures throughput and proves recall.

Results: `matrix.json` (machine-readable, all raw runs) and `matrix.md`
(human-readable summary).

## Endpoint exclusivity

`llama-server` runs with `--parallel 1`, so only one request is processed at a
time. The run was taken with no other client using the endpoint. Concurrent
load would corrupt the client-side TTFT numbers (requests would queue), even
though the server-reported `prompt_ms` / `predicted_ms` remain valid.
