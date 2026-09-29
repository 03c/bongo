# Capacity-sensitivity raw data (BAS-69)

Raw measurements behind [`docs/research/capacity-sensitivity.md`](../../../docs/research/capacity-sensitivity.md).
One IQ2_XS tier, llama.cpp `b11223` Vulkan, Arc Pro B70, Fedora 44. Every directory is one
`bench/run-capacity-sensitivity.sh` invocation.

| directory | memory cap | `--n-cpu-moe` | CPU experts GiB | contexts | status |
| --- | ---: | ---: | ---: | --- | --- |
| `mem16g-ncmoe16/` | 16 GiB (`memory.max`) | 16 (shipped) | 10.62 | 4096, 131072 | complete, needle pass |
| `mem16g-ncmoe24/` | 16 GiB (`memory.max`) | 24 | 16.19 | 4096, 131072 | complete, needle pass |
| `uncapped-ncmoe24/` | none | 24 | 16.19 | 4096, 131072 | same-protocol control for the row above, in its own scope |
| `preflight/` | 16 GiB | 16 | 10.62 | (plumbing check) | `systemd-run` scope + sampler smoke test |
| `preflight/uncapped-ncmoe24-agentcgroup/` | none | 24 | 16.19 | 4096, 131072 | **discarded control**: ran in the agent's own cgroup, so its 4K row competed with the agent host; superseded by `uncapped-ncmoe24/` |

Per run:

| file | contents |
| --- | --- |
| `matrix.json` / `matrix.md` | `bench/harness.py` output: prompt/output tok/s, TTFT, needle, peak VRAM, peak RSS |
| `raw/` | every individual request record (`iq2_xs-ctx<ctx>-r1.json`, `needle.json`) |
| `samples.jsonl` | `bench/capacity_sampler.py` ticks: `/proc/<pid>/io`, `/proc/<pid>/stat` faults, `VmRSS`/`VmHWM`, VRAM `fdinfo`, cgroup `memory.current`/`events`/`stat` |
| `memory.json` | aggregate of `samples.jsonl` (peaks, last cumulative counters) |
| `server-io.json` | `/proc/<pid>/io` at load / after warm-up / after the harness, plus `harness_exit` |
| `cgroup.json` | the `memory.max`, `memory.swap.max` and `memory.events` actually in force |
| `server-start.log`, `harness.log` | server launch and harness stdout/stderr |

The cap is enforced with a transient, reversible cgroup v2 scope:

```sh
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- ./bongo.sh ...
```

`preflight/` was the initial plumbing check; it ran contexts=1024 with a standalone 131072-token
needle, so it is kept as evidence that the scope and sampler work (the cgroup and I/O records are
valid) but its throughput numbers are not part of the study.

Reproduce:

```sh
./bench/run-capacity-sensitivity.sh 16 16
./bench/run-capacity-sensitivity.sh 16 24
./bench/run-capacity-sensitivity.sh none 24
python3 bench/capacity-model.py
```
