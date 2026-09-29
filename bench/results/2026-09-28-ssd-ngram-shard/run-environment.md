# R5 SSD/PLE shard — run environment

Reference box; measured 2026-09-28T07:15:44Z UTC.

## Storage

| Property | Value |
| --- | --- |
| Device | `KIOXIA-EXCERIA G3 SSD` |
| Firmware | `EVFATR.0` |
| Filesystem | `xfs` on `/dev/mapper/fedora-root` |
| Kernel | `7.0.13-200.fc44.x86_64` |
| Logical / physical sector | 512 / 512 B |
| I/O scheduler | `none` |
| read_ahead_kb | 128 |
| CPU / RAM | AMD Ryzen 7 9700X 8-Core Processor / 30Gi |

## Model

| Property | Value |
| --- | --- |
| File | `~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` |
| Size | 39,788,473,344 B (37.06 GiB) |
| PLE tensor | `per_layer_token_embd.weight`, IQ4_NL, `[160, 320001536]` |
| PLE size | 28,800,138,240 B (26.82 GiB) |
| PLE file offset | 361,831,168 |

## Contention

The box is shared with other agents. The latency sweeps in `ssd-ngram-latency-run*.json` were taken while a co-resident `llama-bench` + `route_capture` pair on the same IQ2_XS file held the NVMe busy.
Each JSON records the kernel PSI IO pressure (`/proc/pressure/io`) at the start and end of the run, so a contended run is self-documenting.
