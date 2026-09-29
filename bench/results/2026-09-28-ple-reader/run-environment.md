# M3.4 PLE reader — run environment

Reference box; measured 2026-09-28.

## Storage and host

| Property | Value |
| --- | --- |
| Device | `KIOXIA-EXCERIA G3 SSD` (`nvme0n1`) |
| Filesystem | `xfs` on `/dev/mapper/fedora-root` |
| Kernel | `7.0.13-200.fc44.x86_64` |
| CPU / RAM | AMD Ryzen 7 9700X 8-Core / 30 GiB |
| GPU | Intel Arc Pro B70, Vulkan backend (`b11223`) |

## Model

| Property | Value |
| --- | --- |
| Shard | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` |
| Size | 39,788,473,344 B, mtime 2026-09-28T00:01:08+01:00 |
| Tensor | `per_layer_token_embd.weight`, IQ4_NL, `[160, 320001536]` |
| PLE size | 28,800,138,240 B (26.82 GiB) |
| PLE file offset | 361,831,168 |
| Engine revision | llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`) — not used by the reader; the reader reads the shard directly |

## Contention

The box is shared. The M3.4 reader run recorded kernel PSI IO
(`/proc/pressure/io`) in `raw/ple-reader.json`: `full_avg10` 1.04 -> 11.18.
The R5 device re-run (`raw/ssd-ngram-latency.json`) was quiescent
(`full_avg10` 1.51 -> 5.39) and reproduces R5's quiescent window
(262.4 us vs 259.1 us at token-window QD16). The reader and mmap paths in
`raw/ple-reader.json` were measured on the same device at the same time, so the
relative gain is valid even under contention.
