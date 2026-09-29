# SSD random-4K probe on the reference box

Measurement-only probe feeding the SSD-tier arithmetic in
[`docs/research/moe-offload-landscape.md`](../../../docs/research/moe-offload-landscape.md)
([BAS-65](/BAS/issues/BAS-65)). It bounds the per-token cost of the 16-row n-gram/PLE gather
that a disk-resident second shard would pay. It does **not** replace the full PLE/SDD shard
benchmark reserved for R5: it only measures raw device latency, not the reader, cache, or overlap.

- Date: 2026-09-28
- Hardware: Intel Arc Pro B70 (32 GiB VRAM), AMD Ryzen 7 9700X, 30 GiB RAM, Fedora 44
- Storage: `nvme0n1` KIOXIA-EXCERIA G3 SSD (931.5 GB, firmware `EVFATR.0`), single device, no RAID
- File read: `~/.bongo/models/.../iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`
  (39.79 GB) — read-only, `O_DIRECT`, random 4 KiB offsets across the whole file

## Method

[`rand4k.py`](rand4k.py) opens the GGUF with `O_RDONLY | O_DIRECT`, issues `preadv` of 4 KiB into a
page-aligned `mmap` buffer at uniformly random 4 KiB-aligned offsets, and records per-read latency.
One warm-up pass of 64 reads precedes the measured pass. The same offsets are then replayed from
8 threads, each with its own `O_DIRECT` fd, to measure aggregate queue-depth-8 behaviour. No writes.

## Result

Raw output: [`raw/rand4k.out`](raw/rand4k.out).

| mode | IOPS | bandwidth | p50 latency | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 thread (QD1) | 1,819 | 7.5 MB/s | 358 µs | 1,296 µs | 1,753 µs |
| 8 threads (QD8) | 15,213 | 62.3 MB/s | 450 µs | 1,009 µs | 1,439 µs |

Sequential `O_DIRECT` (`dd bs=1M/4M`, 0.5–2 GiB) measured **1.0–1.7 GB/s**, i.e. ~1.3 GB/s
sustained.

## Reading

- The PLE gather is 16 rows on 16 different pages per token (GGUF inventory §3,
  [`gguf-inventory.md`](../../../docs/research/gguf-inventory.md)). At the measured QD1 p50 that is
  ~5.7 ms of exposed latency per token; at QD8 aggregate it is ~1.05 ms of wall time. Both fit
  inside a decode step at the current ~7.6 tok/s (131 ms) only if the gather is overlapped with
  embedding + layer 0, exactly as Strata schedules it.
- Random 4 KiB on this consumer drive is ~60x slower than its sequential bandwidth. Any proposal
  that reads **weights** (large, dense) from SSD per token is bounded by ~1.3 GB/s, not by the
  4 KiB number; any proposal that reads **sparse rows** is bounded by the 4 KiB number.
- This is a raw-device bound. A `mmap`-served read can be cheaper when the page is already in the
  page cache (see the 0xBakeer and tonyd2wild measurements cited in the landscape doc) and more
  expensive on a major fault. The net is what R5 must measure.
