# M3.4 — the PLE / n-gram second-shard reader

**Status:** implemented and measured, 2026-09-28. Owner: Coder ([BAS-77](/BAS/issues/BAS-77)).
Parent: [BAS-62](/BAS/issues/BAS-62). Extends the R5 characterisation
([BAS-67](/BAS/issues/BAS-67), [`ssd-ngram-shard.md`](ssd-ngram-shard.md)).

**Disposition (ADR-0004):** the reader stays **opt-in** behind `--ple-reader off`. The engine A/B at 128K
is neutral for prefill and decode, so M3.4b closes ([BAS-79](/BAS/issues/BAS-79)) and no further engine work
runs on this line. See [ADR-0004](../adr/0004-ple-reader-disposition.md). The 6.7x/4.3x below is an
isolated read-path result.

**Verdict: the design holds.** Reading the 26.82 GiB table through a dedicated
`O_DIRECT` reader pool with a bounded row cache is **4.3x faster per token** than
the mmap page-fault path on decode and **6.7x faster** on a de-duplicated 2048-token
prefill chunk, on the same device at the same time. The table stays on disk: the
reader records **zero major faults** and the process RSS grows by ~34 MiB, not by
26.82 GiB. The R5 device re-run reproduces the quiescent numbers (token window
**262 us** parallel vs **1,700 us** serial).

The reader is a standalone, reusable implementation measured against the real
tensor. It is not yet linked into llama.cpp; the engine wiring is tracked in the
follow-up issue named at the end of this document.

## 1. Design

R5 measured the tensor exactly: `per_layer_token_embd.weight`, IQ4_NL,
`[160, 320001536]`, 90 B/row, 28,800,138,240 B (26.82 GiB), at absolute file offset
361,831,168. Every token reads 16 rows (8 bigram heads + 8 trigram heads) and the
heads are >= 20,000,003 rows apart, so the 16 rows always land on 16 distinct
4 KiB pages — 64 KiB of page traffic for 1,440 B of useful data.

The reader implements the R5 recommendation:

1. **Direct reads, not mmap.** A pool of `io_depth` threads issues page-aligned
   4 KiB `O_DIRECT` reads. Decode submits one token's 16 pages together; the
   window the engine has to hide is first-issue to last-done.
2. **Bounded in-flight window.** At most `--window` page reads are outstanding in
   one batch, so a slow or busy disk produces bounded latency, not unbounded
   memory.
3. **Bounded LRU row cache** keyed by global row index, storing the 90 B row
   (default `--cache-rows 1000000` = ~90 MB). A 90 MB row cache covers ~1 M rows;
   90 MB of page cache covers only ~23 k useful rows, because a 4 KiB page holds
   ~45 consecutive 90 B rows of which random hashing uses one.
4. **Batch + de-duplicate for prefill.** A chunk's 16N row ids are de-duplicated
   to unique rows, then to unique pages, and read once at high queue depth.
5. **mmap + lazy is the fallback and is never removed.** `MmapPle` in the same
   tool reproduces the llama.cpp `--lazy-mode auto` path (MAP_SHARED + MADV_RANDOM,
   row access = page fault) so the two paths are measured on the same hardware.

Row extraction handles rows that straddle a 4 KiB boundary (the region start,
361,831,168, is not page-aligned: it is 2,816 bytes into a page), so a reader
window never splits a straddling row across two reads.

## 2. Implementation

[`bench/ple_reader.py`](../../bench/ple_reader.py) — one file, three pieces:

| Piece | Role |
| --- | --- |
| `IOPool` | fixed worker pool + bounded aligned arena; two conditions on one lock (workers wait for work, the caller waits for the batch; only the last completion wakes the caller) |
| `PleReader` | geometry from the inventory, `gather(16 rows)` decode path, `prefetch(rows)` prefill path, `RowCache` |
| `MmapPle` | the mmap/fault baseline, including `posix_fadvise(DONTNEED)` eviction so the baseline really faults |

`--scenario selftest,decode,prefill` runs a byte-for-byte correctness check, the
per-token decode comparison, and the prefill comparison, and writes JSON.

## 3. Correctness

`selftest` reads known rows (0, 1, 2, 999, 1,000,000, 12,345,678 and the last row
320,001,535) through the reader, a raw `pread`, and mmap, and requires all three to
match. It passes, including the final padded row. The reader is read-only; there is
no write path and no format parsing beyond the inventory JSON.

## 4. Measured results

### 4.1 R5 device re-run (quiescent)

`bench/measure-ssd-ngram.py`, same harness as R5, run while the device was
quiescent — raw: [`raw/ssd-ngram-latency.json`](../../bench/results/2026-09-28-ple-reader/raw/ssd-ngram-latency.json).
It reproduces R5's quiescent window:

| Scenario | This run p50 | R5 quiescent |
| --- | ---: | ---: |
| direct random QD1 | 130.3 us | 136.5 us |
| direct random QD16 | 172.2 us | 178.8 us |
| token window QD1 | 1,699.8 us | 1,654.7 us |
| **token window QD16** | **262.4 us** | **259.1 us** |
| sequential 1 MiB | 304.1 us (2,760 MiB/s) | — |

### 4.2 Decode window — reader vs the mmap fault path

4,000 real tokens (chat + prose + code; `raw/ple-reader.json`). `reader_direct` is
the pool with no row cache, i.e. every token pays its 16 reads; `mmap_cold` is the
llama.cpp lazy path with the pages dropped from the page cache first.

| Decode path | p50 | p90 | p99 | major faults |
| --- | ---: | ---: | ---: | ---: |
| **reader, no cache** | **333.5 us** | 416.1 us | 518.7 us | 0 |
| mmap cold (faults) | 1,436.9 us | 1,680.9 us | 1,970.6 us | 42,236 |
| mmap warm (page cache) | 4.3 us | 4.8 us | 5.2 us | 0 |
| **reader, 1 M-row cache** | **3.5 us** | 3.9 us | 4.7 us | 0 |

The parallel reader is **4.3x faster than the faulting mmap path** (333 us vs
1,437 us) on identical pages, and the 1 M-row cache makes a decode gather as cheap
as a warm page-cache hit (3.5 us). The 333 us is the Python implementation; R5's
pure-I/O floor for the same window is 262 us, so the design, not the language, sets
the floor.

### 4.3 Prefill — one de-duplicated 2048-token chunk

Same rows for both paths (23,815 unique rows -> 24,228 unique pages, 99.2 MB),
pages evicted first for the mmap path (`raw/ple-reader.json`):

| Prefill path | wall | throughput | pages | major faults |
| --- | ---: | ---: | ---: | ---: |
| **reader (dedup, pool)** | **0.362 s** | **261 MiB/s, 5,655 tok/s** | 24,228 | 0 |
| mmap cold (dedup, faults) | 2.416 s | 39 MiB/s, 848 tok/s | 24,228 | 24,193 |

**6.7x** on the same pages: batching + de-duplication + parallel `O_DIRECT` versus
one page fault per read. This is the mechanism R3 predicted as "up to ~2x prefill
versus mmap faults", measured directly. The engine-level prefill win will be
smaller than 6.7x because only the PLE portion of prefill is replaced; the
standalone number bounds it from above.

The chunk touched 11.8 unique pages/token (not 16) because rows recur across a
chunk, so de-duplication alone removes ~26% of the reads before any cache.

### 4.4 Row cache

Same corpus as R5 (1,174 chat + 5,861 prose + 9,122 code + 22,245 long-docs +
530,682 Python-stdlib tokens); the reader's LRU reproduces the R5 simulation:

| cache | size | shared LRU hit rate |
| ---: | ---: | ---: |
| 25,000 rows | 2.2 MB | 50.3% |
| 100,000 rows | 9.0 MB | 57.4% |
| **1,000,000 rows** | **90.0 MB** | **66.1%** |
| 2,000,000 rows | 180.0 MB | 67.6% |

On the 4,000-token decode run the 1 M-row cache cut pages read from 65,318 (no
cache) to **44,418** — 32% less SSD traffic.

### 4.5 Memory residency

The reader is `O_DIRECT`, so it does not populate the page cache and does not map
the table. Across the whole benchmark the process RSS went **18.7 MiB -> 33.7 MiB**
(`meta.rss_kib_start/end` in `raw/ple-reader.json`), and `ru_majflt` was 0 for every
reader scenario — there is no memory-mapped 26.82 GiB region and no fault path. The
table is never forced resident.

## 5. Keeping the Stage 0 baseline

Nothing in this change touches `bongo.sh`, the pinned engine, or the flags. The
Stage 0 Vulkan `--n-cpu-moe 16` baseline remains the default and is still
selectable. The M3.4 reader is additive and read-only.

## 6. Llama.cpp integration (BAS-79)

The reader is linked into llama.cpp at `4da633776` (llama.cpp `b11223`) by
`bench/ple-reader/ple-reader.patch`, built in a container. Under `--lazy-mode auto`
the PLE tensor is registered with the reader; `--ple-reader on` serves its
`GET_ROWS` from the O_DIRECT pool, while `off` keeps the mmap + lazy path
bit-for-bit. The integration shape:

- The PLE tensor is already `TENSOR_READ_LAZY`; under `--lazy-mode auto` it is a
  CPU `buffer_from_host_ptr` over the full mmap, so `GGML_OP_GET_ROWS` on it
  page-faults (`src/llama-mmap.cpp`: lazy ranges get `MADV_RANDOM` and are excluded
  from `MAP_POPULATE`/prefetch). The patch replaces that buffer with a reader-owned
  buffer: a small library (the C++ port of `IOPool` + `RowCache`) that `pread`s the
  requested rows with `O_DIRECT` and serves `get_rows` from the row cache.
- Flag it (`--ple-reader on|off|auto`), default `off`, with mmap + lazy as the
  fallback; `--ple-reader off` must reproduce today's binary bit-for-bit.
- Keep the window and cache bounded and expose them (`--ple-reader-cache-mb`,
  `--ple-reader-window`), so the same binary can be tuned on any box.
- Gate it with the same protocol: R5 harness, then `bench/harness.py` at 4096 and
  131072 against the pinned baseline; no regression >5%; RSS must not grow by the
  table size.

## 7. Residual risk

- **No engine-level prefill gain (measured).** The rebuilt-server A/B at 4K/128K
  against the pinned Stage-0 baseline (`--n-cpu-moe 16`) and the M3.3 byte-budget
  placement is in `bench/results/2026-09-28-ple-reader-engine/` (`summary.md`).
  The 128K prefill - the only real prefill in the matrix - moves **+0.5%**
  (baseline) and **-0.7%** (M3.3) with the reader on, i.e. inside run-to-run noise.
  No bound is exceeded (no >5% regression), and RSS grows at most +0.39 GiB
  against the 26.82 GiB table, so the table is not forced resident. The 6.7x
  standalone result is a PLE **read-path** result; the engine 128K prefill is not
  PLE-gather-bound, so the reader does not move it. The 4K rows are prefix-cache
  hits (`cache_n=4094`, `prompt_n=3`); `summarize-ab.py` marks them `cache-hit` and
  excludes them from the prefill comparison, so they cannot be quoted as a gain.
- **Baseline/M3.3 regression comparison is measured** against the configs from
  [BAS-72](/BAS/issues/BAS-72) and [BAS-76](/BAS/issues/BAS-76); both are within
  the 5% bound with the reader on.
- **Python overhead.** The test reader is ~25% slower than the R5 I/O floor on
  decode; the C++ port does not carry that.
- **Shared SSD.** The reader benchmark ran at PSI IO `full_avg10` up to 11%; the
  R5 re-run itself was quiescent and matches the R5 floor. Every raw file records
  `io_pressure_start`/`io_pressure_end`.

## 8. Reproduce

```sh
M=~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs
INV=bench/results/2026-09-28-ssd-ngram-shard/raw/gguf-inventory-iq2xs.json

# R5 device harness (read-only, ~5 s quiescent)
python3 bench/measure-ssd-ngram.py --inventory "$INV" \
  --out bench/results/2026-09-28-ple-reader/raw/ssd-ngram-latency.json

# M3.4 reader: correctness + decode window + prefill chunk
python3 bench/ple_reader.py --inventory "$INV" \
  --tokens 'bench/results/2026-09-28-ssd-ngram-shard/raw/tokens-{chat,prose,code}.json' \
  --scenario selftest,decode,prefill --cache-rows 1000000 --io-depth 16 --window 256 \
  --out bench/results/2026-09-28-ple-reader/raw/ple-reader.json
```

Engine revision for every number: llama.cpp `b11223`
(`4da6337767f973e2b4d0797e5b323d77d8565e4a`), tier IQ2_XS, model shard
`...-IQ2_XS-00001-of-00002.gguf` (39,788,473,344 B, mtime 2026-09-28T00:01:08+01:00).
The reader itself does not use the engine; it reads the shard directly.
