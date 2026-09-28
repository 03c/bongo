# The SSD-resident n-gram/PLE "second shard"

**Status:** R5 characterisation, 2026-09-28. Owner: Coder ([BAS-67](/BAS/issues/BAS-67)).
Answers the CEO's question for [BAS-62](/BAS/issues/BAS-62): can the ~29 GB n-gram/PLE table stay on
SSD permanently without stalling decode?

**Short answer: yes — provided the reads are parallel, a bounded row cache exists, and the SSD is not
shared with per-token expert streaming.** The arithmetic is in [§4](#4-the-overlap-budget); the
conditions are in [§8](#8-answer-and-conditions).

Measured on the reference box (Arc Pro B70, Ryzen 7 9700X, 30 GiB RAM, KIOXIA EXCERIA G3 NVMe). Raw
data: [`bench/results/2026-09-28-ssd-ngram-shard/`](../../../bench/results/2026-09-28-ssd-ngram-shard/),
environment in [`run-environment.md`](../../../bench/results/2026-09-28-ssd-ngram-shard/run-environment.md).
Tools: [`bench/measure-ssd-ngram.py`](../../../bench/measure-ssd-ngram.py),
[`bench/ngram-row-cache-sim.py`](../../../bench/ngram-row-cache-sim.py).

## 1. The tensor, exactly

`tools/gguf-inventory.py` over the IQ2_XS shard 1 (raw: `raw/gguf-inventory-iq2xs.json`) gives the
second shard to the byte:

| Property | Value |
| --- | --- |
| Tensor | `per_layer_token_embd.weight` (also called the n-gram / PLE table) |
| Shard | **shard 1 only** (`…-IQ2_XS-00001-of-00002.gguf`), identical in every tier |
| Type | `IQ4_NL` (5 × 18 B blocks = 90 B/row; not re-quantised per tier) |
| Dimensions | `[160, 320001536]` = 51,200,245,760 elements |
| **Size** | **28,800,138,240 B = 26.82 GiB = 28.80 GB** |
| Rows | real `sum(head_vocab_sizes)` = 320,001,446; padded to 320,001,536 |
| Row width | **90 B** (160 values, IQ4_NL) |
| Absolute file offset | **361,831,168** (shard data section 10,967,808 + tensor offset 350,863,360) |

The hash geometry (`qwen4exp.ple.*` in the GGUF) is `ngram_size = 3`, `heads_per_ngram = 8`, so
`ple_n_heads = (3 − 1) × 8 = 16`: a **bigram family (heads 0–7)** and a **trigram family (heads
8–15)**. Every token reads **16 rows, one per head** — 1,440 B of useful data — hashed from the
token and its two predecessors (`src/models/qwen4exp.cpp`, `llm_graph_input_ple::set_input`).

The 16 head offsets are ≥ 20,000,003 rows apart (≥ 1.8 GB), so the 16 rows always land on **16
distinct 4 KiB pages**. The row-cache simulation confirms this on real tokens: **16.00 pages/token,
p50 = p90 = max = 16**, for every prompt in the set (`raw/ngram-row-cache.json`). Page traffic is
therefore **64 KiB/token** against 1,440 B useful — a **45.5×** amplification, as
[`gguf-inventory.md` §3](gguf-inventory.md#3-the-n-gram-table--real-size-type-and-placement) estimated.

## 2. Is it already excluded from VRAM and RAM?

**Yes — by llama.cpp's lazy-read path, not by `-ot`.** Verified against llama.cpp master:

- `qwen4exp.cpp` creates `per_layer_token_embd.weight` with `TENSOR_READ_LAZY`.
- `llama-model-loader.cpp`: lazy tensors resolve to the **CPU** buffer type, and `lazy_read::add()`
  enables row-on-demand loading when **mmap is available** and (in `auto` mode) the tensor is
  **> 4 GiB**. 26.82 GiB qualifies.
- Because `is_lazy` is checked before the user's `-ot` override, `-ot "per_layer_token_embd\.weight=CPU"`
  is a harmless no-op. bongo does not pass it anyway.

The bongo baseline (`~/.bongo/run/bongo-config.env`) is:

```
--n-gpu-layers 99 --n-cpu-moe 16 --ctx-size 131072 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --metrics
```

No `--no-mmap`, no `--lazy-mode off`, no `-ot`. So `--lazy-mode auto` keeps the table file-backed on
the CPU side: **not uploaded to VRAM, not resident in RAM**. The q2_0 server log confirms the
consequence — the whole model "loaded" in 1.23 s, which is only possible if the 26.82 GiB table was
never read. (The only loader warning is about the MoE `--n-cpu-moe` tensor overrides, which are
unrelated to PLE.)

**Guardrail:** do not add `--no-mmap` or `--lazy-mode off`. Either one forces the 28.80 GB resident
and guarantees OOM on a 30 GiB box.

## 3. Measured SSD cost

### 3.1 Method

`bench/measure-ssd-ngram.py` derives the PLE region from the inventory and reads it **read-only**:

- **direct** = `O_DIRECT` 4 KiB random reads (no page cache), at queue depths 1/8/16/32/64;
- **buffered_warm** = re-read offsets already in the page cache (the mmap-lazy best case);
- **buffered_cold_dontneed** = `posix_fadvise(DONTNEED)` a 2 GiB window, then read (the mmap-lazy
  cold case, with the kernel's 128 KiB readahead);
- **token_window** = the exact decode shape: 16 pages/token, fetched with a prefetch width of
  1/2/4/8/16, measuring first-issued to last-done per token — the window a real prefetcher must fit in;
- **sequential_direct_1mib** = 1 MiB blocks (a prefill-style ceiling).

All percentiles are over 4,000 random reads and 1,000 tokens.

### 3.2 Quiescent window (device capability)

Measured 07:49 local, before a co-resident benchmark started. The p50 == p99 signature of the QD1
row is the idle-device signature. Summary preserved in `raw/ssd-ngram-latency-quiescent.txt` (the
JSON was later overwritten by an instrumented, contended repeat).

| Scenario | p50 | p99 | throughput |
| --- | ---: | ---: | ---: |
| direct random, QD1 | **136.5 µs** | 140.3 µs | 7,861 IOPS (31 MB/s) |
| direct random, QD16 | 178.8 µs | 481.8 µs | 76,348 IOPS (313 MB/s) |
| buffered, warm | 0.7 µs | 1.3 µs | page-cache hit |
| buffered, cold (`DONTNEED`) | 103.6 µs | 138.4 µs | — |
| **token window, QD1** | **1,654.7 µs** | 1,872.4 µs | 599 tok/s |
| token window, QD2 | 850.6 µs | 989.9 µs | 1,149 tok/s |
| token window, QD4 | 518.3 µs | 676.7 µs | 1,798 tok/s |
| token window, QD8 | 357.6 µs | 521.4 µs | 2,288 tok/s |
| **token window, QD16** | **259.1 µs** | 416.7 µs | 2,278 tok/s |

Reading the table **serially costs 1.65 ms per token**; reading its 16 pages **in parallel costs
0.26 ms**. That 6.4× gap is the whole design decision.

### 3.3 Contended window (shared box)

The box is shared. Repeats `run1`–`run3` (raw JSON, with kernel PSI IO pressure recorded in each
file) ran while a co-resident `llama-bench` + `route_capture` pair read the same IQ2_XS file. PSI
`full_avg10` rose to **25–29%** during the runs — the disk was fully stalled a quarter of the time.

| Scenario | run1 p50 / p99 | run2 p50 / p99 | run3 p50 / p99 |
| --- | ---: | ---: | ---: |
| direct QD1 | 470 / 2,108 µs | 276 / 1,435 µs | 272 / 1,297 µs |
| direct QD16 | 677 / 1,870 µs | 681 / 2,355 µs | 617 / 1,777 µs |
| direct QD64 | 1,037 / 3,109 µs | 1,111 / 3,286 µs | 1,014 / 3,884 µs |
| sequential 1 MiB | 973 / 1,880 µs | 1,273 / 2,856 µs | 981 / 1,806 µs |
| token window QD1 | 4,761 / 9,697 µs | 6,674 / 12,084 µs | 4,949 / 9,018 µs |
| token window QD16 | 1,053 / 2,149 µs | 913 / 1,831 µs | 1,161 / 2,554 µs |

**A busy SSD costs 2–4×**: QD1 read latency 137 → ~270–470 µs, the parallel token window 0.26 →
0.91–1.16 ms. Sequential throughput fell from the drive's nominal range to 0.75–0.96 GB/s. This is
the "bad SSD" case the CEO asked about, and it is real on a busily shared consumer NVMe.

### 3.4 Notes on the device

The KIOXIA EXCERIA G3 is a DRAM-less consumer drive: QD1 random read is mediocre (137 µs, 7.9k
IOPS) even quiescent, and it degrades under mixed load. A datacentre SSD with DRAM would give
~60–80 µs QD1 and a similar or better parallel window. The conclusions below do not depend on the
drive being fast; they depend on the reads being parallel.

## 4. The overlap budget

Per token: **16 pages = 64 KiB** of page traffic, 1,440 B useful. The PLE gather feeds the per-layer
embedding used at **layer 1** (`qwen4exp.ple.layers = [1]`), so the reads must complete between
"token id known" and "end of layer 0" — the **embedding + layer 0 window**. That is roughly one
layer's time, at most two.

Baseline decode is **7.0–8.0 tok/s** (125–143 ms/token; `docs/research/expert-placement.md`,
llama-server log shows 6.97–7.33 tok/s at 4K). Across 48 layers that is ~2.6 ms/layer.

| Target | token budget | ~layer budget | serial QD1 window | parallel QD16 window | hidden? |
| --- | ---: | ---: | ---: | ---: | --- |
| **8 tok/s** (baseline) | 125 ms | ~2.6 ms | 1.65 ms | 0.26 ms | **yes**, either way |
| **40 tok/s** | 25 ms | ~0.52 ms | 1.65 ms | 0.26 ms | QD16 yes; QD1 no |
| **50 tok/s** | 20 ms | ~0.42 ms | 1.65 ms | 0.26 ms | QD16 yes; QD1 no |
| **60 tok/s** | 16.7 ms | ~0.35 ms | 1.65 ms | 0.26 ms | QD16 yes; QD1 no |

So at the bongo baseline the SSD reads are hidden even if serialised. At a 40–60 tok/s target the
window shrinks to a fraction of a millisecond and **serialised reads (1.65 ms) miss it; 16-way
parallel reads (0.26 ms, p99 0.42 ms) fit.** On a contended SSD the parallel window becomes
0.9–1.2 ms, which is close to the whole 40 tok/s budget — the margin is gone, but the floor from
these measurements is still ~800–1,000 tok/s of pure PLE throughput, far above any decode target.

**Bandwidth is never the problem at decode.** At 50 tok/s and 0 % cache hit: 64 KiB × 50 = 3.2 MB/s
against 313 MB/s measured. Even with 0 % cache and a contended drive (192 MB/s) the PLE needs 1.7 %
of the device.

### 4.1 Where it does become binding

- **Serialised prefill.** A 2,048-token chunk is 32,768 pages = 128 MiB. At a contended QD1 latency
  (~0.4 ms) that is **13 s per chunk**, i.e. a PLE-bound prefill of ~157 tok/s — below the measured
  91–270 tok/s in places. At 49k IOPS (QD64, contended) the same chunk is 0.67 s (~3,050 tok/s).
  **Prefill must batch the chunk's rows, de-duplicate pages, and use high queue depth.**
- **Large decode batches.** `N` concurrent sequences read 16N pages/step. At `N = 16` that is 256
  pages = 1 MiB/step ≈ 5.2 ms at 49k IOPS. Fine against a 20 ms step, but it scales linearly: at
  `N = 64` it is ~21 ms and starts to bind.
- **KV size.** KV is not a PLE cost directly, but a bigger KV or a bigger tier shrinks the RAM left
  for the n-gram cache. `gguf-inventory.md` §6 puts the leftover at **~10.5–13 GiB for IQ2_XS but
  only ~3.5–6 GiB for IQ3_XXS**; the IQ3_XXS case is the one where cache misses (and therefore the
  reads above) will hurt.

## 5. Row cache and recurrence

`bench/ngram-row-cache-sim.py` reimplements the exact `qwen4exp` row hash and replays real token
streams (tokenized with the model's own tokenizer; `raw/tokens-*.json`). Corpus: 1,174 chat tokens,
5,861 prose, 9,122 code, 22,245 long-docs, 530,682 Python stdlib source tokens (569k total).

| Prompt | unique rows/token | intra-prompt recurrence |
| --- | ---: | ---: |
| chat | 10.68 | 33.3 % |
| prose | 10.66 | 33.4 % |
| code | 8.17 | 48.9 % |
| long docs | 7.78 | 51.4 % |
| Python stdlib | 5.11 | **68.1 %** |

Shared LRU across the whole corpus (the server case, cache survives requests):

| cache | size | hit rate |
| ---: | ---: | ---: |
| 25,000 rows | 2.2 MB | 50.3 % |
| 100,000 rows | 9.0 MB | 57.4 % |
| 1,000,000 rows | **90.0 MB** | **66.1 %** |
| 2,000,000 rows | 180.0 MB | 67.6 % |

This is consistent with Strata's numbers (20–34 % intra-prompt, up to 82 % at ~95 MB) — prose and
chat reproduce the 33 % end; code has materially more recurrence. At 66 % hit rate only **5.4
pages/token = 22 KiB/token** reach the SSD.

**Why an explicit 90 B-row cache beats the OS page cache.** Under mmap-lazy the cache is the kernel
page cache, which stores whole 4 KiB pages; a page covers ~45 consecutive 90 B rows but random
hashing uses ~1 of them. A 90 MB explicit cache holds **~1M rows**; 90 MB of page cache holds
**~23k useful rows**. Per byte of RAM the explicit row cache covers ~45× more of the working set,
so the same memory budget yields a much higher hit rate. The OS page cache still helps (it removes
the second read of a page, and warm reads are 0.7 µs), but it is not a substitute.

## 6. Recommended design

1. **Direct-file reads, not mmap, for the PLE table.** Dedicate a reader thread (or io_uring) that
   issues all 16 × 4 KiB `O_DIRECT` reads for a token **as soon as the token id is known**, and
   completes them before layer 1. `posix_fadvise(POSIX_FADV_RANDOM)` if buffered reads are used.
2. **Parallel / asynchronous, never serialised.** 16-way (or at least 8-way) in-flight reads keep
   the per-token window at ~0.26 ms. This is the single most important requirement.
3. **Bounded row cache** in front of the file: keyed by global row index, storing the 90 B row,
   LRU, **1M rows / 90 MB** as a floor (larger is better — the box can spare it). This also avoids
   re-reading a page whose other 44 rows will never be used.
4. **Bounded in-flight window** (e.g. 32–64 pages) with a prefetch depth cap, so a slow SSD produces
   bounded latency instead of unbounded memory growth; the gather blocks on the window, not on
   individual reads.
5. **Keep mmap + `--lazy-mode auto` as the fallback** (do not remove it), so a box without
   `O_DIRECT` support still works. Never force the table resident.
6. **Keep the SSD to the PLE table.** The current placement keeps all experts in VRAM/RAM. If a
   later design streams experts from the same SSD, the two workloads contend and the budget above
   must be split; the contended runs in §3.3 show what that looks like.

## 7. Risk: stalling on a bad SSD

- Serialised reads are the failure mode: 1.65 ms quiescent, 5–7 ms contended, always outside a
  40–60 tok/s layer-0 window. A prefetch with a bounded window and a row cache is what prevents it.
- A DRAM-less consumer drive degrades under any co-resident I/O (measured 2–4×). The PLE path
  should not assume the drive is idle.
- If the SSD cannot keep up, the stall is localised: the PLE gather at layer 1 blocks, and decode
  slows; it does not corrupt state or OOM. The mitigation is to shrink the working set (smaller
  context/batch), grow the row cache, or accept a lower tok/s.
- The explicit row cache is the cheap insurance: 90 MB buys a 66 % hit rate on the measured corpus,
  cutting SSD traffic from 64 KiB to 22 KiB per token and shrinking the required parallelism.

## 8. Answer and conditions

**Yes — the second shard can stay on SSD at all times.** Measured here: 26.82 GiB, 90 B/row, 16
distinct 4 KiB pages per token, 64 KiB of page traffic per token, 0.26 ms per token when the 16
reads are parallel (1.65 ms serial), against a baseline token budget of 125 ms and a 40–60 tok/s
budget of 17–25 ms.

Conditions:

1. **Read the 16 rows in parallel / asynchronously**, not serially. This is mandatory for a 40–60
   tok/s target; at the 8 tok/s baseline even serial reads fit.
2. **Keep a bounded row cache** of at least ~1M rows (90 MB). It is ~45× more RAM-efficient than the
   OS page cache for this access pattern.
3. **Keep experts out of the SSD** (they stay in VRAM/RAM today). Sharing the SSD with per-token
   expert streaming removes the margin.
4. **Keep mmap + lazy on** and never pass `--no-mmap` / `--lazy-mode off`.
5. **Batch and de-duplicate PLE rows for prefill** and cap the in-flight window, so a large chunk or
   a busy SSD cannot serialize the table.

## 9. Reproduce

```sh
# 1. tensor inventory (range-reads the header only)
python3 tools/gguf-inventory.py \
  --local ~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/\
Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --json bench/results/2026-09-28-ssd-ngram-shard/raw/gguf-inventory-iq2xs.json --summary

# 2. SSD latency on the actual PLE region (read-only)
python3 bench/measure-ssd-ngram.py \
  --inventory bench/results/2026-09-28-ssd-ngram-shard/raw/gguf-inventory-iq2xs.json \
  --out bench/results/2026-09-28-ssd-ngram-shard/raw/ssd-ngram-latency.json

# 3. row-cache hit rate on tokenized prompts
python3 bench/ngram-row-cache-sim.py \
  --inventory bench/results/2026-09-28-ssd-ngram-shard/raw/gguf-inventory-iq2xs.json \
  --tokens 'bench/results/2026-09-28-ssd-ngram-shard/raw/tokens-*.json' \
  --out bench/results/2026-09-28-ssd-ngram-shard/raw/ngram-row-cache.json
```

Tokenization used the model's own tokenizer:

```sh
~/.bongo/llama/b11223/vulkan/llama-tokenize -m <shard1.gguf> -f prompt.txt --ids --no-bos
```

`measure-ssd-ngram.py` records `/proc/pressure/io` in every result; a run with high PSI is a
contended run and should be read as the risk case, not the device capability.
