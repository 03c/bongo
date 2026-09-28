# GGUF tensor inventory + 32 GB / 30 GiB buffer-placement plan

Status: complete, 2026-09-27. Owner: Coder ([BAS-51](/BAS/issues/BAS-51)).
Supersedes the tensor-size **estimates** in [intel-arc-b70.md](intel-arc-b70.md) §2 and §3, and resolves
its §2 open question ("exact tensor-to-shard mapping") and its §7 open questions 1 and 5 (partly).

Everything in this document was **read from the published GGUF headers** of
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`ua ru`, HTTP range reads, **no weight download**),
then cross-checked against three artifacts the model authors publish in the same repo
(`SHA256SUMS`, `release-manifest.json`, `tensor-allocation/*.rco-allocation.txt`) and against the
`exact-source-recovery/*.json` capsules. Nothing here is an estimate except where explicitly marked.

Tool: [`tools/gguf-inventory.py`](../../tools/gguf-inventory.py) — standard library only, reads a local GGUF
or a HF repo file with HTTP range requests, resolves split shards, computes every tensor's exact byte size,
buckets tensors into expert / shared-expert / n-gram-table / embed-head / other, and verifies that the
computed sizes tile each shard's data section exactly.

## 0. How to reproduce (the verification for this task)

```sh
python3 tools/gguf-inventory.py --repo \
  hf:ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --json out/iq2xs.json --summary \
  --cross-check-manifest release-manifest.json \
  --cross-check-alloc tensor-allocation/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS.rco-allocation.txt \
  --cross-check-capsule exact-source-recovery/IQ2_XS.json
```

Actual output (IQ2_XS, both shards, 2026-09-27):

```text
source          : ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF
  file          : Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  file          : Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
gguf version   : 3   alignment: 32 B
tensor count   : 1224
tensor bytes   : 68141143040 (63.46 GiB)
bytes fetched  : 12582912 (12.58 MB)
read ops       : 12 (local reads, or HTTP range requests for a repo)

shards:
  file                                                            file GiB  tensors  data GiB fetched MB
  Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf        37.06      342     37.05      11.53
  Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf        26.42      882     26.42       1.05

buckets:
  bucket              tensors          bytes      GiB  types
  experts                 144    35454976000    33.02  IQ1_Mx6, IQ2_Sx68, IQ2_XXSx22, Q2_0x48
  ngram_table               1    28800138240    26.82  IQ4_NLx1
  other                   933     3075622400    2.86  BF16x388, F16x1, F32x388, IQ3_Sx21, IQ4_XSx122, Q6_Kx13
  embed_head                2      675430400    0.63  IQ4_XSx2
  shared_experts          144      134976000    0.13  IQ3_Sx23, IQ4_NLx36, IQ4_XSx57, Q2_0x5, Q6_Kx16, Q8_0x7

layout check   : OK (computed sizes tile every shard exactly, padding < alignment)

cross_checks_manifest:
   {"file": "…-IQ2_XS-00001-of-00002.gguf", "check": "byte size matches release-manifest.json", "expected": 39788473344, "actual": 39788473344, "ok": true}
   {"file": "…-IQ2_XS-00002-of-00002.gguf", "check": "byte size matches release-manifest.json", "expected": 28363693824, "actual": 28363693824, "ok": true}
   {"check": "IQ2_XS: sum of shards ~= unsplit_bytes (header duplication)", "expected": 68152166976, "actual": 68152167168, "delta_bytes": 192, "ok": true}
   {"check": "IQ2_XS: tensor count == manifest tensor_count", "expected": 1224, "actual": 1224, "ok": true}
cross_checks_alloc:
   {"check": "per-tensor type == rco-allocation.txt", "compared": 1224, "mismatches": 0, "not_in_alloc_file": 0, "ok": true}
cross_checks_capsule:
   {"check": "tensor -> shard/offset/bytes == exact-source capsule", "compared": 1224, "mismatches": 0, "ok": true}

wrote out/iq2xs.json
```

**Cost of the whole inventory: 12.58 MB fetched, 12 range requests, ~7 s, per tier.** The three tiers
together cost 37.7 MB. The same run passes for `IQ3_XXS` and `Q2_0` (identical check results: 0 type
mismatches, 0 capsule mismatches, tensor count 1224).

Offline checks that need no network:

```sh
python3 tools/gguf-inventory.py --self-test
# SELF-TEST OK: local GGUF parse, type table, buckets, layout check and unknown-type
#               inference all behaved as expected (6 synthetic tensors)

python3 tools/gguf-inventory.py --local /path/to/any.gguf --summary
```

The `--local` path was exercised on the repo's `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf`
(334 tensors, 907,523,008 bytes of tensor data, layout check OK, 20 KB read).

## 1. Provenance and cross-checks

| Claim | Evidence |
| --- | --- |
| 1,224 tensors, GGUF v3, alignment 32 B, 48 blocks | parsed header (`split.tensors.count` = 1224) |
| Per-tensor byte sizes | computed from `dims` × ggml type; **validated** because the sizes tile each shard's data section with `padding < 32 B` and **zero unaccounted tail** |
| Per-tensor types | match `tensor-allocation/*.rco-allocation.txt` for **1224/1224** tensors in all three tiers |
| Tensor → shard, offset, bytes | match the authors' `exact-source-recovery/*.json` capsule for **1224/1224** tensors in all three tiers |
| Shard byte sizes | match `release-manifest.json` `files[].bytes` and `SHA256SUMS` file list |
| Tensor count | matches `release-manifest.json` `models[].tensor_count` = 1224 |
| Truncation rounding | `sum(shards) − unsplit_bytes` = 192 B (IQ2_XS), 224 B (IQ3_XXS/Q2_0) = the extra per-shard headers; the unsplit file carries one header, the shards carry one each |

`SHA256SUMS` / `release-manifest.json` values are therefore cross-checked **at file-size, file-list and
tensor-count level**. The published SHA-256 *digests* are not recomputed here: hashing 68 GB would mean
downloading the full weights, which is exactly what this task avoids. Each capsule does carry
`streaming_reconstruction_verified: true` against the unsplit `sha256`, so the digest path is covered by
the authors, and our per-tensor map agrees with those capsules byte-for-byte.

### New finding: ggml type 42

The `ffn_down_exps` tensors of every layer use **ggml type id 42** (`block 256, 72 bytes`), which the
`rco-allocation.txt` file labels **`Q2_0`**. Type 42 was not in the ggml tables this tool started with; it
was derived from the file layout and then confirmed two ways: the per-tensor name→type comparison against
`rco-allocation.txt` (1224/1224) and the type histogram (53 `Q2_0` tensors = 48 `ffn_down_exps` + 5
`ffn_down_shexp`). Type 42 is therefore `Q2_0 = 256/72` (2.25 bits/weight) with high confidence.

## 2. Shard mapping — resolves the open question in [intel-arc-b70.md](intel-arc-b70.md) §2

llama.cpp splits a GGUF **at tensor granularity**, and the published `-of-00002` cut is not on a layer
boundary: the split lands in the middle of one layer, and where it lands differs per tier. Layer 13
(IQ2_XS, Q2_0) / layer 12 (IQ3_XXS) is split across the two files.

| Tier | Shard 1 tensors | Shard 2 tensors | Layers fully in shard 1 | Split layer | Tensors of the split layer in shard 1 |
| --- | ---: | ---: | --- | --- | --- |
| IQ2_XS | 342 | 882 | 0–12 | 13 | `attn_gate`, `attn_qkv` |
| IQ3_XXS | 317 | 907 | 0–11 | 12 | `attn_gate`, `attn_qkv` |
| Q2_0 | 344 | 880 | 0–12 | 13 | `attn_gate`, `attn_qkv`, `ffn_down_shexp`, `ffn_down_exps` |

Everything else in shard 1 is global: `token_embd.weight`, `output.weight`, the `output_hc_*` and
`ple_*` tensors, and the n-gram table. **No tensor is stored across a shard boundary.**

Expert tensors (`ffn_{gate,up,down}_exps.weight`) by shard:

| Tier | Expert tensors in shard 1 | Expert bytes in shard 1 | Expert tensors in shard 2 | Expert bytes in shard 2 |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XS | 39 (layers 0–12) | 8.72 GiB | 105 (layers 13–47) | 24.30 GiB |
| IQ3_XXS | 36 (layers 0–11) | 8.69 GiB | 108 (layers 12–47) | 31.27 GiB |
| Q2_0 | 40 (layers 0–13, incl. `blk.13.ffn_down_exps`) | 8.79 GiB | 104 | 22.85 GiB |

Shard headers are not the same size, which is why the "shard 1 is ~39.8 GB in every tier" observation in
intel-arc-b70.md §2 holds: shard 1 always carries the 26.82 GiB n-gram table plus the full tokenizer
metadata (11.0 MB header), while shard 2 carries a minimal 56–58 KB header and no n-gram table.

| Tier | Shard 1 bytes | Shard 1 header | Shard 2 bytes | Shard 2 header | Data bytes total |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_XS | 39,788,473,344 (37.06 GiB) | 10,967,802 | 28,363,693,824 (26.42 GiB) | 56,319 | 68,141,143,040 |
| IQ3_XXS | 39,785,790,560 (37.05 GiB) | 10,966,216 | 36,180,282,560 (33.70 GiB) | 57,905 | 75,955,048,960 |
| Q2_0 | 39,799,117,984 (37.07 GiB) | 10,967,945 | 26,750,834,816 (24.91 GiB) | 56,176 | 66,538,928,640 |

Practical consequence for the downloader: **shard 1 is not optional and not tier-specific** — every tier
needs all 37 GiB of it, because the n-gram table alone is 26.8 GiB and lives there.

## 3. The n-gram table — real size, type and placement

| Property | Value |
| --- | --- |
| Tensor name | `per_layer_token_embd.weight` |
| Shard | **shard 1** (`…-00001-of-00002.gguf`) in all three tiers |
| File offset | 361,831,168 (IQ2_XS) / 461,157,600 (IQ3_XXS) / 461,159,328 (Q2_0) — shard-relative 350,863,360 / 450,191,360 / 450,191,360 |
| ggml type | `IQ4_NL` (block 32, 18 bytes → 4.5 bits/weight) in **all three tiers** (it is not requantised) |
| Dimensions | `[160, 320001536]` = 51,200,245,760 elements |
| **Exact size** | **28,800,138,240 B = 26.82 GiB = 28.80 GB** — identical in all three tiers |
| Share of the tier | 42.3% of IQ2_XS, 37.9% of IQ3_XXS, 43.3% of Q2_0 |
| Real row count | 16 heads, `sum(head_vocab_sizes)` = **320,001,446** rows of `embedding_length_per_layer_input` = 160 values → 90 bytes/row; rows are padded to 320,001,536 (next multiple of 256) |
| Placement decisions it forces | must **never** be uploaded to VRAM (26.82 GiB > total VRAM), and must be **CPU-side** so it can be served by the OS page cache out of the file |

This confirms and sharpens the intel-arc-b70.md estimate ("~29 GB / 28.8 GB, disk-resident"): the real
figure is 28,800,138,240 bytes, it is an ordinary GGUF tensor inside shard 1, not a side file, and it is
`IQ4_NL`, so a row read is a 90-byte IQ4_NL dequant (3 × 18-byte blocks + 36 bytes of 4-bit payload).

**Access pattern (risk).** `qwen4exp.ple.ngram_size` = 3 and `heads_per_ngram` = 8 give
`ple_n_heads = (ngram_size − 1) × heads_per_ngram = 2 × 8 = **16**`, which is exactly the 16 entries in
`ple.head_offsets` / `ple.head_vocab_sizes`. So a token reads **16 rows, one per head** (~1.4 KB of useful
data) chosen by hash, i.e. scattered over 26.8 GiB. Because each 90-byte row sits on its own 4 KiB page, the
*page* traffic can be ~45× the useful bytes: 16 random rows ≈ 64 KiB of pages per token. At 50 tok/s that is
~3 MB/s of random 4 KiB reads. This is the single biggest unknown in the whole plan and it is a measurement
item (§7), not a solved problem.

**Runtime support, verified upstream (llama.cpp master, checked 2026-09-27).** The n-gram table is not
something bongo has to place by hand — llama.cpp already has a mechanism for it:

- `src/models/qwen4exp.cpp` creates `per_layer_token_embd.weight` with the **`TENSOR_READ_LAZY`** flag.
- `src/llama-model-loader.cpp:lazy_read::buft()` resolves lazy tensors to the **CPU** buffer type, and
  `lazy_read::add()` only enables lazy reading when **mmap is available** and (in `auto` mode) the tensor is
  **larger than 4 GiB**. Our table is 26.82 GiB, so it qualifies.
- `--lazy-mode` (`-lzm`, default `auto`) controls this: `on` = read such tensors' rows from disk on demand,
  `auto` = on for tensors > 4 GiB, `off` = keep them resident. `--load-mode`/`--no-mmap` interacts: without
  mmap the loader warns that the tensor "is loaded into RAM in full".
- Because `is_lazy` is checked *before* the user's `-ot` override, an explicit
  `-ot "per_layer_token_embd\.weight=CPU"` is a harmless no-op — lazy already forces the CPU buffer type.
- `LLAMA_LAZY_MODE_AUTO` degrades to `OFF` if **any** device reports `mmap_support == false`, so on the Arc
  box this must be checked for the SYCL device before trusting it.

Consequences for bongo: **do not pass `--no-mmap` or `--lazy-mode off`** (either forces 28.80 GB resident
and guarantees OOM), and make `--lazy-mode` / `mmap_support` part of the baseline configuration and the
benchmark record ([BAS-52](/BAS/issues/BAS-52)).

## 4. IQ2_XS per-tensor-class sizes (the tier we start with)

Classes are per tensor name with the layer index stripped, so 48 layers collapse into one row.
Bucket totals for all three tiers are in §5.

| Class | Tensors | Bytes | GiB | Types |
| --- | ---: | ---: | ---: | --- |
| `per_layer_token_embd.weight` | 1 | 28800138240 | 26.82 | IQ4_NL x1 |
| `ffn_gate_exps.weight` | 48 | 12065177600 | 11.24 | IQ1_M x3, IQ2_S x34, IQ2_XXS x11 |
| `ffn_up_exps.weight` | 48 | 12065177600 | 11.24 | IQ1_M x3, IQ2_S x34, IQ2_XXS x11 |
| `ffn_down_exps.weight` | 48 | 11324620800 | 10.55 | Q2_0 x48 |
| `attn_qkv.weight` | 36 | 498688000 | 0.46 | IQ3_S x1, IQ4_XS x35 |
| `output.weight` | 1 | 337715200 | 0.31 | IQ4_XS x1 |
| `token_embd.weight` | 1 | 337715200 | 0.31 | IQ4_XS x1 |
| `hc_attn_down.weight` | 48 | 314572800 | 0.29 | BF16 x48 |
| `hc_attn_up.weight` | 48 | 314572800 | 0.29 | BF16 x48 |
| `hc_ffn_down.weight` | 48 | 314572800 | 0.29 | BF16 x48 |
| `hc_ffn_up.weight` | 48 | 314572800 | 0.29 | BF16 x48 |
| `attn_gate.weight` | 36 | 296878080 | 0.28 | IQ3_S x11, IQ4_XS x22, Q6_K x3 |
| `ssm_out.weight` | 36 | 295772160 | 0.28 | IQ3_S x6, IQ4_XS x29, Q6_K x1 |
| `ffn_gate_inp.weight` | 48 | 251658240 | 0.23 | F32 x48 |
| `attn_q.weight` | 12 | 197345280 | 0.18 | IQ3_S x1, IQ4_XS x11 |
| `attn_output.weight` | 12 | 103219200 | 0.10 | IQ3_S x1, IQ4_XS x10, Q6_K x1 |
| `ple_key.weight` | 1 | 52428800 | 0.05 | BF16 x1 |
| `ffn_down_shexp.weight` | 48 | 47667200 | 0.04 | IQ4_NL x36, Q2_0 x5, Q8_0 x7 |
| `ffn_gate_shexp.weight` | 48 | 44070400 | 0.04 | IQ3_S x9, IQ4_XS x31, Q6_K x8 |
| `ffn_up_shexp.weight` | 48 | 43238400 | 0.04 | IQ3_S x14, IQ4_XS x26, Q6_K x8 |
| `indexer.q_proj.weight` | 12 | 31457280 | 0.03 | BF16 x12 |
| `ple_value.weight` | 1 | 13107200 | 0.01 | BF16 x1 |
| `attn_v.weight` | 12 | 10250240 | 0.01 | IQ4_XS x7, Q6_K x5 |
| `attn_k.weight` | 12 | 9359360 | 0.01 | IQ3_S x1, IQ4_XS x8, Q6_K x3 |
| `ssm_alpha.weight` | 36 | 8847360 | 0.01 | BF16 x36 |
| `ssm_beta.weight` | 36 | 8847360 | 0.01 | BF16 x36 |
| `indexer.k_proj.weight` | 12 | 7864320 | 0.01 | BF16 x12 |
| `output_hc_down.weight` | 1 | 6553600 | 0.01 | BF16 x1 |
| `output_hc_up.weight` | 1 | 6553600 | 0.01 | BF16 x1 |
| `ssm_conv1d.weight` | 36 | 5898240 | 0.01 | F32 x36 |
| `hc_attn_inject.weight` | 48 | 3932160 | 0.00 | BF16 x48 |
| `hc_ffn_inject.weight` | 48 | 3932160 | 0.00 | BF16 x48 |
| `hc_attn_norm.weight` | 48 | 1966080 | 0.00 | F32 x48 |
| `hc_ffn_norm.weight` | 48 | 1966080 | 0.00 | F32 x48 |
| `ffn_gate_inp_shexp.weight` | 48 | 491520 | 0.00 | F32 x48 |
| `ple_conv1d.weight` | 1 | 81920 | 0.00 | F16 x1 |
| `output_hc_norm.weight` | 1 | 40960 | 0.00 | F32 x1 |
| `ple_norm_conv.weight` | 1 | 40960 | 0.00 | F32 x1 |
| `ple_norm_key.weight` | 1 | 40960 | 0.00 | F32 x1 |
| `ple_norm_query.weight` | 1 | 40960 | 0.00 | F32 x1 |
| `ssm_norm.weight` | 36 | 18432 | 0.00 | F32 x36 |
| `attn_k_norm.weight` | 12 | 12288 | 0.00 | F32 x12 |
| `attn_q_norm.weight` | 12 | 12288 | 0.00 | F32 x12 |
| `ssm_a` | 36 | 6912 | 0.00 | F32 x36 |
| `ssm_dt.bias` | 36 | 6912 | 0.00 | F32 x36 |
| `indexer.k_norm.weight` | 12 | 6144 | 0.00 | F32 x12 |
| `indexer.q_norm.weight` | 12 | 6144 | 0.00 | F32 x12 |

Full per-tensor JSON (1224 rows) for each tier is reproducible with `--json`.

### Notes on this table

- **Per-layer expert bytes vary** because GSQ/RCO mixes quant types per layer. IQ2_XS expert bytes per
  layer are 0.56 GiB (`IQ1_M` gate/up), 0.62 GiB (`IQ2_XXS` gate/up) or 0.72 GiB (`IQ2_S` gate/up), all
  with `Q2_0` `ffn_down`. This matters for the placement rule: a *layer-count* rule
  (`--n-cpu-moe N`) yields slightly different byte counts than a *byte-budget* rule.
- **Hyper-connections cost real VRAM for no inference benefit on the GPU**: the 8 `hc_*` classes per layer
  (16 BF16 + 4 F32 tensors) total **1,270,087,680 B = 1.18 GiB** of IQ2_XS's 3.62 GiB non-expert, non-ngram
  weight budget (32.6%). They are the single largest non-expert block after attention.
- **`ffn_gate_inp.weight` is F32** (48 × 5.24 MB = 0.23 GiB): the router is unquantised in every tier. It
  must stay on the GPU — it is read once per token per layer.
- **No MTP / NextN tensors exist in this GGUF.** A scan of all 1224 names for `nextn|mtp|draft|eagle`
  returns nothing, `qwen4exp.block_count` is 48, and the only blocks are `blk.0`–`blk.47`. The base model
  *and* the Swift checkpoint both carry a 1-layer MTP head (31 `mtp.*` tensors), but llama.cpp's `qwen4exp`
  conversion drops it and the `qwen4exp` runtime has no MTP path. **Speculation on bongo must therefore come
  from the n-gram/PLE path, not from an MTP head.** The full evidence is in
  [intel-arc-b70.md](intel-arc-b70.md) §2.1; the old assumption in that doc's §2/§5 is corrected there.

## 5. Tier totals

| Tier | Experts | n-gram table | Shared experts | Embed + head | Other (attn/ssm/hc/norm/router) | Tensor bytes | File bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IQ2_XS` | 33.02 GiB | 26.82 GiB | 0.13 GiB | 0.63 GiB | 2.86 GiB | 63.46 GiB | 68.15 GB |
| `IQ3_XXS` | 39.97 GiB | 26.82 GiB | 0.11 GiB | 0.66 GiB | 3.18 GiB | 70.74 GiB | 75.97 GB |
| `Q2_0` | 31.64 GiB | 26.82 GiB | 0.10 GiB | 0.66 GiB | 2.75 GiB | 61.97 GiB | 66.55 GB |

"Non-expert bytes" = tensor bytes − experts = 30.44 GiB (IQ2_XS) / 30.77 GiB (IQ3_XXS) / 30.33 GiB
(Q2_0), of which **26.82 GiB is the n-gram table**. So the weights that genuinely must be VRAM-resident
are only **3.62 GiB / 3.95 GiB / 3.51 GiB**.

This replaces the intel-arc-b70.md §2 "expert bytes (approx)" column (`~34/36/43 GB`) with measured
values: **35.45 GB / 42.91 GB / 33.97 GB** (decimal).

## 6. Buffer placement plan: 32 GB VRAM + 30 GiB RAM

### 6.1 The budget

VRAM is 32 GB decimal = **29.80 GiB**. RAM is **30 GiB** (≈ 23–24 GiB free while the agent stack runs on
this box).

| Bucket | IQ2_XS | IQ3_XXS | Q2_0 | Where | Basis |
| --- | ---: | ---: | ---: | --- | --- |
| Driver / display reserve | 0.50 | 0.50 | 0.50 | VRAM | assumption |
| Graph / compute buffers | 1.50 | 1.50 | 1.50 | VRAM | assumption (measure) |
| KV cache @128K, `--cache-type-k/v q8_0` | 1.50 | 1.50 | 1.50 | VRAM | 12 full-attn layers × 2 kv heads × 256 × 2 (K,V) × 1 B = 12 KiB/token × 131072 |
| Indexer KV + linear-attention state | 0.30 | 0.30 | 0.30 | VRAM | est. from `attention.indexer.*`, `ssm.state_size` |
| Non-expert, non-ngram weights | 3.62 | 3.95 | 3.51 | VRAM | **measured** |
| Expert weights | 33.02 | 39.97 | 31.64 | VRAM + RAM | **measured** |
| N-gram table | 26.82 | 26.82 | 26.82 | SSD, page cache | **measured** |

The combined expert pool is VRAM ≤ 29.80 − 0.50 − 1.50 − 1.50 − 0.30 − weights ≈ **22.4 / 22.1 / 22.5 GiB**
plus whatever RAM we are willing to give it. Total capacity (≈ 22.4 + 24 ≈ 46 GiB) is **larger than the
IQ3_XXS expert set (39.97 GiB)** — so every tier *fits* in the combined buffer. The binding constraints
are (a) how many expert bytes we can keep on the GPU, and (b) how many we can keep **resident in RAM at
the same time as a usable n-gram page cache**.

### 6.2 The proposed rule

llama.cpp's default puts every tensor on the GPU, which is impossible here because the n-gram table alone
is 26.82 GiB. The rule is therefore explicit:

```sh
# llama-server / llama-cli, IQ2_XS on the reference box (32 GB VRAM + 30 GiB RAM)

# 0) Keep the default loading path: mmap ON, --lazy-mode auto. The n-gram table (26.82 GiB)
#    is then read on demand from the file instead of holding a buffer. Do NOT pass
#    --no-mmap or --lazy-mode off: either makes the whole table resident and OOMs.

# 1) The n-gram table must stay in the file on the CPU side. This is already what
#    --lazy-mode auto does; the line is a guard, not the mechanism.
-ot "per_layer_token_embd\.weight=CPU"

# 2) Static expert split: keep 48-N layers of experts on the GPU, offload the first
#    N layers' experts to the CPU.  N = 19 for IQ2_XS (see the table below).
--n-cpu-moe 19

# exact equivalent of (2), if you want the layer set to be explicit and independent
# of llama.cpp's "first N layers" convention. This is byte-for-byte what
# `--n-cpu-moe 19` expands to (llama.cpp master, common/common.h:
#   LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
#   llm_ffn_block_regex(i, ...) = "blk\\.<i><regex>"  for i in 0..N-1):
#   -ot "blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU" \
#   ... one line per layer 0..18 ...

# everything else defaults to the GPU:
-n --n-gpu-layers 999
```

`--n-cpu-moe N` offloads only `ffn_*_exps` (the routed experts); shared experts, routers, attention, SSM
and the hyper-connections stay on the GPU, which is what the plan wants. It does **not** touch
`per_layer_token_embd.weight` (the pattern is `\.ffn_(up|down|gate|gate_up)_(ch|)exps`), so the `-ot`
line for the n-gram table is mandatory. Upstream semantics verified against llama.cpp master
`common/arg.cpp` ("keep the Mixture of Experts (MoE) weights of the first N layers in the CPU") on
2026-09-27; `--cpu-moe` is the all-layers form of the same override.

One caveat: the two overrides must not conflict. `-ot "per_layer_token_embd\.weight=CPU"` is a separate
pattern from the expert patterns, so they compose cleanly. Buffer type `CPU` is the registered name of
`ggml_backend_cpu_buffer_type()`.

Recommended `N` per tier, chosen as the largest GPU residency that still leaves ≥ 1.5 GiB of VRAM
headroom against the budget above:

| Tier | Rule | Expert bytes on GPU | Expert bytes on CPU (RAM-resident) | Non-expert VRAM | KV+state | Buffers+driver | **Estimated VRAM used** | Headroom |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IQ2_XS` | `--n-cpu-moe 19` | 20.34 GiB | 12.68 GiB (13.62 GB) | 3.62 | 1.80 | 2.00 | **27.76 GiB** of 29.80 | 2.04 GiB |
| `IQ3_XXS` | `--n-cpu-moe 27` | 19.96 GiB | 20.01 GiB (21.48 GB) | 3.95 | 1.80 | 2.00 | **27.71 GiB** of 29.80 | 2.09 GiB |
| `Q2_0` | `--n-cpu-moe 17` | 20.43 GiB | 11.21 GiB (12.03 GB) | 3.51 | 1.80 | 2.00 | **27.74 GiB** of 29.80 | 2.06 GiB |

Resulting RAM picture (30 GiB total):

| Tier | CPU-resident experts | OS + runtime + agent stack | Left for n-gram page cache | Comment |
| --- | ---: | ---: | ---: | --- |
| `IQ2_XS` | 12.68 GiB | ~4 GiB (dedicated) / ~6.5 GiB (shared with agents) | **~13 GiB / ~10.5 GiB** | comfortable; ~40% of the n-gram table can live in cache |
| `IQ3_XXS` | 20.01 GiB | ~4 GiB / ~6.5 GiB | **~6 GiB / ~3.5 GiB** | **the tier that is at risk**; page-cache misses on the n-gram table will hurt |
| `Q2_0` | 11.21 GiB | ~4 GiB / ~6.5 GiB | **~14.8 GiB / ~12.3 GiB** | comfortable, same as IQ2_XS but slightly lower quality (KLD 0.424) |

Per-token work split for IQ2_XS `--n-cpu-moe 19`: 19 of 48 layers run their 10 active experts on the CPU
(190 expert FFNs/token) and 29 layers run theirs on the GPU (290 expert FFNs/token). The GPU still holds
all attention, SSM, routers, shared experts, the output head and the KV cache, so the CPU is doing
matrix work on a hot 12.68 GiB resident set rather than serialising the whole forward pass.

### 6.3 Choosing N differently

For a *byte* budget instead of a layer count, use the explicit `-ot` form and include the first K layers
whose expert bytes sum to the target. Aggressive/conservative alternatives for IQ2_XS:

| `--n-cpu-moe N` | CPU expert GiB | GPU expert GiB | Estimated total VRAM | Headroom |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 10.62 | 22.40 | 29.82 | −0.02 (does not fit) |
| 17 | 11.34 | 21.68 | 29.10 | 0.70 |
| 18 | 12.06 | 20.96 | 28.38 | 1.42 |
| **19** | **12.68** | **20.34** | **27.76** | **2.04** |
| 20 | 13.31 | 19.71 | 27.13 | 2.67 |
| 21 | 14.03 | 18.99 | 26.41 | 3.39 |
| 24 | 16.19 | 16.83 | 24.25 | 5.55 |
| 28 | 18.97 | 14.05 | 21.47 | 8.33 |
| 32 | 21.66 | 11.36 | 18.78 | 11.02 |

`N = 0` (all experts on the GPU) needs 36.64 GiB for weights alone and does not fit at any context. The
smallest N that fits at 128K is 17, and it fits with only 0.70 GiB to spare; start at 19 and only go
below it after measuring real VRAM use.

### 6.4 Why this is the floor, not the design

This static split is a **baseline** to be measured, not bongo's end state. It has two known inefficiencies
that the Stage 1 work (expert placement, [BAS-53](/BAS/issues/BAS-53)) is meant to remove:

1. The GPU-resident expert set is chosen by *layer index*, not by *usage*. Routing is concentrated, so a
   hot-expert cache of the same 20.3 GiB would serve many more tokens per byte than "layers 19–47" does.
2. Offloaded experts are computed on the CPU. With an adaptive cache, the CPU would compute only the
   experts that miss the cache in that step, and the split would follow traffic instead of a fixed line.

The concrete measurable success condition for the baseline is: `./bongo.sh` boots IQ2_XS at 128K with
`--n-cpu-moe 19` and `-ot per_layer_token_embd.weight=CPU` without OOM, no expert page-cache thrashing
(major-fault rate stable per token), and a recorded prompt/output tok/s. [BAS-52](/BAS/issues/BAS-52)
owns the numbers.

## 7. Open questions and what must be measured

1. **VRAM accounting.** Driver reserve, graph buffers and the KV/indexer/SSM figures above are estimates.
   The measured total from `xpu-smi` after load decides the real `N`. (blocks [BAS-52](/BAS/issues/BAS-52))
2. **N-gram page-cache behaviour.** 16 random 90-byte rows per token over 26.8 GiB implies ~64 KiB of page
   traffic per token. The mechanism to make this work exists upstream (`--lazy-mode auto` + mmap, §3), but
   nobody has measured it on `xe`/SYCL: (a) confirm the SYCL device reports `mmap_support == true`, or
   `auto` silently degrades to `off`; (b) measure major faults/token and tok/s against a cold and a warm
   page cache. This is the highest-risk item in the plan, and it is the same risk Strata describes for its
   SSD-resident table. (blocks [BAS-53](/BAS/issues/BAS-53))
3. **IQ3_XXS viability.** It fits the combined buffer (39.97 GiB experts against ~46 GiB of pool) but
   leaves only ~3.5–6 GiB of RAM for the n-gram cache at `N = 27`. Either accept more VRAM pressure
   (`N = 27`) and measure the cache, or treat IQ3_XXS as needing SSD expert streaming. Do not assume it
   fits because the totals add up.
4. **No MTP head in the GGUF — resolved, not just absent.** The base model and the Swift checkpoint do
   have a 1-layer MTP head, but the published GGUF drops it and llama.cpp `qwen4exp` cannot convert or run
   one, so `--spec-draft-*` / MTP work is invalid for this model (not merely deferred). See
   [intel-arc-b70.md](intel-arc-b70.md) §2.1.
5. **`--no-mmap` must not be used** with this model: it would try to make all 68 GB resident, including
   the 26.82 GiB n-gram table, and it also switches the PLE table out of lazy-read mode
   (`lazy_read::add` returns false without mmap). Keep the default mmap so the n-gram table stays
   file-backed and reclaimable. The same applies to `--lazy-mode off`.
6. **Which heads are actually read** per token is now answered from upstream source:
   `ple_n_heads = (ngram_size − 1) × heads_per_ngram = 16`, so all 16 heads are consulted per token.
