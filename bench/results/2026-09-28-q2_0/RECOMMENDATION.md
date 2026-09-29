# Q2_0 tier at 128K on the Arc Pro B70 — measurement + recommendation

- Date: 2026-09-28
- Issue: [BAS-59](/BAS/issues/BAS-59) (feasibility measurement only)
- Hardware: Intel Arc Pro B70 ("Battlemage G31", 32 GiB VRAM), AMD Ryzen 7 9700X, 32 GB system RAM, Fedora 44
- Engine: llama.cpp **b11223** (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), **Vulkan** backend
  (`--device Vulkan1`; no Level Zero/SYCL on this host, per [ADR-0002](../../../docs/adr/0002-baseline-engine.md))
- Model: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, tier **Q2_0**

## Decision

**Keep IQ2_XS as the default tier. Q2_0 is feasible at 128K on this box, but it is a
quality *down*grade and a decode *regression*, not the step up the user asked for.**

- Q2_0's development KLD is **0.4244**, worse than IQ2_XS's **0.3413** (lower is better).
  It is a *smaller but lower-quality* quant, not a higher tier. It is not "above IQ2_XS"
  in the quality sense.
- Decode is **11–26% slower** than the IQ2_XS baseline at every context (128K: 6.85 vs
  7.98 tok/s).
- Prefill/TTFT is **~12–25% faster** and peak system RAM is much lower (6.6 vs 14.1 GiB),
  but that is not worth a quality downgrade for the default tier.
- The actual quality upgrade, **IQ3_XXS (KLD 0.2401), is infeasible on this hardware** —
  see [IQ3_XXS](#iq3_xxs-is-infeasible-on-this-hardware) below.

Q2_0 can stay as an optional tier for prefill-dominated workloads or memory-tight hosts,
but it should not replace IQ2_XS.

## Proprietary provenance

Download command:

```sh
./bongo.sh --tier q2_0 \
  --llama-bin ~/.bongo/llama/b11223/vulkan \
  --runtime dir --runtime-dir ~/.bongo/runtime-empty \
  --backend vulkan --n-cpu-moe 15 --detach --yes
```

Shard SHA-256 (full-file `sha256sum`; both match the repo `SHA256SUMS`):

| shard | bytes | sha256 |
| --- | ---: | --- |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 39,799,117,984 | `79e2a3873f1b05df835a1a191b20f2e74b9e12df949cd89861d17d98b6ef9678` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | 26,750,834,816 | `ef3bb04fb4e14f04624abd8acf25faf1603594a034050b048df85ec152ad5ae0` |

- llama.cpp commit: `4da6337767f973e2b4d0797e5b323d77d8565e4a` (build `b11223`).
- Tensor inventory and per-layer expert bytes come from `tools/gguf-inventory.py` (header
  range reads only, no weight download). Q2_0's routed experts are uniform at
  **0.6592 GiB/layer** (707,788,800 B), **31.64 GiB total**; recorded in
  [`expert-bytes-q2_0.json`](expert-bytes-q2_0.json).

## Finding the smallest `--n-cpu-moe` that fits 128K

`--n-cpu-moe N` offloads the routed experts of the first N layers to the CPU; smaller N =
more expert bytes on the GPU = more VRAM pressure. The IQ2_XS Stage 0 baseline sat at the
edge (`n=16` fits, `n=12` device-loses at 128K). Q2_0 has ~1.4 GiB fewer expert bytes, so
the edge moves down.

Command pattern (the sweep script restarts the server per config, warms the page cache
with a discarded 4K prefill, then runs the harness at 4096 and 131072):

```sh
BONGO_SWEEP_TIER=q2_0 \
BONGO_SWEEP_OUT=bench/results/2026-09-28-q2_0/sweep \
BONGO_SWEEP_CONFIGS=16,15,12,11,10 \
BONGO_SWEEP_CONTEXTS=4096,131072 \
./bench/sweep-expert-placement.sh
```

Raw per-config records: [`sweep/ncmoe-<N>/`](sweep/) (`matrix.json`, `matrix.md`, raw
requests, placement-after-load, server logs). Summary: [`sweep-summary.json`](sweep-summary.json).

| `--n-cpu-moe` | loads | fits 128K | GPU experts GiB | CPU experts GiB | 4K prompt tok/s | 4K output tok/s | 128K prompt tok/s | 128K output tok/s | 128K VRAM GiB | needle |
| ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | yes | **no** | 25.05 | 6.59 | 296.9 | 11.19 | *device lost* | *device lost* | 31.85 (peak) | fail |
| **11** | yes | **yes** | **24.39** | **7.25** | 288.9 | 12.38 | 151.4 | 7.44 | 31.20 | **pass** |
| 12 | yes | yes | 23.73 | 7.91 | 279.0 | 8.81 | 149.7 | 7.63 | 30.54 | pass |
| 15 | yes | yes | 21.75 | 9.89 | 256.3 | 9.82 | 141.6 | 6.81 | 28.56 | pass |
| 16 | yes | yes | 21.09 | 10.55 | 247.9 | 6.98 | 139.0 | 6.91 | 27.90 | pass |

**Smallest fitting `--n-cpu-moe` = 11.** `n=16` (the sweep's starting anchor) and `n=15`
both fit comfortably; `n=10` loads (31.34 GiB after load) and serves 4K, then loses the
device at 128K:

```
context 131072 failed: {"error":{"code":500,"message":"decode() failed: vk::Queue::submit: ErrorDeviceLost",...}}
```

This is the same failure mode as IQ2_XS `n=12`: the 128K KV cache and compute buffers push
a config that fits at load over the ~31.92 GiB usable-VRAM edge. `n=11` peaks at
31.20 GiB — only ~0.7 GiB of headroom — so it is the *edge* config, not a comfortable
default. The shipped `bongo.sh` default for Q2_0 (`--n-cpu-moe 15`, 28.56 GiB peak) is the
conservative choice.

## Full benchmark (1K/4K/32K/128K) at the smallest fitting N

Server:

```sh
./bongo.sh --tier q2_0 \
  --gguf-dir ~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/q2_0 \
  --llama-bin ~/.bongo/llama/b11223/vulkan \
  --runtime dir --runtime-dir ~/.bongo/runtime-empty \
  --backend vulkan --n-cpu-moe 11 --detach --yes

./bench/run.sh --tier q2_0 --repeats 3 --repeats-deep 1 --deep-threshold 32768 \
  --out-dir bench/results/2026-09-28-q2_0 --hash-mode full
```

| context | prompt tokens | prompt tok/s | output tok/s | TTFT ms | prefill ms | repeats |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1025 | 254.6 | 14.79 | 4029 | 19617 | 3 |
| 4096 | 4097 | 288.3 | 13.24 | 14214 | 15103 | 3 |
| 32768 | 32768 | 208.0 | 10.36 | 157585 | 157561 | 1 |
| 131072 | 130861 | 150.8 | 6.85 | 867641 | 867540 | 1 |

- Peak VRAM: **31.20 GiB** (`fdinfo`); peak system RAM: **6.60 GiB** (`proc_status_sum`).
- 128K needle: **pass** (`VAULT-COORD-7391-QXZ`, 130,861 prompt tokens), so the 128K window
  is usable, not merely accepted.
- Error/malformed cases all handled (400/500 where expected); see [`matrix.md`](matrix.md).

### Against the IQ2_XS Stage 0 baseline

Baseline: [`bench/results/2026-09-27-baseline/matrix.md`](../2026-09-27-baseline/matrix.md)
(IQ2_XS, `--n-cpu-moe 16`, peak VRAM 29.27 GiB, peak RSS 14.07 GiB).

| metric | IQ2_XS `n=16` | Q2_0 `n=11` | Q2_0 vs IQ2_XS |
| --- | ---: | ---: | ---: |
| 4K prompt tok/s | 231.5 | 288.3 | **+24.6%** |
| 4K output tok/s | 17.71 | 13.24 | **-25.3%** |
| 128K prompt tok/s | 133.2 | 150.8 | **+13.3%** |
| 128K output tok/s | 7.98 | 6.85 | **-14.2%** |
| 128K TTFT | 982,856 ms | 867,641 ms | **-11.7%** |
| peak VRAM | 29.27 GiB | 31.20 GiB | +1.93 GiB |
| peak RSS | 14.07 GiB | 6.60 GiB | **-7.47 GiB** |
| dev KLD (lower better) | 0.3413 | 0.4244 | **worse** |

## Quality: Q2_0 is not a higher tier

From the model authors' `release-manifest.json` (`development_kld`, lower is better):

| tier | tensor bytes | dev KLD | file bytes |
| --- | ---: | ---: | ---: |
| IQ2_XS | 63.46 GiB | 0.3413 | 68.15 GB |
| **Q2_0** | **61.97 GiB** | **0.4244** | **66.55 GB** |
| IQ3_XXS | 70.74 GiB | 0.2401 | 75.97 GB |

Q2_0 is smaller *and* lower quality than IQ2_XS. The KLD is a 512-token evaluation context,
so it does not capture long-context degradation; the 128K needle only proves retrieval, not
general quality.

## IQ3_XXS is infeasible on this hardware

IQ3_XXS is the only true quality upgrade (KLD 0.2401), and it is **out of reach on 32 GB
VRAM + 32 GB RAM**, with its byte size as the reason:

- Tensor bytes **75,955,048,960 B = 70.74 GiB** (file bytes 75.97 GB); the routed expert set
  alone is **42.91 GB / 39.97 GiB**.
- The all-GPU expert set does not fit: IQ2_XS `--n-cpu-moe 0` already fails to load with
  33.02 GiB of experts, so 39.97 GiB is impossible.
- Keeping enough experts on the CPU to fit VRAM (IQ3_XXS needs `--n-cpu-moe ~27`) leaves
  only ~3.5–6 GiB of RAM for the 26.82 GiB n-gram table at 128K, versus ~12–14 GiB for
  IQ2_XS/Q2_0 — it cannot hold a usable page cache and would thrash.
- Streaming IQ3_XXS experts from SSD is explicitly out of scope for this measurement.

Do not expect IQ3_XXS to fit because the combined VRAM+RAM totals add up; the binding
constraints are per-pool, not the sum.

## Limitations

- Single repeat at 32768 and 131072 (a 128K prefill is ~14–15 min here); 3 repeats at
  1024/4096. 1K TTFT `cv` is high because the first 1K repeat is a cold prefill.
- Vulkan only. These numbers do not transfer to a working SYCL backend (SYCL is being
  restored under [BAS-57](/BAS/issues/BAS-57)); its MoE kernels may move the balance.
- The full bench ran at the *edge* config `n=11`. The `n=15` default is safer
  (28.56 GiB peak) and its 128K numbers are in the sweep table and `sweep/ncmoe-15/`.
- `n=12`'s sweep record has no `matrix.json` because its harness run was interrupted during
  the (pre-fix) unbounded `negative_max_tokens` error case; its 4K/128K/needle raw results
  are complete and are the source for the row above.

## Harness fix landed with this measurement

The `negative_max_tokens` error case sends `max_tokens: -1`, which llama.cpp treats as
"generate until EOS/context". One runaway generation blocked every later error case and
counted against the 3600 s client timeout (it truncated the IQ2_XS `n=48` sweep). The case
is now **last** in the suite and bounded by `--error-timeout` (default 120 s,
`BONGO_ERROR_TIMEOUT`); a timeout is recorded as "server accepted the negative value and
generated unboundedly". No measured result changes and `bash bench/selftest.sh` still
passes. Commit `61e8720`.
