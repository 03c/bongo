# BAS-76 — byte-budget `-ot` placement + held-out LRU A/B

Raw results for the two steps of [BAS-76](/BAS/issues/BAS-76). Analysis:
[`docs/research/dynamic-expert-lru.md`](../../../docs/research/dynamic-expert-lru.md).

## Files

| file | what |
| --- | --- |
| `placement-iq2_xs-22.40.json` | Step-1 placement at the 22.40 GiB budget: resident/CPU layers, the exact `-ot` pattern, coverage, and the `--n-cpu-moe 16` delta. |
| `lru-ab.json` | Step-2 held-out A/B: per-corpus coverage for frozen profile, cold LRU, profile-LRU, per-layer LRU, and the two whole-layer baselines. |
| `lru-ab.txt` | Human-readable console output of `bench/sim-expert-lru.py`. |
| `byte-budget-22_40/` | Step-1 GPU run, byte-budget `-ot` (matrix + raw + prefix-cache + exact `server-flags.json`). |
| `ncmoe-16/` | The pinned Stage 0 baseline re-run under the same protocol, same GPU session. |
| `placement-ab/` | Computed A/B (`placement-ab.json` + `.md`) from the two `matrix.json` files. |

Every number is on llama.cpp `b11223-4da633776`, Vulkan (`--device Vulkan1`), tier
`iq2_xs`, context 131072, with the exact argv in each run's `server-flags.json`.

## Numbers

### Offline (no GPU)

- Step-1 coverage (whole-layer event fraction, exact for whole-layer residency): byte-budget
  `-ot` at 22.40 GiB keeps **33 resident layers / 22.217 GiB**, coverage **0.7021**; the pinned
  `--n-cpu-moe 16` keeps 32 layers / 22.40 GiB, coverage **0.6596** → **+4.25 pp** at slightly
  *less* VRAM (inside the R4 +4–8 pp window).
- Step-2 held-out prefill coverage, `profile_lru` / frozen / cold LRU:
  doc 0.993 / 0.963 / 0.987; code 0.986 / 0.881 / 0.980; chat 0.988 / 0.985 / 0.947; convo 0.992 / 0.977 / 0.969.
  Decode held out, `profile_lru` / frozen / cold LRU: doc 0.985 / 0.945 / 0.903; chat 0.986 / 0.962 / 0.862.
- Verdict: the conflict is resolved in favour of a **dynamic LRU seeded by the profile**; the
  frozen profile is not shipped as the policy. R4 reproduces (0.881–0.985); R3's "static ~10%,
  LRU 67–81%" is a different model/workload regime.

### GPU A/B (same session, agentic `cache_prompt=true`)

`bench/compare-placement-ab.py` over `byte-budget-22_40/matrix.json` (A) and
`ncmoe-16/matrix.json` (B):

| context | A prompt tok/s | B prompt tok/s | Δ prompt | A output tok/s | B output tok/s | Δ output | A TTFT ms | B TTFT ms | Δ TTFT | A VRAM GiB | B VRAM GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096* | 1.143 | 11.791 | — | 14.772 | 14.626 | +1.00% | 2678.5 | 312.2 | — | 29.05 | 29.07 |
| 131072 | 135.224 | 131.395 | **+2.91%** | 8.010 | 7.951 | **+0.74%** | 937660.6 | 964978.7 | **−2.83%** | 29.29 | 29.51 |

\* The 4096 row is a cache artifact of the agentic profile (the first measured 4K request did not
reuse the same warm slot the baseline did); it is **not** used. The authoritative 4K/agentic-turn
path is `measure-prefix-cache.py` below.

- 128K **needle: pass** for both configs (the needle is folded into the 128K harness run).
- Step-1 acceptance: **met** — +4.25 pp coverage at slightly less VRAM, and no 128K decode
  regression (+0.74%). Prefill +2.91% and TTFT −2.83% move in the right direction. The gain is
  single-digit percent, as the analysis predicted for whole-layer residency (32 → 33 layers);
  R4's +18–37% assumed per-expert residency, which is the Step-2 engine.

### `measure-prefix-cache.py` (byte-budget, same session)

| case | prompt tok/s | TTFT ms | output tok/s |
| --- | ---: | ---: | ---: |
| cold_p4096 | 172.53 | 23799.6 | 13.88 |
| grow_p4096_d512 | 143.62 | 3583.4 | 13.60 |
| cold_p16384 | 202.86 | 80778.1 | 10.31 |
| grow_p16384_d512 | 132.56 | 3898.1 | 9.34 |
| cold_p24576 | 190.36 | 129115.8 | 9.67 |
| grow_p24576_d512 | 113.48 | 4556.6 | 11.24 |
| slot save/erase/restore + after-restore | — | 257.6 | — (restore verified) |

The same-session baseline prefix-cache was not run (the sweep ran it only for the placement branch;
fixed in `bench/sweep-byte-budget-placement.sh` for future runs). The recorded
[`../2026-09-28-prefix-cache/`](../2026-09-28-prefix-cache/) baseline is a different session and
`--ctx-size 32768`, so it is indicative only, not a controlled A/B.

## Reproduce

```sh
python3 bench/gen-ot-placement.py \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --analysis bench/results/2026-09-28-expert-activation/analysis.json \
  --budget-gib 22.40 --tier iq2_xs \
  --out bench/results/2026-09-28-byte-budget-placement/placement-iq2_xs-22.40.json

python3 bench/sim-expert-lru.py \
  --raw bench/results/2026-09-28-expert-activation/raw \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --budget-gib 22.40 --corpora doc,code,chat,convo \
  --decode doc=bench/results/2026-09-28-expert-activation/raw/doc_dec.tsv.gz,chat=bench/results/2026-09-28-expert-activation/raw/chat_dec.tsv.gz \
  --out bench/results/2026-09-28-byte-budget-placement/lru-ab.json

# GPU A/B (needs an idle Arc Pro B70; refuses to start if port 8080 is busy, takes the shared lock)
bench/sweep-byte-budget-placement.sh
bench/sweep-byte-budget-placement.sh --n-cpu-moe 16

# computed A/B
python3 bench/compare-placement-ab.py \
  --a bench/results/2026-09-28-byte-budget-placement/byte-budget-22_40/matrix.json \
  --b bench/results/2026-09-28-byte-budget-placement/ncmoe-16/matrix.json \
  --a-label byte-budget-22.40 --b-label ncmoe-16 \
  --out bench/results/2026-09-28-byte-budget-placement/placement-ab
```

### Metadata correction

The harness auto-discovers server pids by scanning `/proc/*/cmdline` for the server name. On the
busy shared box it also matched a concurrent shell whose command line happened to contain that
name, so the first discovered pid (and therefore `matrix.json` `server.flags`) was wrong. The
`server.pids` / `server.flags` fields in both `matrix.json` files were corrected post-hoc from the
authoritative `server-flags.json` written at launch; no measured number changed.
`bench/sweep-byte-budget-placement.sh` now passes `--server-pid "$pid"` so this cannot recur.
