# BAS-76 — byte-budget `-ot` placement + held-out LRU A/B

Raw results for the two steps of [BAS-76](/BAS/issues/BAS-76). Analysis:
[`docs/research/dynamic-expert-lru.md`](../../../docs/research/dynamic-expert-lru.md).

## Files

| file | what |
| --- | --- |
| `placement-iq2-xs-22.40.json` | Step-1 placement at the 22.40 GiB budget: resident/CPU layers, the exact `-ot` pattern, coverage, and the `--n-cpu-moe 16` delta. |
| `lru-ab.json` | Step-2 held-out A/B: per-corpus coverage for frozen profile, cold LRU, profile-LRU, per-layer LRU, and the two whole-layer baselines. |
| `lru-ab.txt` | Human-readable console output of `bench/sim-expert-lru.py`. |
| `byte-budget-22_40/` | Step-1 GPU re-run (Stage-1 sweep, 4K + 128K, VRAM, prefix cache) — written by `bench/sweep-byte-budget-placement.sh`. |
| `ncmoe-16/` | The pinned Stage-0 baseline re-run under the same protocol. |

## Numbers (offline, no GPU)

- Byte-budget `-ot` at 22.40 GiB: **33 resident layers / 22.217 GiB**, coverage **0.7021** vs
  `--n-cpu-moe 16` **0.6596** → **+4.25 pp**.
- Held-out prefill coverage, `profile_lru` / frozen / cold LRU:
  doc 0.993 / 0.963 / 0.987; code 0.986 / 0.881 / 0.980; chat 0.988 / 0.985 / 0.947; convo 0.992 / 0.977 / 0.969.
- Held-out decode coverage, `profile_lru` / frozen / cold LRU: doc 0.985 / 0.945 / 0.903; chat 0.986 / 0.962 / 0.862.

## Reproduce

```sh
python3 bench/gen-ot-placement.py \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --analysis bench/results/2026-09-28-expert-activation/analysis.json \
  --budget-gib 22.40 --tier iq2_xs \
  --out bench/results/2026-09-28-byte-budget-placement/placement-iq2-xs-22.40.json

python3 bench/sim-expert-lru.py \
  --raw bench/results/2026-09-28-expert-activation/raw \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --budget-gib 22.40 --corpora doc,code,chat,convo \
  --decode doc=bench/results/2026-09-28-expert-activation/raw/doc_dec.tsv.gz,chat=bench/results/2026-09-28-expert-activation/raw/chat_dec.tsv.gz \
  --out bench/results/2026-09-28-byte-budget-placement/lru-ab.json

# GPU re-run (needs an idle Arc Pro B70; refuses to start if port 8080 is busy)
bench/sweep-byte-budget-placement.sh
bench/sweep-byte-budget-placement.sh --n-cpu-moe 16
```
