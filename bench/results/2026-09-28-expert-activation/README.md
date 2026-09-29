# 2026-09-28 — expert-activation skew (BAS-66 / R4)

Raw per-`(layer, expert)` MoE router captures from the actual IQ2_XS model on the reference box, and the
coverage analysis that decides whether a hot-expert residency beats the static `--n-cpu-moe` layer rule.

**Findings:** [`docs/research/expert-activation-skew.md`](../../../docs/research/expert-activation-skew.md)

## Contents

| path | what |
| --- | --- |
| `raw/*.tsv.gz` | raw `ffn_moe_topk` captures (prefill + teacher-forced decode) |
| `corpora/*.txt` | the exact prompt inputs |
| `analysis.json` | all computed numbers (coverage curves, transfers, concentration, budget) |
| `coverage.txt` | human-readable analysis summary |
| `run-environment.md` | box, engine, model, command lines, token counts, timings |

## Headline

At the shipped `--n-cpu-moe 16` expert budget (22.40 GiB), a frequency-ranked hot set serves **98.5%** of
routed expert selections versus **66.0%** for the layer rule (+32.5 pp), and **88-99%** out of sample. The gap
is real and large, but the measured throughput response says it buys **prefill (+18-27% at 128K, +25-37% at
4K)** and **4K decode (+38-57%)**, and **~nothing at 128K decode**, which is attention/KV-bound.

## Reproduce

```sh
bench/run-expert-activation.sh --model <first GGUF shard> --decode 256
python3 bench/analyze-expert-activation.py \
  --raw bench/results/2026-09-28-expert-activation/raw \
  --decode doc=bench/results/2026-09-28-expert-activation/raw/doc_dec.tsv.gz,chat=bench/results/2026-09-28-expert-activation/raw/chat_dec.tsv.gz \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --out bench/results/2026-09-28-expert-activation/analysis.json \
  --corpora doc,code,chat,convo
```
