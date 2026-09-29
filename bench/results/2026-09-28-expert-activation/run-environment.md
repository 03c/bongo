# Run environment — expert-activation capture (BAS-66)

Raw router captures and the analysis for
[`docs/research/expert-activation-skew.md`](../../../docs/research/expert-activation-skew.md).

## Box

| | |
| --- | --- |
| CPU | AMD Ryzen 7 9700X (8C/16T) |
| RAM | 30 GiB |
| GPU | Intel Arc Pro B70 ("Battlemage G31"), 32 GiB VRAM, `xe` driver |
| OS | Fedora 44 |
| Kernel-visible usable VRAM | 31.92 GiB (`0x7f9000000`, 32.00 GiB physical minus stolen) |

Same box as the Stage-0 baseline and the Stage-1 `--n-cpu-moe` sweep.

## Engine and model

| | |
| --- | --- |
| llama.cpp | `b11223`, `version: 0.5.0-dev`, commit `4da6337767f973e2b4d0797e5b323d77d8565e4a`, `built with GNU 11.4.0` |
| Backend of the build used | Vulkan (`~/.bongo/llama/b11223/vulkan`) |
| Backend of the capture | **CPU** (`llama_model_params.n_gpu_layers = 0`) — router selections are weight-determined, and CPU-only avoids the 33 GiB-expert / 32 GiB-VRAM ceiling |
| Model | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, tier IQ2_XS, arch `qwen4exp` |
| Shards | `Swift-...-IQ2_XS-00001-of-00002.gguf` (39,788,473,344 B) + `-00002-of-00002.gguf` (28,363,693,824 B) = 68,152,167,168 B; shard SHA-256 in [`../../2026-09-27-baseline/matrix.md`](../../2026-09-27-baseline/matrix.md) |
| Experts | 48 layers x 512 experts, 10 active per token per layer, 33.02 GiB of expert weights |

## Tool build

`bench/tools/route_capture.c` was compiled against the **local** bongo shared libraries, with headers fetched
at the exact commit the build reports:

```sh
ZIG=~/.local/lib/python3.14/site-packages/ziglang/zig    # pip install --user ziglang
$ZIG cc -O2 -I <headers@4da6337767> -o route_capture bench/tools/route_capture.c \
    -L~/.bongo/llama/b11223/vulkan -lllama -lggml -lggml-base
cp route_capture ~/.bongo/llama/b11223/vulkan/   # ggml_backend_load_all() resolves plugins next to the exe
```

`bench/run-expert-activation.sh` automates this (header fetch -> zig -> compile -> capture) and is the
recorded reproduction path.

## Captures

Command per corpus:

```sh
LD_LIBRARY_PATH=~/.bongo/llama/b11223/vulkan \
  ~/.bongo/llama/b11223/vulkan/route_capture \
    "$MODEL" corpora/<name>.txt raw/<name>.tsv 8192 [<decode_steps> raw/<name>_dec.tsv]
```

| file | phasing | tokens | events | wall time |
| --- | --- | ---: | ---: | ---: |
| `raw/code.tsv.gz` | prefill | 3,794 | 1,783,190 | 7:03 |
| `raw/doc.tsv.gz` | prefill | 3,971 | 1,866,380 | 7:57 (incl. decode) |
| `raw/doc_dec.tsv.gz` | decode (teacher-forced) | 256 | 122,880 | |
| `raw/chat.tsv.gz` | prefill | 536 | 251,930 | ~3:30 (incl. decode) |
| `raw/chat_dec.tsv.gz` | decode (teacher-forced) | 128 | 61,440 | |
| `raw/convo.tsv.gz` | prefill | 1,008 | 473,770 | 2:38 |

Wall time is dominated by paging the 68 GB mmapped model, not by the router. `n_ctx = 8192`,
`n_batch = n_ubatch = n_ctx` so each prefill is a single graph. Decode runs after the prefill with the tail
tokens teacher-forced (single-token graphs).

## Raw file format

One line per computed layer per graph, tab-separated:

```
TOPK  ffn_moe_topk-<il>  ne0 ne1 ne2 ne3  idx...
```

`ne0 = 10` (active experts), `ne1 = token count`. Flat order is token-major:
`experts(token, k) = idx[token*ne0 + k]`. Decode files additionally contain `STEP <i>` marker lines before
each single-token graph.

## Analysis

```sh
python3 bench/analyze-expert-activation.py \
  --raw bench/results/2026-09-28-expert-activation/raw \
  --decode doc=.../raw/doc_dec.tsv.gz,chat=.../raw/chat_dec.tsv.gz \
  --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
  --out bench/results/2026-09-28-expert-activation/analysis.json \
  --corpora doc,code,chat,convo
```

Outputs `analysis.json` (all numbers) and the summary reproduced as
[`coverage.txt`](coverage.txt). Per-layer expert bytes come from
[`../../2026-09-27-expert-placement/expert-bytes-iq2_xs.json`](../../2026-09-27-expert-placement/expert-bytes-iq2_xs.json).
