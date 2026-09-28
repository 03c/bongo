# 2026-09-28 — prefix-cache (agentic turn) measurement

Raw result of `bench/measure-prefix-cache.py`, run for the plan-review comment on
[BAS-62](/BAS/issues/BAS-62). See [`docs/research/agentic-prefix-cache.md`](../../../docs/research/agentic-prefix-cache.md)
for the analysis.

## How it was run

```sh
# server (Vulkan, shipped placement, 32K context so the 31K point fits)
~/.bongo/llama/b11223/vulkan/llama-server \
  --model ~/.bongo/models/.../iq2_xs/Swift-...-00001-of-00002.gguf \
  --ctx-size 32768 --jinja --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-gpu-layers 99 --n-cpu-moe 16 \
  --host 127.0.0.1 --port 8080 --parallel 1 --alias bongo-iq2_xs \
  --device Vulkan1

# measurement
python3 bench/measure-prefix-cache.py --prefixes 4096,16384,24576 --delta 512
python3 bench/measure-prefix-cache.py --prefixes 31744 --delta 512 --out bench/results/2026-09-28-prefix-cache-31k
```

- Engine: llama.cpp `b11223` (`4da633776`), Vulkan (`--device Vulkan1`).
- The harness normally sends `cache_prompt: false`; these runs override it to `true` for
  `hit`/`grow` cases and keep `false` for `cold` cases.
- `--parallel 1` means one slot, so a later request on the same prefix reuses slot 0's KV.

## Files

- `prefix-cache.json` — prefixes 4K/16K/24K (12 runs incl. `cold`, `hit`, `grow`, `grow-repeat`).
- `../2026-09-28-prefix-cache-31k/prefix-cache.json` — the 31K point.

Each run records `status`, `prompt_n`, `prompt_ms`, `prompt_tps`, `ttft_ms`, `output_tps`, `wall_ms`.
`prompt_n` near 4 on a `hit`/`grow-repeat` row is the proof that the server reused the cached KV.

## Caveats

- Single box, single run each, Vulkan only; not repeated for variance.
- `hit` and `grow` share the slot sequentially, so the `grow` cost includes attention over the cached
  prefix, which is the intended thing to measure.
- 156K/256K are extrapolated in the doc; the cold re-prefill of a 156K prompt was not run (~20 min).
