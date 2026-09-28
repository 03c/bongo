# bongo benchmark harness

Reproducible Stage 0 baseline for bongo against a running OpenAI-compatible
`llama-server` endpoint (the one `bongo.sh` starts).

## Run it

```sh
./bench/run.sh
```

That is the whole procedure. It measures the default tier (`iq2_xs`) at 1024,
4096, 32768 and 131072 context tokens and writes:

```
bench/results/<YYYY-MM-DD>-baseline/
  matrix.json      # machine-readable, one record per context + raw runs
  matrix.md        # human-readable summary
  raw/             # every individual request/response (timings, memory)
  .hashes.json     # cached shard hashes
```

Nothing needs editing between runs. Useful overrides:

| flag | meaning |
| --- | --- |
| `--base-url URL` | endpoint root, default `http://127.0.0.1:8080/v1` (`BONGO_BASE_URL`) |
| `--tier NAME` | label recorded in results, default `iq2_xs` (`BONGO_TIER`) |
| `--contexts 1024,4096,32768,131072` | context targets (`BONGO_CONTEXTS`) |
| `--repeats N` | repeats per context, default 3 (`BONGO_REPEATS`) |
| `--repeats-deep N` | repeats for contexts at/above `--deep-threshold`, default same as `--repeats` (`BONGO_REPEATS_DEEP`) |
| `--deep-threshold N` | context length at/above which `--repeats-deep` applies, default disabled (`BONGO_DEEP_THRESHOLD`) |
| `--max-tokens N` | decode length used for output tok/s, default 128 |
| `--needle-context N` | context of the recall check, default 131072 |
| `--gguf-dir DIR` | where to hash the model shards (`BONGO_GGUF_DIR`) |
| `--hash-mode full\|sampled\|none` | shard hashing, default `full` |
| `--server-pid PID` | llama-server pid for memory sampling (auto-detected) |
| `--out-dir DIR` | results directory |

Exit codes: `0` full success, `3` partial/negative result recorded (for example
128K OOM), `2` endpoint unreachable.

## What each number means

- **prompt tokens** — the server-reported token count of the prompt actually
  processed (`timings.prompt_n`). Prompts are sized with the server's
  `/tokenize` endpoint, so the target context is honest, not a character-count
  guess, and are capped so `prompt tokens + max_tokens <= n_ctx`.
- **prompt tok/s** — prefill throughput: prompt tokens per second of prompt
  processing (`timings.prompt_per_second`, cross-checked against
  `prompt_n / prompt_ms`).
- **output tok/s** — decode throughput: generated tokens per second
  (`timings.predicted_per_second`). Requests set `ignore_eos`, so every context
  decodes the full `--max-tokens` and throughput is comparable even when the
  model would otherwise stop early on a short prompt.
- **TTFT ms** — time to first token, measured client side on a streaming
  request: from sending the request to the first non-empty content chunk. The
  same streaming request also returns the server's final `timings`, so prefill
  and decode throughput are measured on that request and a second prefill is not
  paid for. Servers that do not emit stream timings fall back to a non-stream
  request for throughput.
- **prefill ms** — server-reported prompt-processing time for the non-stream
  request.
- **peak VRAM** — highest GPU memory attributable to the server process during
  the run. Method is recorded (`fdinfo` = `/proc/<pid>/fdinfo` `drm-*vram*`
  counters; `sysfs`; `xpu-smi`). On the reference box `fdinfo` is the primary
  source because `xpu-smi`/`intel_gpu_top` are not installed.
- **peak system RAM** — highest `VmRSS` summed over the `llama-server` process
  (and children that appear as `llama-server`). The process high-water mark
  `VmHWM` is recorded too.
- **needle** — a passphrase is planted at ~50% depth of a full-length prompt and
  the model must echo it. `pass` proves the context window is real and usable,
  not merely accepted. When the needle context is also a measured context, the
  check is folded into that run (one ~128K prefill both measures throughput and
  proves recall); otherwise it runs as its own request.
- **cv** — coefficient of variation (stdev / median) across repeats. Reported
  wherever `repeats >= 2`.

## Variance

`matrix.json` keeps every raw run; `matrix.md` and each `summary` block report
`n`, `median`, `mean`, `min`, `max`, `stdev` and `cv` per metric. The default is
`--repeats 3`; a metric with high `cv` (> 0.1) should be treated as noisy and
re-measured on an idle machine.

A full prefill at 128K takes minutes on the reference box, so a single command
can spend a different repeat budget at depth:

```sh
./bench/run.sh --tier iq2_xs --repeats 3 --repeats-deep 1 --deep-threshold 131072
```

This keeps 3 repeats for 1K/4K/32K and uses 1 repeat at 128K. `matrix.md` and
`matrix.json.config` record both budgets, so the variance that was actually
measured is always visible.

## Negative results

A clean negative result is a valid outcome. If a context fails (OOM, context
exceeded, server crash, timeout) the harness records the exact HTTP status and
error body, stops climbing, and reports `highest_working_context`. If the whole
endpoint is unreachable it writes a fatal `matrix.json` and exits `2`.

## Malformed / boundary cases

`matrix.json["error_cases"]` always runs these requests and records the server's
answer, so the harness proves it handles errors instead of hanging:

- empty prompt
- missing `prompt` field
- unknown model id
- negative `max_tokens`
- prompt longer than `n_ctx`
- non-JSON request body

## Self-test (no GPU or model needed)

```sh
./bench/selftest.sh
```

Starts `bench/mock_server.py` (an OpenAI-compatible stub) and checks the happy
path, the 128K-style OOM path (records the error and the highest working
context), and the unreachable-endpoint path. Use this to validate changes to the
harness without the reference machine.
