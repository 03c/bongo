# 2026-09-28 — prefix-cache M3.0a, 256K q8 slot save/restore

The 256K point of the slot-KV-persistence measurement, following the 31K point in
[`../2026-09-28-prefix-cache-m3.0a/`](../2026-09-28-prefix-cache-m3.0a/). This closes the
last open M3.0a method bullet: size the persistence cost for a full-size (~4 GiB) KV and
check whether the restored KV is reused, which it is not at 31K.

Run with [`bench/run-slot-restore-256k.sh`](../../../bench/run-slot-restore-256k.sh), which
takes the shared single-GPU flock (BAS-80), starts the pinned server through `bongo.sh`,
and refuses to start if any other `llama-server` is alive.

## How it was run

- Engine: llama.cpp **`b11223`** (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), **Vulkan**
  (`--device Vulkan1`).
- Model: IQ2_XS `Swift-1.5-Qwen3.8-Flash-Next` (`qwen4exp`, 36 Gated-DeltaNet + 12
  full-attention layers), alias `bongo-iq2_xs`.
- Exact argv:

  ```
  --model .../Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  --ctx-size 262144 --jinja --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0
  --n-gpu-layers 99 --n-cpu-moe 18
  --host 127.0.0.1 --port 8080 --parallel 1
  --alias bongo-iq2_xs --metrics --cache-prompt
  --slot-save-path /home/cchild/.bongo/slot-restore-256k/run/slots --device Vulkan1
  ```

- `--n-cpu-moe 18` is the safe 256K placement from
  [`../2026-09-28-ctx256-fit/`](../2026-09-28-ctx256-fit/); `n16` sat within 0.03 GiB of the
  observed device-loss threshold. Measured `drm-resident-vram0` was 31,952,000 KiB
  (30.47 GiB), ~1.4 GiB under the threshold — the fit prediction held.

## Save stage (completed 14:52Z)

Cold prefill of a 261,997-token needle prompt, then `save` and `erase`.

| quantity | value |
| --- | ---: |
| prefill prompt tokens | 261,997 |
| prefill time | 2,348,797 ms |
| prefill rate | 111.5 tok/s |
| TTFT | 2,349,033 ms |
| `save` wall time | 1,560.8 ms (server `save_ms` 1,558.6) |
| `save` bytes (`n_written`) | **3,980,332,632** (3.71 GiB) |
| `save` tokens (`n_saved`) | 262,028 |
| `erase` wall time | 45.7 ms |
| `erase` tokens (`n_erased`) | 262,028 |
| slot file on disk | 3,980,332,632 B, byte-identical to `n_written` |

Cost per token is ~15.2 kB, and the save rate is ~2.55 GB/s. Both are the right order of
magnitude for a 262K q8 hybrid KV, and the write is fast enough that persistence is not
itself a bottleneck — the reuse gap below is the real problem.

The prefill rate is higher than the 84.8 tok/s of
[`../2026-09-28-ctx256-full/`](../2026-09-28-ctx256-full/): that run overlapped other GPU
work, this one had the box to itself.

## Restore stage

**Status: queued.** The first attempt was interrupted by a `SIGTERM` at 15:15:21Z, after
the save stage had completed and while the restore stage was in its post-restore verify
turn. The 3.98 GB slot file survived on disk, so the restore was resumed on its own with
`./bench/run-slot-restore-256k.sh --restore-only` rather than paying for a second 39-minute
prefill. The restore still comes off disk into a server process that never saw the
original prefill, which is the property being measured.

The queued run is `bash ./bench/run-slot-restore-256k.sh --restore-only` at script revision
`b9992e1` (recorded in `queued-script-rev.txt`), waiting on the single-GPU flock behind the
BAS-79 PLE A/B. It starts on its own when the lock frees; `--plan` confirms the file it
will resume from and its size. Expect ~1 min of server start, then the restore, then the
verify turn: ~40 min if the restored KV is not reused, seconds if it is.

Result: see `slot-restore-256k.json` (`restore` and `reuse` blocks) and
`slot-restore-restore.json` once the run completes.

## Bugs found and fixed in the measurement code

Both were in `bench/measure-slot-restore.py` and both made the records look *worse* than
the system is, which is how they were caught:

1. **`slot_file_bytes` was always `null`.** The script built the slot path as
   `<filename>.bin`, but `action=save` writes the filename **verbatim** — `ctx256-slot`
   lands as `ctx256-slot`, not `ctx256-slot.bin`. The stat therefore hit a path that never
   existed. Fixed by probing the verbatim name first and `.bin` as a fallback, and by
   recording which one matched. The 256K byte count in this README is the on-disk `stat`,
   which matches the server's own `n_written` exactly.
2. **Every `n_past_*` occupancy field was `null`.** The `b11223` Vulkan build reports
   `n_prompt_tokens` / `n_prompt_tokens_processed` / `n_prompt_tokens_cache` on `/slots` and
   no longer reports `n_past`, so the reads silently returned `None`. Fixed with a
   `slot_filled_tokens()` helper that tries the current names and falls back to `n_past`,
   used at all four call sites.

`bench/mock_server.py` gained a `GET /slots` route reporting the b11223 field shape (and a
correct `n_erased`), so both fixes are covered end-to-end without needing the GPU.

The measurement script now also **checkpoints its record after each expensive step**, so a
signal during the ~40-minute verify turn can no longer discard the restore timing — which
is the number the task exists to measure.

## Caveats

- Single box, single pass; no variance repeats.
- `--skip-delta` is set at 256K: the 131K delta turn exercises *in-session* prefix reuse
  over a trimmed cache, not restore, and its two cold prefills cost about an hour.
- The restore-stage `runner.log` records `rc=143` for the interrupted attempt. That log is
  kept for the record; the authoritative artefacts are the `slot-restore-*.json` files.
