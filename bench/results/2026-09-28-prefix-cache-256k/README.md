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

## Results

Engine `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), Vulkan, iq2_xs
`Swift-1.5-Qwen3.8-Flash-Next` (`qwen4exp`: 36 Gated-DeltaNet + 12 full-attention layers),
exact argv above, run 2026-09-28 14:12Z–16:36Z. Save stage rc=0, restore stage rc=0.

### Save (stage 1)

| quantity | value |
| --- | ---: |
| prefill prompt tokens | 261,997 |
| prefill time | 2,348,797 ms |
| prefill rate | 111.5 tok/s |
| `save` wall / server `save_ms` | 1,560.8 / 1,558.6 ms |
| `save` bytes (`n_written`) | **3,980,332,632** (3.71 GiB) |
| `save` tokens (`n_saved`) | 262,028 |
| `erase` wall / tokens (`n_erased`) | 45.7 ms / 262,028 |

The slot file on disk is 3,980,332,632 B, byte-identical to `n_written`. That is ~15.2 kB
per token, and **writing a full-size KV is cheap: about 1.6 s.** The save is not flat in
size, though — at 2.55 GB/s it is 1.8x slower per byte than the 31K save (4.58 GB/s), so
the write rate does degrade with KV size, just gently.

Note the two points do not agree on bytes per token: 15,190 B/token at 256K against
18,459 B/token at 31K (0.82x). KV size is not simply proportional to token count across
these two runs. The runs differ in `--n-cpu-moe` (18 vs 16) and context size, and this is
one sample each, so the cause is not established here; it is worth a follow-up if exact KV
sizing matters for capacity planning.

The prefill rate is above the 84.8 tok/s of
[`../2026-09-28-ctx256-full/`](../2026-09-28-ctx256-full/): that run overlapped other GPU
work, this one had the box to itself.

### Restore (stage 2, after a real server restart)

| quantity | 31K | **256K** |
| --- | ---: | ---: |
| bytes | 585,931,732 | **3,980,332,632** |
| server `restore_ms` | 264.4 | **11,521.1** |
| wall `restore_elapsed_ms` | 264.9 | **11,521.6** |
| effective rate | 2.22 GB/s | **0.35 GB/s** |
| tokens restored | 31,743 | **262,028** |

**Restore degrades far worse with size than save does.** 6.79x the bytes costs **43.6x
the time**, so restore is 6.4x worse than linear and its effective read rate falls from
2.22 GB/s to 0.35 GB/s — 6.4x slower per byte, against save's 1.8x. The round trip
(save + restore) is 13.1 s for a 3.71 GiB KV, against 0.39 s at 31K: 6.8x the bytes for
33x the time. This matters for any design that restores on model swap, where a 256K swap
pays 11.5 s of restore before it knows whether the restore helped.

### Reuse: the restored KV is **not** reused

`restore_verified: false`. The request after the restore re-prefilled the whole prompt:

| quantity | value |
| --- | ---: |
| post-restore `prompt_n` | 261,997 |
| post-restore `cache_n` | **0** |
| post-restore `cached_tokens` | **0** |
| post-restore prefill | 2,351,235 ms (111.4 tok/s) |
| post-restore TTFT | 2,351,427.7 ms |

Confirmed independently mid-flight: at 16:23Z, 188,416 tokens into that request,
`/slots` reported `n_prompt_tokens_cache: 0` — the engine was re-processing from token 0
and never consulted the 262,028 tokens it had just read off disk.

**This is the same verdict as the 31K point, and at 256K it is much worse in absolute
terms.** Restoring 3.71 GiB costs 11.5 s and then buys nothing: the next request spends
another 2,351 s re-prefilling. The restore is a 204x net loss in time. Combined with the
31K result, the conclusion is size-independent — on this hybrid `qwen4exp` model a restored
slot is inert, and persisting a large KV is pure cost.

Cause is unchanged from 31K (`docs/bongo-sh.md` § Slot KV persistence): `action=save`
writes the slot's tokens plus its sequence state (KV + recurrent state) but the server does
not persist its context checkpoints, and the engine needs a checkpoint to resume a prefix
on hybrid/recurrent memory. In-session prefix reuse is unaffected — only cross-restart
restore is.

## How the restore was measured after an interrupted first run

The 14:12Z run completed the save stage and was then `SIGTERM`ed at 15:15:21Z during the
restore stage's verify turn, discarding the restore timing. The 3.98 GB slot file survived
on disk, so the restore was resumed on its own with
`./bench/run-slot-restore-256k.sh --restore-only` at script revision `b9992e1`
(`queued-script-rev.txt`) rather than paying for a second 39-minute prefill. The restore
still came off disk into a server process that never saw the original prefill, which is
the property being measured, so the number is unchanged in meaning. It took the lock after
100 s of queueing behind the BAS-79 PLE A/B, and `slot-restore-256k.json` records
`restore_only_resume: true`.

The runner's own `runner.log` (from the first attempt) records `rc=143`; it is kept for the
record but is not an authoritative artefact.

## Bugs found in the measurement code while doing this

All were in `bench/measure-slot-restore.py` and all made the records wrong rather than
merely incomplete. They are fixed; the first two are why the committed raw JSON carries
`slot_file_bytes_observed` and response-body occupancy fields instead of the original keys.

1. **`slot_file_bytes` was always `null`.** The script built the slot path as
   `<filename>.bin`, but `action=save` writes the filename **verbatim** — `ctx256-slot`
   lands as `ctx256-slot`, not `ctx256-slot.bin` — so the `stat` hit a path that never
   existed. Now probes the verbatim name first and `.bin` as a fallback, recording which
   matched.
2. **Every `n_past_*` occupancy field was `null`.** The `b11223` build dropped `n_past` from
   `/slots`.
3. **…and my first fix for (2) was wrong, and worse than the null.** I mapped
   `n_prompt_tokens_cache` to "tokens resident in the slot". Those fields describe the
   slot's **last request**, not the KV it holds: on a server that had just restored 262,028
   tokens and served nothing they read 0, and `n_prompt_tokens_cache` is the reused-token
   count *of that request* — it read 0 at 188K tokens into a re-prefill. Presenting that as
   occupancy would have turned a real measurement into a misleading one. Occupancy now
   comes from the `action=save`/`restore`/`erase` response bodies (`n_saved`, `n_restored`,
   `n_erased`) and the next request's `cache_n`; `/slots` is recorded under
   `slot_progress_*`, named for what it is.
4. **Server timings were read from the wrong place.** `save_ms`/`restore_ms` are nested
   under `timings` in the response body, so a flat lookup missed them. Now read via
   `timings.<key>`.
5. **The record is now checkpointed after each expensive step**, so a signal during the
   ~40-minute verify turn can no longer discard the restore timing. That was precisely the
   failure that cost the first run, and it is why the interrupted run's `measure-restore.log`
   is empty while this one's restore numbers are on disk from 15:57Z.

`bench/mock_server.py` gained a `GET /slots` route reporting the b11223 field shape, and
now emits `n_written`/`n_read`/`timings` on save/restore, so all of the above is covered
end-to-end **without the GPU**.

## Caveats

- Single box, single pass; no variance repeats. Both the restore nonlinearity (6.4x worse
  than linear) and the bytes-per-token discrepancy between the 31K and 256K points are
  single samples and are not characterised.
- `--skip-delta` was set: the 131K delta turn exercises *in-session* prefix reuse over a
  trimmed cache, not restore, and its two cold prefills cost about an hour.
- The 256K save and restore halves come from two different invocations (`save` at 14:12Z,
  `--restore-only` at 15:55Z) because of the interruption. Same server flags, same model,
  same slot file; the restart between them is real and is the thing being measured.

