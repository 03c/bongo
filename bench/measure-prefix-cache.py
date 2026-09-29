#!/usr/bin/env python3
"""Measure llama-server prefix-cache (prompt reuse) behaviour for agentic turns.

The Stage 0 harness always sent ``cache_prompt: false``, so every published
number was a *cold* prefill.  An agentic coding session is the opposite: a small
first prompt that grows, where each turn re-sends the whole history and only the
new suffix is new.  This script measures that cached path directly.

Sequence per prefix size P:
  cold   : P,          cache_prompt=false  -> full prefill cost
  hit    : P,          cache_prompt=true   -> full KV reuse
  grow   : P + D,      cache_prompt=true   -> prefill only the D-token delta
  grow2  : P + D,      cache_prompt=true   -> repeat, should be a full hit

Then, when the server was started with ``--slot-save-path``, it times the slot
KV persistence path:
  prime  : P,          cache_prompt=true   -> the slot holds P
  save   : POST /slots/{id}?action=save    -> KV written to disk
  erase  : POST /slots/{id}?action=erase   -> slot emptied
  restore: POST /slots/{id}?action=restore -> KV read back
  after  : P,          cache_prompt=true   -> must be a full hit if restore worked

Runs against a live OpenAI-compatible llama-server (default 127.0.0.1:8080).
Stdlib only; imports the shared bench_lib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import (  # noqa: E402
    CORPUS,
    MemorySampler,
    Tokenizer,
    get_json,
    now_iso,
    post_json,
    read_text,
    server_root,
)
from harness import build_needle_document, streaming_measure  # noqa: E402

BASE = os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BONGO_MODEL", "bongo-iq2_xs")


def server_record(pid):
    """Engine revision, props and the exact server argv, so a number is reproducible."""
    root = server_root(BASE)
    props_res = get_json(root, "/props", 15)
    rec = {
        "base_url": BASE,
        "server_root": root,
        "props": props_res.json if isinstance(props_res.json, dict) else None,
        "props_status": props_res.status,
        "pid": pid,
        "flags": None,
        "cmdline": None,
    }
    if pid:
        raw = read_text(f"/proc/{pid}/cmdline")
        if raw:
            rec["flags"] = raw.split("\x00")
        rec["cmdline"] = raw
    return rec


def make_delta(tk, tokens, tag):
    """A distinct suffix blob, so it is not a prefix of any prior prompt."""
    seed = f" | turn {tag} "
    return tk.size_to(tokens, corpus="The reviewer noted the following delta. " + CORPUS, seed=seed)


def run_case(tk, label, prompt, cache_prompt, max_tokens=4, timeout=900):
    rec = streaming_measure(
        BASE,
        MODEL,
        prompt,
        max_tokens,
        timeout,
        cache_prompt=bool(cache_prompt),
    )
    rec["label"] = label
    rec["cache_prompt"] = bool(cache_prompt)
    rec["ts"] = time.time()
    print(
        f"{label:26s} status={rec.get('status')} "
        f"prompt_n={rec.get('prompt_tokens')} cache_n={rec.get('cache_n')} "
        f"prompt_ms={rec.get('prompt_ms')} prompt_tps={rec.get('prompt_tps')} "
        f"ttft_ms={rec.get('ttft_ms')} out_tps={rec.get('output_tps')} "
        f"wall_ms={rec.get('wall_ms')}",
        flush=True,
    )
    return rec


def measured_run(sampler, tk, label, prompt, cache_prompt, timeout, max_tokens=4):
    """``run_case`` plus the VRAM/RAM peak over that case's own window.

    The sampler is optional so the script still runs against the mock server
    (selftest), where there is no ``llama-server`` process to attribute bytes to.
    """
    t0 = time.time()
    rec = run_case(tk, label, prompt, cache_prompt, max_tokens=max_tokens, timeout=timeout)
    if sampler is not None:
        rec["memory"] = sampler.window_peak(t0, time.time())
    return rec


def slot_action(slot_id, action, filename, timeout):
    """POST /slots/{id}?action=save|restore|erase on the server root (not /v1)."""
    root = server_root(BASE)
    res = post_json(root, f"/slots/{slot_id}?action={action}", {"filename": filename}, timeout)
    body = res.json if isinstance(res.json, dict) else None
    rec = {
        "action": action,
        "slot_id": slot_id,
        "filename": filename,
        "status": res.status,
        "wall_ms": round(res.elapsed_ms, 2),
        "response": body,
    }
    if res.status != 200:
        rec["error"] = res.body.decode("utf-8", "replace")[:1000]
    return rec


def measure_slot(tk, slot_id, prefix_text, prefix_tokens, save_dir, timeout):
    """Time save -> erase -> restore and prove the restored KV is reused.

    After restore the next request must:
      - reuse the prefix (cache_n >= prefix_tokens * 0.99)
      - hit TTFT parity with a full cache hit (< 1 s at the prefix size)
    """
    filename = f"bongo-prefix-cache-slot{slot_id}.bin"
    ckpt_path = filename + ".ckpt"
    prime = run_case(tk, f"slot{slot_id}_prime_p{prefix_tokens}", prefix_text, True, max_tokens=1, timeout=timeout)
    save = slot_action(slot_id, "save", filename, timeout)
    if save_dir:
        path = os.path.join(save_dir, filename)
        save["file"] = path
        save["file_bytes"] = os.path.getsize(path) if os.path.isfile(path) else None
        ckpt_full = os.path.join(save_dir, ckpt_path)
        save["ckpt_file_bytes"] = os.path.getsize(ckpt_full) if os.path.isfile(ckpt_full) else None
    erase = slot_action(slot_id, "erase", filename, timeout)
    restore = slot_action(slot_id, "restore", filename, timeout)
    after = run_case(
        tk, f"slot{slot_id}_after_restore_p{prefix_tokens}", prefix_text, True, max_tokens=1, timeout=timeout
    )
    after_cache_n = after.get("cache_n")
    restored = bool(after_cache_n is not None and prefix_tokens and after_cache_n >= prefix_tokens * 0.99)
    after_ttft = after.get("ttft_ms")
    ttft_ok = after_ttft is not None and after_ttft < 1000
    return {
        "slot_id": slot_id,
        "filename": filename,
        "prefix_tokens": prefix_tokens,
        "prime": prime,
        "save": save,
        "erase": erase,
        "restore": restore,
        "after_restore": after,
        "restore_verified": restored and ttft_ok,
        "restore_reuse": restored,
        "restore_ttft_ms": after_ttft,
        "restore_ttft_ok": ttft_ok,
    }


def check_needle(tk, prefix_text, prefix_tokens, max_tokens=32, timeout=900):
    """Correctness: run the needle over the restored slot and confirm the needle text survives.

    ``harness.build_needle_document`` plants a unique sentinel (bench_lib.NEEDLE) at ~50%
    depth and already appends the recall question, so it is the fixed prompt as-is. (Do not
    append a second question or leak the answer into the prompt: that makes the model skip
    the code and continue the document, a false failure.) This proves the restored KV (and
    the restored checkpoints that re-anchor recurrent state) still drive correct generation,
    not just cache reuse.
    """
    from bench_lib import NEEDLE  # local import keeps the CLI import surface unchanged
    needle_text = build_needle_document(prefix_text)
    rec = streaming_measure(BASE, MODEL, needle_text, max_tokens, timeout, cache_prompt=True)
    answer = (rec.get("text") or "")
    ok = NEEDLE.lower() in answer.lower()
    if ok:
        print(f"  needle pass (p{prefix_tokens}): sentinel found in answer", flush=True)
    else:
        print(f"  needle FAIL (p{prefix_tokens}): sentinel missing; answer[:120]={answer[:120]!r}", flush=True)
    return {
        "prompt_tokens": rec.get("prompt_tokens"),
        "cache_n": rec.get("cache_n"),
        "ttft_ms": rec.get("ttft_ms"),
        "answer": answer,
        "needle_present": ok,
        "status": rec.get("status"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="4096,16384,24576")
    ap.add_argument("--delta", type=int, default=512)
    ap.add_argument("--out", default="bench/results/2026-09-28-prefix-cache")
    ap.add_argument("--slot-id", type=int, default=0)
    ap.add_argument(
        "--slot-save-dir",
        default=os.environ.get("BONGO_SLOT_SAVE_DIR", ""),
        help="directory the server was started with --slot-save-path (for file size)",
    )
    ap.add_argument("--no-slot", dest="measure_slot", action="store_false", default=True)
    ap.add_argument(
        "--server-pid",
        type=int,
        default=int(os.environ.get("BONGO_SERVER_PID", "0")) or None,
        help="llama-server pid (records the exact argv)",
    )
    ap.add_argument(
        "--needle",
        action="store_true",
        help="Run the needle correctness check at the largest prefix size after the slot restore cycle.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("BONGO_CASE_TIMEOUT", "7200")),
        help="Per-request timeout in seconds. A 128K/256K cold prefill runs for "
        "25-50 min, so the 900 s run_case default is not enough at long context.",
    )
    ap.add_argument(
        "--cached-only",
        action="store_true",
        help="Skip the explicit cold (cache_prompt=false) case and prime the "
        "prefix with cache_prompt=true instead. The prime request pays the same "
        "full prefill, so this changes the reported labels, not the runtime.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tk = Tokenizer(BASE, timeout=args.timeout)
    if not tk.available:
        tk.count("warmup")

    # Attribute VRAM to the server serving this run. ``--server-pid`` is set by
    # run-prefix-cache.sh; without it the sampler discovers llama-server by argv.
    sampler = MemorySampler(server_pids=[args.server_pid] if args.server_pid else None)
    sampler.start()
    run_start = time.time()

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    results = {
        "schema": "bongo.prefix-cache.v1",
        "generated_at": now_iso(),
        "base_url": BASE,
        "model": MODEL,
        "delta_tokens": args.delta,
        "slot_id": args.slot_id,
        "slot_save_dir": args.slot_save_dir or None,
        "server": server_record(args.server_pid),
        "runs": [],
        "slot": None,
    }

    # One warmup so the first measured request is not the first touch. It uses
    # cache_prompt=false so it cannot seed the slot for a later hit.
    measured_run(sampler, tk, "warmup", tk.size_to(256), False, args.timeout, max_tokens=1)

    for p in prefixes:
        p_text = tk.size_to(p)
        real_p = tk.count(p_text)
        d_text = make_delta(tk, args.delta, p)
        grow_text = p_text + d_text
        real_grow = tk.count(grow_text)
        print(f"\n== prefix target {p} (actual {real_p}); with delta actual {real_grow} ==", flush=True)

        if not args.cached_only:
            results["runs"].append(
                {"prefix_target": p, **measured_run(sampler, tk, f"cold_p{p}", p_text, False, args.timeout)}
            )
        else:
            # The product path never pays an explicit cold pass: the first
            # request primes the slot with cache_prompt=true and is the prefill.
            results["runs"].append(
                {"prefix_target": p, **measured_run(sampler, tk, f"prime_p{p}", p_text, True, args.timeout)}
            )
        results["runs"].append(
            {"prefix_target": p, **measured_run(sampler, tk, f"hit_p{p}", p_text, True, args.timeout)}
        )
        results["runs"].append(
            {
                "prefix_target": p,
                **measured_run(sampler, tk, f"grow_p{p}_d{args.delta}", grow_text, True, args.timeout),
            }
        )
        results["runs"].append(
            {"prefix_target": p, **measured_run(sampler, tk, f"grow_p{p}_repeat", grow_text, True, args.timeout)}
        )

    flags = []
    if args.server_pid:
        raw = read_text(f"/proc/{args.server_pid}/cmdline")
        if raw:
            flags = raw.split("\x00")
    results["server"] = server_record(args.server_pid)
    results["slot_save_checkpoints"] = "--save-slot-checkpoints" in (flags or [])

    # Slot persistence: the cross-idle/cross-restart feature. Uses the last
    # prefix size so it times the largest KV measured in this run.
    if args.measure_slot and prefixes:
        p = prefixes[-1]
        p_text = tk.size_to(p)
        real_p = tk.count(p_text)
        print(f"\n== slot save/restore at prefix {p} (actual {real_p}) ==", flush=True)
        results["slot"] = measure_slot(tk, args.slot_id, p_text, real_p, args.slot_save_dir, timeout=args.timeout)
        # correctness: the needle must still be retrievable after the restore cycle
        if args.needle and args.measure_slot:
            print(f"\n== needle correctness after restore at prefix {p} (actual {real_p}) ==", flush=True)
            results["needle_after_restore"] = check_needle(
                tk, p_text, real_p, max_tokens=32, timeout=args.timeout
            )

    sampler.stop()
    results["cached_only"] = bool(args.cached_only)
    results["request_timeout_s"] = args.timeout
    results["memory"] = sampler.window_peak(run_start, time.time())
    results["memory"]["vram_method"] = sampler.vram_method
    case_peaks = [
        (r.get("memory") or {}).get("vram_peak_bytes")
        for r in results["runs"]
        if (r.get("memory") or {}).get("vram_peak_bytes")
    ]
    if case_peaks:
        results["memory"]["vram_peak_bytes"] = max(case_peaks)

    suffix = "json" if args.out.endswith(".json") else ""
    path = args.out if suffix else os.path.join(args.out, "prefix-cache.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
