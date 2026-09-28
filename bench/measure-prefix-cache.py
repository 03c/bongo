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

from bench_lib import Tokenizer, CORPUS, now_iso, post_json, server_root  # noqa: E402
from harness import streaming_measure  # noqa: E402

BASE = os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BONGO_MODEL", "bongo-iq2_xs")


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
    """Time save -> erase -> restore and prove the restored KV is reused."""
    filename = f"bongo-prefix-cache-slot{slot_id}.bin"
    prime = run_case(tk, f"slot{slot_id}_prime_p{prefix_tokens}", prefix_text, True, max_tokens=1, timeout=timeout)
    save = slot_action(slot_id, "save", filename, timeout)
    if save_dir:
        path = os.path.join(save_dir, filename)
        save["file"] = path
        save["file_bytes"] = os.path.getsize(path) if os.path.isfile(path) else None
    erase = slot_action(slot_id, "erase", filename, timeout)
    restore = slot_action(slot_id, "restore", filename, timeout)
    after = run_case(
        tk, f"slot{slot_id}_after_restore_p{prefix_tokens}", prefix_text, True, max_tokens=1, timeout=timeout
    )
    after_cache_n = after.get("cache_n")
    restored = bool(after_cache_n is not None and prefix_tokens and after_cache_n >= prefix_tokens * 0.99)
    return {
        "slot_id": slot_id,
        "filename": filename,
        "prefix_tokens": prefix_tokens,
        "prime": prime,
        "save": save,
        "erase": erase,
        "restore": restore,
        "after_restore": after,
        "restore_verified": restored,
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
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tk = Tokenizer(BASE, timeout=900)
    if not tk.available:
        tk.count("warmup")

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    results = {
        "schema": "bongo.prefix-cache.v1",
        "generated_at": now_iso(),
        "base_url": BASE,
        "model": MODEL,
        "delta_tokens": args.delta,
        "slot_id": args.slot_id,
        "slot_save_dir": args.slot_save_dir or None,
        "runs": [],
        "slot": None,
    }

    # One warmup so the first measured request is not the first touch.
    run_case(tk, "warmup", tk.size_to(256), False, max_tokens=1, timeout=300)

    for p in prefixes:
        p_text = tk.size_to(p)
        real_p = tk.count(p_text)
        d_text = make_delta(tk, args.delta, p)
        grow_text = p_text + d_text
        real_grow = tk.count(grow_text)
        print(f"\n== prefix target {p} (actual {real_p}); with delta actual {real_grow} ==", flush=True)

        results["runs"].append({"prefix_target": p, **run_case(tk, f"cold_p{p}", p_text, False)})
        results["runs"].append({"prefix_target": p, **run_case(tk, f"hit_p{p}", p_text, True)})
        results["runs"].append({"prefix_target": p, **run_case(tk, f"grow_p{p}_d{args.delta}", grow_text, True)})
        results["runs"].append({"prefix_target": p, **run_case(tk, f"grow_p{p}_repeat", grow_text, True)})

    # Slot persistence: the cross-idle/cross-restart feature. Uses the last
    # prefix size so it times the largest KV measured in this run.
    if args.measure_slot and prefixes:
        p = prefixes[-1]
        p_text = tk.size_to(p)
        real_p = tk.count(p_text)
        print(f"\n== slot save/restore at prefix {p} (actual {real_p}) ==", flush=True)
        results["slot"] = measure_slot(tk, args.slot_id, p_text, real_p, args.slot_save_dir, timeout=900)

    suffix = "json" if args.out.endswith(".json") else ""
    path = args.out if suffix else os.path.join(args.out, "prefix-cache.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
