#!/usr/bin/env python3
"""Prove llama-server slot save/restore at 256K on the reference box.

A 262144-token q8 KV cannot be re-prefilled in interactive time (~30 min), so a
lost in-memory cache must be recoverable from disk.  This script measures the
save, erase (simulated cache loss), restore and the post-restore cache-hit turn.

Run it in two stages so a real server restart can sit between them:

    # stage 1: save the populated slot, then erase it
    python3 bench/measure-slot-restore.py --stage save --out bench/results/<dir>

    # (restart llama-server here)

    # stage 2: restore from disk and prove the cache-hit turn
    python3 bench/measure-slot-restore.py --stage restore --out bench/results/<dir>

Both stages reconstruct the *same* 256K needle prompt the harness sent, so the
post-restore request can hit the restored KV prefix.  Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import CORPUS, Tokenizer, get_json, now_iso, server_root  # noqa: E402
from harness import build_needle_document, streaming_measure  # noqa: E402

BASE = os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BONGO_MODEL", "bongo-iq2_xs")


def slots(base, timeout):
    res = get_json(server_root(base), "/slots", timeout)
    return res.status, res.json


def build_prompt(tk, tokens):
    hard_max = tokens
    filler = tk.size_to(tokens, CORPUS, hard_max=hard_max)
    return build_needle_document(filler)


def _json_int(body, key):
    """Read one integer out of a llama-server /slots response body ("" if absent)."""
    try:
        return json.loads(body).get(key)
    except (TypeError, ValueError):
        return None


def run_delta_turn(tk, base, args):
    """One agentic delta turn over the restored slot: a 131K-prefix hit, then the
    same prefix plus a 512-token delta.

    This measures *in-session* prefix reuse over a trimmed cache, not restore
    itself, and at 256K its two cold prefills cost about an hour, so
    `--skip-delta` drops it when only the save/restore numbers matter.
    """
    prefix_text = tk.size_to(args.delta_prefix, CORPUS, hard_max=args.delta_prefix)
    delta_text = tk.size_to(
        args.delta,
        corpus="The reviewer noted the following delta. " + CORPUS,
        seed=" | turn 1 ",
    )
    grow_text = prefix_text + delta_text
    turn = {
        "prefix_tokens_requested": args.delta_prefix,
        "delta_tokens_requested": args.delta,
        "prompt_tokens": tk.count(grow_text),
    }
    extra = {"cache_prompt": True, "ignore_eos": True}

    hit = streaming_measure(base, args.model, prefix_text, args.max_tokens, args.timeout, extra=extra)
    turn["prefix_hit"] = {
        "prompt_tokens": hit.get("prompt_tokens"),
        "cached_tokens": (hit.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens"),
        "ttft_ms": hit.get("ttft_ms"),
        "status": hit.get("status"),
    }

    grow = streaming_measure(base, args.model, grow_text, args.max_tokens, args.timeout, extra=extra)
    turn["grow"] = {
        "prompt_tokens": grow.get("prompt_tokens"),
        "prompt_ms": grow.get("prompt_ms"),
        "prompt_tps": grow.get("prompt_tps"),
        "cached_tokens": (grow.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens"),
        "ttft_ms": grow.get("ttft_ms"),
        "output_ms": grow.get("output_ms"),
        "output_tps": grow.get("output_tps"),
        "wall_ms": grow.get("wall_ms"),
        "status": grow.get("status"),
    }
    return turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["save", "restore"], required=True)
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--prefill", dest="prefill", action="store_true", default=True)
    ap.add_argument("--no-prefill", dest="prefill", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--filename", default="ctx256-slot")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--delta-prefix", type=int, default=131072)
    ap.add_argument("--delta", type=int, default=512)
    ap.add_argument(
        "--skip-delta",
        action="store_true",
        help="Skip the 128K-class delta turn. It exercises in-session prefix reuse over the "
             "restored slot, not restore itself, and at 256K its two cold prefills cost ~1 h.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = Path(args.out) / f"slot-restore-{args.stage}.json"
    base = args.base_url.rstrip("/")
    tk = Tokenizer(base, timeout=args.timeout)

    # Reserve exactly what the harness reserves (max_tokens + 128 needle room).
    reserve = max(args.max_tokens, 64) + 128
    target = args.ctx - reserve
    prompt = build_prompt(tk, target)
    prompt_tokens = tk.count(prompt)

    record = {
        "schema": "bongo.slot-restore.v1",
        "generated_at": now_iso(),
        "stage": args.stage,
        "base_url": base,
        "model": args.model,
        "ctx": args.ctx,
        "planned_prompt_tokens": prompt_tokens,
        "filename": args.filename,
    }

    status, before = slots(base, args.timeout)
    record["slots_http_status"] = status
    record["slot_n_past_before"] = (before or [{}])[args.slot].get("n_past") if isinstance(before, list) else None

    save_path = Path(os.environ.get("BONGO_SLOT_SAVE_PATH", ".")) / Path(args.filename).with_suffix(".bin")

    if args.stage == "save":
        # Populate the slot with the full 256K needle prompt first, so the save
        # captures a real 262144-token KV (not an empty slot).
        if args.prefill:
            pre = streaming_measure(
                base,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout,
                extra={"cache_prompt": False, "ignore_eos": True},
            )
            record["prefill"] = pre
            record["prefill_prompt_tokens"] = pre.get("prompt_tokens")
            record["prefill_ms"] = pre.get("prompt_ms")
            record["prefill_tps"] = pre.get("prompt_tps")
            record["prefill_ttft_ms"] = pre.get("ttft_ms")
        status, populated = slots(base, args.timeout)
        record["slot_n_past_after_prefill"] = (
            (populated or [{}])[args.slot].get("n_past") if isinstance(populated, list) else None
        )

        t0 = time.time()
        res = None
        try:
            import urllib.request

            body = json.dumps({"filename": args.filename}).encode()
            req = urllib.request.Request(
                f"{server_root(base)}/slots/{args.slot}?action=save",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                res = {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
        except Exception as exc:  # noqa: BLE001
            res = {"status": None, "error": repr(exc)}
        record["save"] = res
        record["save_elapsed_ms"] = (time.time() - t0) * 1000.0
        record["slot_file"] = str(save_path)
        record["slot_file_bytes"] = save_path.stat().st_size if save_path.exists() else None

        t0 = time.time()
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{server_root(base)}/slots/{args.slot}?action=erase", data=b"{}", method="POST"
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                record["erase"] = {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
        except Exception as exc:  # noqa: BLE001
            record["erase"] = {"status": None, "error": repr(exc)}
        record["erase_elapsed_ms"] = (time.time() - t0) * 1000.0
        status, after = slots(base, args.timeout)
        record["slot_n_past_after_erase"] = (
            (after or [{}])[args.slot].get("n_past") if isinstance(after, list) else None
        )
    else:
        record["slot_file_bytes_before_restore"] = save_path.stat().st_size if save_path.exists() else None
        t0 = time.time()
        try:
            import urllib.request

            body = json.dumps({"filename": args.filename}).encode()
            req = urllib.request.Request(
                f"{server_root(base)}/slots/{args.slot}?action=restore",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                record["restore"] = {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
        except Exception as exc:  # noqa: BLE001
            record["restore"] = {"status": None, "error": repr(exc)}
        record["restore_elapsed_ms"] = (time.time() - t0) * 1000.0
        status, after = slots(base, args.timeout)
        record["slot_n_past_after_restore"] = (
            (after or [{}])[args.slot].get("n_past") if isinstance(after, list) else None
        )

        rec = streaming_measure(
            base,
            args.model,
            prompt,
            args.max_tokens,
            args.timeout,
            extra={"cache_prompt": True, "ignore_eos": True},
        )
        record["verify_hit"] = rec
        record["verify_hit_prompt_tokens"] = rec.get("prompt_tokens")
        record["verify_hit_cache_n"] = rec.get("cache_n")
        record["verify_hit_cached_tokens"] = (rec.get("usage") or {}).get("prompt_tokens_details", {}).get(
            "cached_tokens"
        )
        record["verify_hit_prompt_ms"] = rec.get("prompt_ms")
        record["verify_hit_ttft_ms"] = rec.get("ttft_ms")
        # A restored slot only helps if the next request reuses it instead of
        # re-processing every token (see docs/bongo-sh.md "Slot KV persistence").
        restored_n = ((record.get("restore") or {}).get("body") or "")
        record["restore_n_restored"] = _json_int(restored_n, "n_restored")
        record["restore_verified"] = bool(
            record["verify_hit_cache_n"] and record["verify_hit_cache_n"] > 0
        )

        # A 128K-class agentic delta turn over the restored (trimmed) cache.
        # The shorter prefix is a prefix of the 256K prompt, so the slot is
        # rewound to ~131K and only the 512-token suffix is prefilled.
        record["delta_turn"] = (
            {"skipped": True, "reason": "--skip-delta"}
            if args.skip_delta
            else run_delta_turn(tk, base, args)
        )

    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
