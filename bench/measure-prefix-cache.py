#!/usr/bin/env python3
"""Measure llama-server prefix-cache (prompt reuse) behaviour for agentic turns.

The Stage 0 harness always sends ``cache_prompt: false``, so every published
number is a *cold* prefill.  An agentic coding session is the opposite: a small
first prompt that grows, where each turn re-sends the whole history and only the
new suffix is new.  This script measures that cached path directly.

Sequence per prefix size P:
  cold   : P,          cache_prompt=false  -> full prefill cost
  hit    : P,          cache_prompt=true   -> full KV reuse
  grow   : P + D,      cache_prompt=true   -> prefill only the D-token delta
  grow2  : P + D,      cache_prompt=true   -> repeat, should be a full hit

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

from bench_lib import Tokenizer, CORPUS, now_iso  # noqa: E402
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
        extra={"cache_prompt": bool(cache_prompt), "ignore_eos": True},
    )
    rec["label"] = label
    rec["ts"] = time.time()
    print(
        f"{label:26s} status={rec.get('status')} "
        f"prompt_n={rec.get('prompt_tokens')} prompt_ms={rec.get('prompt_ms')} "
        f"prompt_tps={rec.get('prompt_tps')} ttft_ms={rec.get('ttft_ms')} "
        f"out_tps={rec.get('output_tps')} wall_ms={rec.get('wall_ms')}",
        flush=True,
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="4096,16384,24576")
    ap.add_argument("--delta", type=int, default=512)
    ap.add_argument("--out", default="bench/results/2026-09-28-prefix-cache")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tk = Tokenizer(BASE, timeout=900)
    if not tk.available:
        tk.count("warmup")

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    results = {"schema": "bongo.prefix-cache.v1", "generated_at": now_iso(), "base_url": BASE,
               "model": MODEL, "delta_tokens": args.delta, "runs": []}

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

    suffix = "json" if args.out.endswith(".json") else ""
    path = args.out if suffix else os.path.join(args.out, "prefix-cache.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
