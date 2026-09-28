#!/usr/bin/env python3
"""Profile the warm-prefix (cached-turn) path of a running llama-server.

This is the M3.6 measurement (BAS-130).  It answers "where does a 512-token
delta turn go?" by timing the server-reported prefill of the *delta* only,
against a prefix that is already resident in the slot KV.

For each prefix size P it measures:

  cold   : P tokens, cache_prompt=false   -> primes the slot (the expensive step)
  hit    : P tokens, cache_prompt=true    -> full reuse, P cached, ~4 new tokens
  grow_D : P+D tokens, cache_prompt=true  -> the measured delta turn (D new tokens)
  decode : P tokens, cache_prompt=true, max_tokens=N
                                          -> decode rate at context P after a hit

The `hit` record is the fixed per-request forward cost; `grow_D - hit` is the
marginal cost of processing D new tokens while attending over the cached prefix.
Measuring several D gives a per-token slope instead of a single point.

The decomposition into components (attention / MoE experts / SSM / dense / host)
is not done here: it comes from *ablation configs* (different server flags) run
by ``bench/run-warm-prefix-profile.sh`` against the same prefixes, and compared
by ``bench/analyze-warm-prefix.py``.  This script records the raw evidence.

Stdlib only; imports the shared bench_lib.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import CORPUS, Tokenizer, get_json, now_iso, read_text, server_root  # noqa: E402
from harness import streaming_measure  # noqa: E402

BASE = os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BONGO_MODEL", "bongo-iq2_xs")


def server_record(pid):
    """Engine revision, props and the exact server argv, so a number is reproducible."""
    root = server_root(BASE)
    props_res = get_json(root, "/props", 20)
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


def run_case(tk, label, prompt, cache_prompt, max_tokens=1, timeout=1800, ignore_eos=True):
    t0 = time.perf_counter()
    rec = streaming_measure(
        BASE,
        MODEL,
        prompt,
        max_tokens,
        timeout,
        cache_prompt=bool(cache_prompt),
        extra={"ignore_eos": ignore_eos} if ignore_eos else None,
    )
    rec["client_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    rec["label"] = label
    rec["cache_prompt"] = bool(cache_prompt)
    rec["max_tokens"] = max_tokens
    rec["ts"] = time.time()
    print(
        f"{label:26s} status={rec.get('status')} "
        f"prompt_n={rec.get('prompt_tokens')} cache_n={rec.get('cache_n')} "
        f"prompt_ms={rec.get('prompt_ms')} prompt_tps={rec.get('prompt_tps')} "
        f"ttft_ms={rec.get('ttft_ms')} out_n={rec.get('output_tokens')} "
        f"out_tps={rec.get('output_tps')} wall_ms={rec.get('wall_ms')}",
        flush=True,
    )
    return rec


def make_delta(tk, tokens, tag):
    """A distinct suffix blob, so it is not a prefix of any prior prompt."""
    seed = f" | warm-prefix delta {tag} "
    return tk.size_to(tokens, corpus="The reviewer noted the following delta. " + CORPUS, seed=seed)


def timed_tokenize(tk, text, timeout=300):
    """Time a raw /tokenize call; part of serving overhead, measured on the host."""
    t0 = time.perf_counter()
    n = tk.count(text)
    return {"tokens": n, "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3)}


def main():
    ap = argparse.ArgumentParser(description="warm-prefix (cached-turn) profiler")
    ap.add_argument("--prefixes", default="16384,131072")
    ap.add_argument("--deltas", default="128,512,1024")
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("BONGO_CTX", "131072")),
                    help="server n_ctx; prefixes are capped so prefix+delta fits")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--label", default=os.environ.get("BONGO_PROFILE_LABEL", "baseline"))
    ap.add_argument("--server-pid", type=int, default=int(os.environ.get("BONGO_SERVER_PID", "0")) or None)
    ap.add_argument("--flags-note", default=os.environ.get("BONGO_PROFILE_FLAGS_NOTE", ""))
    ap.add_argument("--warmup-tokens", type=int, default=256)
    args = ap.parse_args()

    tk = Tokenizer(BASE, timeout=1800)
    if not tk.available:
        tk.count("warmup")

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]

    results = {
        "schema": "bongo.warm-prefix-profile.v1",
        "generated_at": now_iso(),
        "label": args.label,
        "base_url": BASE,
        "model": MODEL,
        "deltas": deltas,
        "prefixes": prefixes,
        "server": server_record(args.server_pid),
        "flags_note": args.flags_note,
        "warmup": None,
        "tokenize_overhead": [],
        "points": [],
    }

    # One discarded request so the first measured request is not first-touch.
    results["warmup"] = run_case(
        tk, "warmup", tk.size_to(args.warmup_tokens), False, max_tokens=1, timeout=600
    )

    for p in prefixes:
        # Never build a prompt that overflows n_ctx once the largest delta (and
        # the generated token) is appended.
        hard_max = max(1, args.ctx - max(deltas) - 16)
        p_text = tk.size_to(p, hard_max=hard_max)
        real_p = tk.count(p_text)
        if real_p > hard_max:
            real_p = hard_max
        print(f"\n== prefix target {p} (actual {real_p}) ==", flush=True)
        results["tokenize_overhead"].append(
            {"prefix_target": p, "prefix_tokens": real_p, **timed_tokenize(tk, p_text)}
        )
        point = {"prefix_target": p, "prefix_tokens": real_p, "runs": []}

        # Prime the slot cold; this is the expensive step and is recorded so the
        # reader can see the warm/cold ratio.
        point["runs"].append(run_case(tk, f"cold_p{real_p}", p_text, False))
        # Full reuse: verifies the prefix is actually in the slot KV.
        point["runs"].append(run_case(tk, f"hit_p{real_p}", p_text, True))
        # Deltas, ascending, so each grow starts from the same cached P.
        for d in deltas:
            d_text = make_delta(tk, d, f"p{real_p}_d{d}")
            grow_text = p_text + d_text
            real_grow = tk.count(grow_text)
            if real_grow + 1 > args.ctx:
                print(f"skip grow d={d}: {real_grow}+1 > ctx {args.ctx}", flush=True)
                continue
            rec = run_case(tk, f"grow_p{real_p}_d{d}", grow_text, True)
            rec["delta_target"] = d
            rec["prompt_real_tokens"] = real_grow
            point["runs"].append(rec)
        # Decode rate at this context after a fresh full hit.
        run_case(tk, f"hit2_p{real_p}", p_text, True)
        point["runs"].append(
            run_case(
                tk,
                f"decode_p{real_p}",
                p_text,
                True,
                max_tokens=args.decode_tokens,
                ignore_eos=True,
            )
        )
        point["runs"][-1]["decode_tokens_requested"] = args.decode_tokens
        results["points"].append(point)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
