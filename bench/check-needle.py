#!/usr/bin/env python3
"""Needle recall check against a running llama-server.

The M4.1 levers (``--load-mode none``, ``--no-op-offload``, ``--threads``) change
where host-resident MoE weights are read/computed, not the model.  This is the
cheap correctness guard: plant the standard bongo sentinel at ~50% depth of an
8K-token document, ask for it with greedy sampling, and check the answer.
Requires no GPU beyond the live server.

Stdlib only; imports the shared bench_lib + harness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import CORPUS, NEEDLE, Tokenizer, get_json, now_iso, read_text, server_root  # noqa: E402
from harness import build_needle_document, completion_call  # noqa: E402

BASE = os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BONGO_MODEL", "bongo-iq2_xs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=8192)
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("BONGO_CTX", "131072")))
    ap.add_argument("--needle-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--label", default=os.environ.get("BONGO_PROFILE_LABEL", "baseline"))
    ap.add_argument("--server-pid", type=int, default=int(os.environ.get("BONGO_SERVER_PID", "0")) or None)
    ap.add_argument("--flags-note", default=os.environ.get("BONGO_PROFILE_FLAGS_NOTE", ""))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tk = Tokenizer(BASE, timeout=args.timeout)
    if not tk.available:
        tk.count("warm")

    hard_max = max(1, args.ctx - args.needle_tokens - 64)
    filler = tk.size_to(min(args.prefix, hard_max), CORPUS, hard_max=hard_max)
    document = build_needle_document(filler)
    res = completion_call(BASE, MODEL, document, args.needle_tokens, args.timeout, cache_prompt=False)
    answer = res.get("text") or ""
    passed = NEEDLE.lower() in answer.lower()

    rec = {
        "schema": "bongo.needle-check.v1",
        "generated_at": now_iso(),
        "label": args.label,
        "base_url": BASE,
        "model": MODEL,
        "prefix_target": args.prefix,
        "needle": NEEDLE,
        "status": "pass" if passed else "fail",
        "answer": answer[:400],
        "prompt_tokens": res.get("prompt_tokens"),
        "prompt_ms": res.get("prompt_ms"),
        "http_status": res.get("status"),
        "error": res.get("error") or res.get("stream_error"),
        "flags_note": args.flags_note,
        "server": {
            "server_root": server_root(BASE),
            "props_status": (get_json(server_root(BASE), "/props", 20).status),
            "flags": [x for x in (read_text(f"/proc/{args.server_pid}/cmdline") or "").split("\x00") if x]
            if args.server_pid else None,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"{args.label}: needle {rec['status']} (prompt_n={rec['prompt_tokens']}) answer={rec['answer']!r}")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
