#!/usr/bin/env python3
"""Rank the GPU terms of one 4K decode step from a GGML_VK_PERF_LOGGER log.

Companion to ``bench/run-m4.4-decode-profile.sh`` (BAS-159).  The Vulkan perf
logger prints one ``Vulkan Timings:`` block per ``ggml_backend_graph_compute``.
The llama.cpp scheduler splits a decode step into many Vulkan sub-graphs (one per
MoE/offload boundary, ~70 per token on this model), so a single decode step is a
*run* of blocks, not one block, and per-op names alone cannot separate prefill
from decode.

The profile runner therefore primes the 4096-token prefix with one request and
then measures a ``cache_prompt=true`` request whose prepended suffix is a single
token.  Every graph that second request computes is a one-token graph, so the
decode step is simply *all* perf blocks after that request's ``launch_slot_``
line -- the last one in the log.  This tool selects exactly that window, divides
by the number of one-token graphs (generated tokens + 1), and prints the ranked
per-class and per-op table.

Usage:
    bench/analyze-decode-perf.py <llama-server.log> [--out FILE] [--json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

# ``parse-vk-perf.py`` is not importable by name (hyphen); load it by path so the
# op classifier and the row regex stay in one place.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("parse_vk_perf", os.path.join(_HERE, "parse-vk-perf.py"))
parse_vk_perf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parse_vk_perf)
classify = parse_vk_perf.classify
ROW_RE = parse_vk_perf.ROW_RE
TOTAL_RE = parse_vk_perf.TOTAL_RE

LAUNCH_RE = re.compile(r"launch_slot_?: id\s+\d+ \| task (?P<task>-?\d+) \|")
EVAL_RE = re.compile(r"eval time =\s*(?P<ms>[0-9.]+) ms\s*/\s*(?P<tokens>\d+) tokens")


def blocks_with_positions(lines):
    """Return [(start_line, end_line, rows, total_us)] for every perf block."""
    blocks = []
    cur = None
    for n, line in enumerate(lines):
        if line.strip() == parse_vk_perf.BLOCK_HEADER:
            if cur is not None:
                blocks.append(cur)
            cur = {"start": n, "rows": [], "total_us": None}
            continue
        if cur is None:
            continue
        s = line.strip()
        tm = TOTAL_RE.match(s)
        if tm:
            cur["total_us"] = float(tm.group("total"))
            continue
        if ROW_RE.match(s):
            cur["rows"].append(s)
    if cur is not None:
        blocks.append(cur)
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="llama-server log written with GGML_VK_PERF_LOGGER=1")
    ap.add_argument("--out", default=None, help="write the ranked JSON here")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the table")
    ap.add_argument("--steps", type=int, default=None,
                    help="one-token graphs in the measured window (default: output tokens + 1)")
    args = ap.parse_args()

    with open(args.log, errors="replace") as fh:
        lines = fh.read().splitlines()

    launch_at = None
    for n, line in enumerate(lines):
        m = LAUNCH_RE.search(line)
        if m:
            launch_at = n
    if launch_at is None:
        print(f"no launch_slot_ line found in {args.log}", file=sys.stderr)
        return 2

    steps = args.steps
    if steps is None:
        evals = list(EVAL_RE.finditer("\n".join(lines[launch_at:])))
        if not evals:
            print("no 'eval time' line after the last launch; pass --steps", file=sys.stderr)
            return 2
        steps = int(evals[-1].group("tokens")) + 1

    decode = [b for b in blocks_with_positions(lines) if b["start"] > launch_at]
    if not decode:
        print(f"no perf blocks after the measured launch in {args.log}", file=sys.stderr)
        return 2

    class_us: dict[str, float] = {}
    op_us: dict[str, float] = {}
    gpu_busy_us = 0.0
    for b in decode:
        gpu_busy_us += b["total_us"] or 0.0
        for raw in b["rows"]:
            m = ROW_RE.match(raw)
            if not m:
                continue
            name = m.group("name")
            total = float(m.group("total"))
            cls = classify(name)
            class_us[cls] = class_us.get(cls, 0.0) + total
            op_us[name] = op_us.get(name, 0.0) + total

    per_step = gpu_busy_us / steps
    classes = sorted(
        ({"class": cls, "us_per_step": total / steps, "pct": 100.0 * total / gpu_busy_us}
         for cls, total in class_us.items()),
        key=lambda r: -r["us_per_step"],
    )
    ops = sorted(
        ({"name": name, "us_per_step": total / steps, "pct": 100.0 * total / gpu_busy_us,
          "total_us": round(total, 1)}
         for name, total in op_us.items()),
        key=lambda r: -r["us_per_step"],
    )

    result = {
        "log": args.log,
        "measured_launch_line": launch_at,
        "n_blocks_in_window": len(decode),
        "n_decode_steps": steps,
        "n_blocks_per_step": len(decode) / steps,
        "gpu_busy_us_per_step": per_step,
        "classes": classes,
        "ops": ops,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"measured window: {len(decode)} blocks / {steps} decode steps "
          f"= {len(decode) / steps:.1f} GPU sub-graphs per step")
    print(f"GPU busy per decode step: {per_step / 1000.0:.2f} ms")
    print()
    print(f"{'rank':>4}  {'term':<22} {'ms/step':>9} {'pct':>6}")
    for i, row in enumerate(classes, 1):
        print(f"{i:>4}  {row['class']:<22} {row['us_per_step'] / 1000.0:>9.3f} {row['pct']:>5.1f}%")
    print(f"{'':>4}  {'TOTAL':<22} {per_step / 1000.0:>9.3f} {'100.0':>5}%")
    print()
    print("individual ops >= 2% of the decode step:")
    for o in ops:
        if o["pct"] >= 2.0:
            print(f"      {o['us_per_step'] / 1000.0:>8.3f} ms  {o['pct']:>5.1f}%  {o['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
