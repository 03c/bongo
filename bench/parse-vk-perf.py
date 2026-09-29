#!/usr/bin/env python3
"""Parse GGML_VK_PERF_LOGGER output from a llama-server log.

The Vulkan backend prints one ``Vulkan Timings:`` block per graph compute when
``GGML_VK_PERF_LOGGER=1`` is set.  Each row is
``<op-name>: <count> x <avg-us> us = <total-us> us (<gflops> GFLOPS/s)`` and the
block ends with ``Total time: <us> us.``.  This parser turns those rows into
per-op-class GPU-busy time so the delta-turn GPU breakdown can be read directly,
instead of inferred from ablations alone (BAS-130).

Rows are attributed to the request whose ``slot print_timing ... prompt eval
time = ... / N tokens`` line follows the block, so a caller can select the
512-token delta turn.

Output: JSON list of blocks + an aggregate, on stdout or to ``--out``.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

BLOCK_HEADER = "Vulkan Timings:"
ROW_RE = re.compile(
    r"^(?P<name>.+?): (?P<count>\d+) x (?P<avg>[0-9.]+) us = (?P<total>[0-9.]+) us"
    r"(?: \((?P<gflops>[0-9.]+) GFLOPS/s\))?$"
)
TOTAL_RE = re.compile(r"^Total time: (?P<total>[0-9.]+) us\.$")
PROMPT_RE = re.compile(r"prompt eval time =\s*(?P<ms>[0-9.]+) ms\s*/\s*(?P<tokens>\d+) tokens")
LAUNCH_RE = re.compile(r"launch_slot_?: id\s+\d+ \| task (?P<task>-?\d+) \|")


def classify(name: str) -> str:
    n = name
    if "MUL_MAT_ID" in n:
        return "moe_experts"
    if "FLASH_ATTN_EXT" in n:
        return "full_attention"
    if any(k in n for k in ("GATED_DELTA_NET", "GATED_LINEAR_ATTN", "SSM_CONV", "SSM_SCAN", "LIGHTNING_INDEXER")):
        return "recurrent_deltanet"
    if "CONV" in n:
        return "conv"
    if "MUL_MAT" in n:
        return "dense_matmul"
    if any(k in n for k in ("RMS_NORM", "NORM", "GLU", "SOFT_MAX", "ROPE", "SCALE", "SILU", "GELU")):
        return "norm_activation"
    if any(k in n for k in ("CPY", "CONT", "PERMUTE", "RESHAPE", "VIEW", "TRANSPOSE", "GET_ROWS", "SET_ROWS",
                            "DUP", "PAD", "CONCAT", "REPEAT", "ARANGE", "DIAG", "FILL", "SUM", "MEAN", "CLAMP")):
        return "memory_layout"
    if any(k in n for k in ("ADD", "MUL", "SUB", "DIV", "SQR", "SQRT", "EXP", "LOG", "SIN", "COS")):
        return "elementwise"
    return "other"


def parse(log_text):
    blocks = []
    cur_rows = []
    cur_total = None
    recent_task = None
    lines = log_text.splitlines()

    def finalize(prompt_tokens):
        nonlocal cur_rows, cur_total
        if not cur_rows and cur_total is None:
            return
        ops = []
        class_totals = {}
        for raw in cur_rows:
            m = ROW_RE.match(raw)
            if not m:
                continue
            name = m.group("name")
            total = float(m.group("total"))
            cls = classify(name)
            ops.append(
                {
                    "name": name,
                    "class": cls,
                    "count": int(m.group("count")),
                    "avg_us": float(m.group("avg")),
                    "total_us": total,
                    "gflops": float(m.group("gflops")) if m.group("gflops") else None,
                }
            )
            class_totals[cls] = class_totals.get(cls, 0.0) + total
        blocks.append(
            {
                "block": len(blocks) + 1,
                "task": recent_task,
                "prompt_tokens": prompt_tokens,
                "total_us": cur_total,
                "dirty": bool(cur_total is None or sum(class_totals.values()) > cur_total * 1.5 + 50),
                "class_us": {k: round(v, 1) for k, v in sorted(class_totals.items(), key=lambda kv: -kv[1])},
                "ops": sorted(ops, key=lambda o: -o["total_us"]),
            }
        )
        cur_rows = []
        cur_total = None

    for line in lines:
        line = line.strip()
        lm = LAUNCH_RE.search(line)
        if lm:
            recent_task = int(lm.group("task"))
        if line == BLOCK_HEADER:
            # a new block starts; close the previous one without prompt info
            finalize(None)
            continue
        if cur_total is None:
            tm = TOTAL_RE.match(line)
            if tm:
                cur_total = float(tm.group("total"))
                continue
        if ROW_RE.match(line):
            cur_rows.append(line)
            continue
        pm = PROMPT_RE.search(line)
        if pm:
            finalize(int(pm.group("tokens")))
            continue
    finalize(None)
    return blocks


def aggregate(blocks, prompt_tokens=None, task=None):
    agg = {}
    gpu_busy = 0.0
    n_blocks = 0
    for b in blocks:
        if task is not None and b.get("task") != task:
            continue
        if prompt_tokens is not None and b.get("prompt_tokens") != prompt_tokens:
            continue
        n_blocks += 1
        if b.get("total_us"):
            gpu_busy += b["total_us"]
        # Rows can be interleaved across two concurrent prints; only trust the
        # class split from blocks whose rows reconcile with their Total time.
        if b.get("dirty"):
            continue
        for op in b["ops"]:
            key = op["class"]
            entry = agg.setdefault(key, {"class": key, "total_us": 0.0, "ops": {}})
            entry["total_us"] += op["total_us"]
            o = entry["ops"].setdefault(op["name"], {"total_us": 0.0, "count": 0})
            o["total_us"] += op["total_us"]
            o["count"] += op["count"]
    out = []
    for entry in agg.values():
        entry["total_us"] = round(entry["total_us"], 1)
        for o in entry["ops"].values():
            o["total_us"] = round(o["total_us"], 1)
        out.append(entry)
    out.sort(key=lambda e: -e["total_us"])
    return out, round(gpu_busy, 1), n_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="llama-server log with GGML_VK_PERF_LOGGER output")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="only aggregate blocks whose following request reported N prompt tokens")
    ap.add_argument("--task", type=int, default=None, help="only blocks from this server task id")
    ap.add_argument("--summary", action="store_true", help="print a human summary")
    args = ap.parse_args()

    with open(args.log, errors="replace") as fh:
        blocks = parse(fh.read())
    agg, gpu_busy_us, n_blocks = aggregate(blocks, args.prompt_tokens, args.task)
    # Keep the per-op evidence compact: when a task/prompt filter is given, only
    # that request's blocks belong in the artifact.
    out_blocks = blocks
    if args.task is not None:
        out_blocks = [b for b in blocks if b.get("task") == args.task]
    elif args.prompt_tokens is not None:
        out_blocks = [b for b in blocks if b.get("prompt_tokens") == args.prompt_tokens]
    result = {
        "log": args.log,
        "task": args.task,
        "prompt_tokens": args.prompt_tokens,
        "gpu_busy_us": gpu_busy_us,
        "n_blocks": n_blocks,
        "blocks": out_blocks,
        "aggregate": agg,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out} ({len(blocks)} blocks)")
    if args.summary or not args.out:
        for b in blocks:
            if args.task is not None and b.get("task") != args.task:
                continue
            print(
                f"block {b['block']:>3} task={b['task']} prompt_tokens={b['prompt_tokens']} "
                f"dirty={b['dirty']} total={b['total_us']} us classes={b['class_us']}"
            )
        print("--- aggregate ---")
        print(f"blocks={n_blocks} gpu_busy={gpu_busy_us/1000.0:.1f} ms")
        grand = sum(e["total_us"] for e in agg)
        for e in agg:
            pct = 100.0 * e["total_us"] / grand if grand else 0.0
            print(f"{e['class']:>20}: {e['total_us']:>12.1f} us  {pct:5.1f}%")
        print(f"{'TOTAL':>20}: {grand:>12.1f} us")


if __name__ == "__main__":
    main()
