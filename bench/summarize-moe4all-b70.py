#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-TBD
"""Summarise the MoE4All/INFR B70 matrix raw artifacts (BAS-181).

Reads `bench/results/2026-09-29-moe4all-b70/raw/*.meta.json` plus the sibling
`*.log` for the MTP A/B chat turns, and prints a markdown summary on stdout.
Deterministic: no timestamps of its own, so the committed `matrix.md` only
changes when the measurements do.

Usage: bench/summarize-moe4all-b70.py [--raw DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

RUN_RE = re.compile(
    r"\[prefill (\d+) tok @ ([\d.]+) tok/s \(([\d.]+) ms\) \| decode (\d+) tok @ ([\d.]+) tok/s\]"
)


def load_bench(metas: list[dict]) -> list[dict]:
    rows = []
    for m in metas:
        if m.get("kind") != "bench":
            continue
        j = m.get("json") or [{}]
        j = j[0] if j else {}
        reps = j.get("reps_ts") or []
        rows.append(
            {
                "case": m["case"],
                "model": m["model"],
                "ctx": m["ctx"],
                "cache": m["cache"],
                "depth": m["depth"],
                "metric": f"pp{m['prefill']}" if m.get("prefill") else f"tg{m['decode']}",
                "cold": m["case"].endswith("_cold"),
                "avg": j.get("avg_ts"),
                "reps": reps,
                "median": statistics.median(reps) if reps else None,
                "min": min(reps) if reps else None,
                "max": max(reps) if reps else None,
                "submit_cap": j.get("submit_cap"),
                "exit": m["exit"],
                "wall": m["wall_secs"],
            }
        )
    return rows


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def bench_table(rows: list[dict]) -> str:
    out = [
        "| quant | ctx (flag) | cache | depth | metric | avg tok/s | reps (tok/s) | median | min–max | submit cap | exit |",
        "| --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    order = {"iq2xs": 0, "q2_0": 1, "iq3xs": 2}
    rows = sorted(
        rows,
        key=lambda r: (order.get(r["model"], 9), r["ctx"], r["depth"], r["metric"], r["cache"]),
    )
    for r in rows:
        reps = ", ".join(fmt(v) for v in r["reps"])
        tag = " (cold)" if r.get("cold") else ""
        out.append(
            f"| `{r['model']}`{tag} | {r['ctx']} | {r['cache']} | {r['depth']} | {r['metric']} | "
            f"{fmt(r['avg'])} | {reps} | {fmt(r['median'])} | "
            f"{fmt(r['min'])}–{fmt(r['max'])} | {r['submit_cap']} | {r['exit']} |"
        )
    return "\n".join(out)


def mtp_rows(raw: Path) -> list[dict]:
    rows = []
    for log in sorted(raw.glob("iq2xs_ctx4096_*_rep*.log")):
        name = log.stem
        mode = "mtp" if "_mtp_" in name else "ordinary"
        rep = name.rsplit("rep", 1)[-1]
        txt = log.read_text(errors="replace")
        m = RUN_RE.search(txt)
        dec = float(m.group(5)) if m else None
        alpha = None
        am = re.search(r"alpha=([0-9.]+)", txt)
        if am:
            alpha = float(am.group(1))
        acc = None
        cm = re.search(r"\[qwen4 mtp summary\] (\d+) cycles, (\d+)/(\d+) accepted", txt)
        if cm:
            acc = (int(cm.group(2)), int(cm.group(3)))
        out = (raw / f"{name}.out")
        reply = out.read_text(errors="replace").strip() if out.exists() else ""
        rows.append(
            {
                "name": name,
                "mode": mode,
                "rep": rep,
                "decode": dec,
                "alpha": alpha,
                "acc": acc,
                "reply": reply,
            }
        )
    return rows


def mtp_table(rows: list[dict]) -> str:
    out = [
        "| arm | rep | decode tok/s | accept (alpha) | reply |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: (r["mode"], r["rep"])):
        acc = f"{r['acc'][0]}/{r['acc'][1]} (alpha={r['alpha']:.3f})" if r["acc"] else "n/a"
        out.append(
            f"| {r['mode']} | {r['rep']} | {fmt(r['decode'], 1)} | {acc} | `{r['reply']}` |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        default=str(Path(__file__).resolve().parent / "results/2026-09-29-moe4all-b70/raw"),
    )
    args = ap.parse_args()
    raw = Path(args.raw)
    metas = []
    for p in sorted(raw.glob("*.meta.json")):
        try:
            metas.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue

    bench = load_bench(metas)
    print("# MoE4All/INFR Arc B70 matrix — generated summary\n")
    print(f"Cases with meta: {len(metas)} (bench: {len(bench)}).\n")
    print("## Quant x context (flag) / depth table\n")
    print(bench_table(bench))
    print("\n## MTP A/B (`iq2xs`, ctx 4096, depth 0, `--temp 0`, 32 new tokens)\n")
    print(mtp_table(mtp_rows(raw)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
