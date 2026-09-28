#!/usr/bin/env python3
"""Summarise the M3.4b PLE-reader engine A/B (BAS-79).

Reads the four ``matrix.json`` files written by ``bench/ple-reader/run-ab.sh``
under the results root and prints a comparison of

    baseline-off  vs  baseline-on
    m33-off       vs  m33-on

per context (prefill tok/s, decode tok/s, TTFT, peak VRAM, peak RSS) plus a
markdown summary.  Missing runs are reported, not treated as fatal, so a partial
A/B can still be summarised.

    python3 bench/ple-reader/summarize-ab.py \
        [--root bench/results/2026-09-28-ple-reader-engine]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load(root: str, label: str):
    path = os.path.join(root, label, "matrix.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def ctx_row(matrix, want):
    if not matrix:
        return None
    for result in matrix.get("results", []):
        if int(result.get("target_context", -1)) == want:
            runs = result.get("runs") or []
            if not runs:
                return None
            # one repeat: take the first completed run
            run = next((r for r in runs if r.get("status") == 200), runs[0])
            summary = result.get("summary", {})
            return {
                "prompt_tps": summary.get("prompt_tps", {}).get("median"),
                "output_tps": summary.get("output_tps", {}).get("median"),
                "ttft_ms": summary.get("ttft_ms", {}).get("median"),
                "prompt_tokens": summary.get("prompt_tokens", {}).get("median"),
                "vram_peak_bytes": (run.get("memory") or {}).get("vram_peak_bytes"),
                "rss_peak_bytes": (run.get("memory") or {}).get("system_ram_peak_bytes"),
                "needle": matrix.get("needle", {}).get("status"),
            }
    return None


def pct(new, old):
    if not new or not old:
        return None
    return (new - old) / old * 100.0


def gib(n):
    return f"{n / 1024**3:.2f}" if n else "n/a"


def main() -> int:
    default_root = "bench/results/2026-09-28-ple-reader-engine"
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    configs = {
        "baseline": ("baseline-off", "baseline-on"),
        "m33": ("m33-off", "m33-on"),
    }
    contexts = [4096, 131072]

    out = {"schema": "bongo.ple-reader-ab.v1", "root": args.root, "configs": {}}
    lines = ["# M3.4b PLE reader engine A/B", ""]
    lines.append("| config | ctx | prefill tok/s off | prefill tok/s on | delta | decode tok/s off | decode tok/s on | TTFT off (ms) | TTFT on (ms) | VRAM peak (GiB) | RSS peak (GiB) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for name, (off_label, on_label) in configs.items():
        m_off = load(args.root, off_label)
        m_on = load(args.root, on_label)
        entry = {"labels": [off_label, on_label], "contexts": {}}
        for want in contexts:
            a = ctx_row(m_off, want)
            b = ctx_row(m_on, want)
            row = {"off": a, "on": b}
            if a and b:
                row["prefill_delta_pct"] = pct(b.get("prompt_tps"), a.get("prompt_tps"))
                row["decode_delta_pct"] = pct(b.get("output_tps"), a.get("output_tps"))
            entry["contexts"][str(want)] = row

            def v(d, key):
                return d.get(key) if d else None

            delta = row.get("prefill_delta_pct")
            lines.append(
                "| {cfg} | {ctx} | {p0} | {p1} | {dp} | {d0} | {d1} | {t0} | {t1} | {v} | {r} |".format(
                    cfg=name,
                    ctx=want,
                    p0=f"{v(a,'prompt_tps'):.1f}" if v(a, "prompt_tps") else "n/a",
                    p1=f"{v(b,'prompt_tps'):.1f}" if v(b, "prompt_tps") else "n/a",
                    dp=f"{delta:+.1f}%" if delta is not None else "n/a",
                    d0=f"{v(a,'output_tps'):.2f}" if v(a, "output_tps") else "n/a",
                    d1=f"{v(b,'output_tps'):.2f}" if v(b, "output_tps") else "n/a",
                    t0=f"{v(a,'ttft_ms'):.0f}" if v(a, "ttft_ms") else "n/a",
                    t1=f"{v(b,'ttft_ms'):.0f}" if v(b, "ttft_ms") else "n/a",
                    v=gib(v(b, "vram_peak_bytes") or v(a, "vram_peak_bytes")),
                    r=gib(v(b, "rss_peak_bytes") or v(a, "rss_peak_bytes")),
                )
            )
        out["configs"][name] = entry

    text = "\n".join(lines) + "\n"
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(text)
    if not any(out["configs"][c]["contexts"].values() for c in out["configs"]):
        print("no completed runs found yet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
