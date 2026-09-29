#!/usr/bin/env python3
"""Summarize the M4.4 (BAS-159) 4K decode lever screen.

Reads every ``<out>/<config>/decode4k.json`` written by
``bench/run-m4.4-decode-profile.sh`` and writes ``summary.json`` with the
per-config decode distribution plus ``summary.md`` with the ranked table.

Stdlib only; no server needed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics


def load_configs(out_root):
    rows = []
    for name in sorted(os.listdir(out_root)):
        path = os.path.join(out_root, name, "decode4k.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            rec = json.load(fh)
        runs = [r for r in rec.get("runs", []) if r.get("status") == 200 and r.get("output_tps")]
        if not runs:
            continue
        tps = [r["output_tps"] for r in runs]
        prompt_ms = [r["prompt_ms"] for r in runs if r.get("prompt_ms")]
        rows.append(
            {
                "config": name,
                "repeats": len(tps),
                "tps": [round(t, 3) for t in tps],
                "tps_median": round(statistics.median(tps), 3),
                "tps_min": round(min(tps), 3),
                "tps_max": round(max(tps), 3),
                "prompt_ms_median": round(statistics.median(prompt_ms), 1) if prompt_ms else None,
                "output_tokens": runs[0].get("output_tokens"),
                "note": _read_note(out_root, name),
                "env": _read_first_line(out_root, name, "env.txt"),
                "flags": _read_first_line(out_root, name, "command.txt"),
            }
        )
    return rows


def _read_note(out_root, name):
    path = os.path.join(out_root, name, "note.txt")
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _read_first_line(out_root, name, fname):
    path = os.path.join(out_root, name, fname)
    try:
        with open(path) as fh:
            return fh.readline().strip()
    except OSError:
        return ""


def render(rows, baseline):
    base = next((r for r in rows if r["config"] == baseline), None)
    base_tps = base["tps_median"] if base else None
    lines = [
        "# M4.4 4K decode lever screen",
        "",
        "4096-token prompt, 128 generated tokens, `cache_prompt=false`, discarded warm-up,",
        "engine llama.cpp `b11223` + the M4.2 patch, Vulkan1, iq2_xs, q8 KV, `--ctx-size 131072`,",
        "`--load-mode none`, `GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1`.",
        "",
        "| config | repeats | 4K decode tok/s | median | vs `%s` |" % baseline,
        "| --- | --: | --- | --: | --: |",
    ]
    for r in sorted(rows, key=lambda r: -r["tps_median"]):
        delta = ""
        if base_tps:
            delta = f"{(r['tps_median'] / base_tps - 1) * 100:+.2f}%"
        runs = " / ".join(f"{t:.2f}" for t in r["tps"])
        lines.append(
            f"| `{r['config']}` | {r['repeats']} | {runs} | **{r['tps_median']:.2f}** | {delta} |"
        )
    lines.append("")
    lines.append("| config | note |")
    lines.append("| --- | --- |")
    for r in sorted(rows, key=lambda r: -r["tps_median"]):
        lines.append(f"| `{r['config']}` | {r['note']} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="bench/results/2026-09-29-m4.4-decode")
    ap.add_argument("--baseline", default="decode16")
    args = ap.parse_args()

    rows = load_configs(args.out_root)
    base = next((r for r in rows if r["config"] == args.baseline), None)
    base_tps = base["tps_median"] if base else None
    for r in rows:
        r["vs_baseline_pct"] = round((r["tps_median"] / base_tps - 1) * 100, 2) if base_tps else None

    summary = {"baseline": args.baseline, "baseline_tps_median": base_tps, "configs": rows}
    with open(os.path.join(args.out_root, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    text = render(rows, args.baseline)
    with open(os.path.join(args.out_root, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
