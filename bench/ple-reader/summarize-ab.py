#!/usr/bin/env python3
"""Summarise the M3.4b PLE-reader engine A/B (BAS-79).

Reads the four ``matrix.json`` files written by ``bench/ple-reader/run-ab.sh``
under the results root and prints a comparison of

    baseline-off  vs  baseline-on
    m33-off       vs  m33-on

per context (prefill tok/s, decode tok/s, TTFT, peak VRAM, peak RSS) plus a
provenance block (tier, flags, engine revision) and an explicit **acceptance
verdict** for the three bounds on the issue:

  1. no more than ``--max-regression``% regression with ``--ple-reader on``
     against the repinned baseline config and against the M3.3 config,
  2. a measured prefill gain where the mmap path was faulting,
  3. RSS must not grow by the table size (the table is never forced resident).

Missing runs are reported, not treated as fatal, so a partial A/B can still be
summarised; the verdict then reads ``incomplete`` rather than ``pass``/``fail``.

    python3 bench/ple-reader/summarize-ab.py \
        [--root bench/results/2026-09-28-ple-reader-engine] \
        [--md-out summary.md] [--json-out summary.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# The PLE table the reader serves: 320001536 rows x 90 B/row, as reported by
# llama-server when it enables the reader.  RSS must not grow by this much.
DEFAULT_TABLE_BYTES = 320001536 * 90

# "a gain" has to clear measurement noise; a 1% move is the reporting floor.
GAIN_FLOOR_PCT = 1.0

# Which direction counts as better per metric.  Throughput is higher-better;
# TTFT is lower-better, so a negative TTFT delta is an improvement and must not
# be counted as a regression.
METRICS = {
    "prefill": "higher_better",
    "decode": "higher_better",
    "ttft": "lower_better",
}


def load(root: str, label: str):
    path = os.path.join(root, label, "matrix.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_provenance(root: str, label: str):
    """Flags/tier actually used for a config, from run-ab.sh's server-flags.json."""
    path = os.path.join(root, label, "server-flags.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    argv = data.get("argv") or []

    def flag(name):
        return argv[argv.index(name) + 1] if name in argv else None

    model = flag("--model") or ""
    return {
        "argv": argv,
        "tier": model.rstrip("/").split("/")[-2] if "/" in model else None,
        "ctx": flag("--ctx-size"),
        "n_cpu_moe": flag("--n-cpu-moe"),
        "cache_type_k": flag("--cache-type-k"),
        "cache_type_v": flag("--cache-type-v"),
        "flash_attn": flag("--flash-attn"),
        "device": flag("--device"),
        "ple_reader": flag("--ple-reader"),
        "override_tensor": flag("--override-tensor"),
    }


def engine_revision(bin_dir: str):
    """Best-effort engine commit for the PLE tree, for the provenance header."""
    if not bin_dir or not os.path.isdir(os.path.join(bin_dir, "..")):
        return None
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(bin_dir.rstrip("/")), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


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


def fmt(v, spec="{:.1f}"):
    if v is None:
        return "n/a"
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    default_root = "bench/results/2026-09-28-ple-reader-engine"
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--max-regression", type=float, default=5.0,
                    help="percent drop tolerated with --ple-reader on (issue bound: 5)")
    ap.add_argument("--table-bytes", type=int, default=DEFAULT_TABLE_BYTES,
                    help="size of the PLE table the reader must not force resident")
    ap.add_argument("--engine-bin", default=os.path.expanduser(
        "~/.bongo/engine/llama.cpp-ple/build-vulkan/bin"),
        help="Patched llama-server build dir, for the provenance header")
    args = ap.parse_args()

    configs = {
        "baseline": ("baseline-off", "baseline-on"),
        "m33": ("m33-off", "m33-on"),
    }
    contexts = [4096, 131072]

    out = {
        "schema": "bongo.ple-reader-ab.v1",
        "root": args.root,
        "engine_revision": engine_revision(args.engine_bin),
        "table_bytes": args.table_bytes,
        "max_regression_pct": args.max_regression,
        "configs": {},
    }
    lines = ["# M3.4b PLE reader engine A/B", ""]
    lines.append("| config | ctx | prefill tok/s off | prefill tok/s on | delta | decode tok/s off | decode tok/s on | TTFT off (ms) | TTFT on (ms) | VRAM peak (GiB) | RSS peak (GiB) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    regressions, gains, rss_rows, compared, missing = [], [], [], 0, []

    for name, (off_label, on_label) in configs.items():
        m_off = load(args.root, off_label)
        m_on = load(args.root, on_label)
        entry = {
            "labels": [off_label, on_label],
            "provenance": {
                "off": load_provenance(args.root, off_label),
                "on": load_provenance(args.root, on_label),
            },
            "contexts": {},
        }
        for want in contexts:
            a = ctx_row(m_off, want)
            b = ctx_row(m_on, want)
            row = {"off": a, "on": b}
            if a and b:
                compared += 1
                row["prefill_delta_pct"] = pct(b.get("prompt_tps"), a.get("prompt_tps"))
                row["decode_delta_pct"] = pct(b.get("output_tps"), a.get("output_tps"))
                row["ttft_delta_pct"] = pct(b.get("ttft_ms"), a.get("ttft_ms"))
                for metric in ("prefill", "decode", "ttft"):
                    d = row.get(f"{metric}_delta_pct")
                    if d is None:
                        continue
                    # throughput drops are regressions; a TTFT *rise* is one
                    if METRICS[metric] == "higher_better":
                        regressed = d < -args.max_regression
                    else:
                        regressed = d > args.max_regression
                    if regressed:
                        regressions.append(
                            {"config": name, "ctx": want, "metric": metric, "delta_pct": d})
                if (row.get("prefill_delta_pct") or 0) > GAIN_FLOOR_PCT:
                    gains.append({"config": name, "ctx": want,
                                  "delta_pct": row["prefill_delta_pct"]})
                if a.get("rss_peak_bytes") and b.get("rss_peak_bytes"):
                    rss_rows.append({
                        "config": name, "ctx": want,
                        "rss_growth_bytes": b["rss_peak_bytes"] - a["rss_peak_bytes"],
                        "rss_off_bytes": a["rss_peak_bytes"],
                        "rss_on_bytes": b["rss_peak_bytes"],
                    })
            else:
                missing.append({"config": name, "ctx": want,
                                "have_off": bool(a), "have_on": bool(b)})
            entry["contexts"][str(want)] = row

            def v(d, key):
                return d.get(key) if d else None

            lines.append(
                "| {cfg} | {ctx} | {p0} | {p1} | {dp} | {d0} | {d1} | {t0} | {t1} | {v} | {r} |".format(
                    cfg=name,
                    ctx=want,
                    p0=fmt(v(a, "prompt_tps")),
                    p1=fmt(v(b, "prompt_tps")),
                    dp=f"{row['prefill_delta_pct']:+.1f}%" if row.get("prefill_delta_pct") is not None else "n/a",
                    d0=fmt(v(a, "output_tps"), "{:.2f}"),
                    d1=fmt(v(b, "output_tps"), "{:.2f}"),
                    t0=fmt(v(a, "ttft_ms"), "{:.0f}"),
                    t1=fmt(v(b, "ttft_ms"), "{:.0f}"),
                    v=gib(v(b, "vram_peak_bytes") or v(a, "vram_peak_bytes")),
                    r=gib(v(b, "rss_peak_bytes") or v(a, "rss_peak_bytes")),
                )
            )
        out["configs"][name] = entry

    # ---- acceptance -------------------------------------------------------
    rss_bad = [r for r in rss_rows
               if abs(r["rss_growth_bytes"]) >= args.table_bytes]
    if compared == 0:
        verdict = "incomplete"
    elif regressions:
        verdict = "fail"
    elif not gains:
        verdict = "fail-no-gain"
    else:
        verdict = "pass"

    out["acceptance"] = {
        "verdict": verdict,
        "compared_pairs": compared,
        "regressions": regressions,
        "prefill_gains": gains,
        "rss_rows": rss_rows,
        "rss_violations": rss_bad,
        "missing": missing,
    }

    lines += ["", "## Acceptance", ""]
    lines.append(f"- **verdict: `{verdict}`** ({compared} off/on pair(s) compared, "
                 f"tolerance -{args.max_regression}%)")
    lines.append(f"- engine revision: `{out['engine_revision'] or 'n/a'}`; "
                 f"PLE table: {gib(args.table_bytes)} GiB")
    lines.append(f"- >{args.max_regression}% regressions: "
                 f"{len(regressions)}" + (f" -> {regressions}" if regressions else ""))
    lines.append(f"- prefill gains: {len(gains)}" + (f" -> {gains}" if gains else ""))
    for r in rss_rows:
        lines.append(f"- RSS {r['config']} @{r['ctx']}: off {gib(r['rss_off_bytes'])} GiB -> "
                     f"on {gib(r['rss_on_bytes'])} GiB (growth {gib(r['rss_growth_bytes'])} GiB, "
                     f"table {gib(args.table_bytes)} GiB)")
    if rss_bad:
        lines.append(f"- **RSS grew by the table size: {rss_bad}**")
    if missing:
        lines.append(f"- missing pairs: {missing}")

    lines += ["", "## Provenance", ""]
    for name, (off_label, on_label) in configs.items():
        prov = out["configs"][name]["provenance"]
        for side, label in (("off", off_label), ("on", on_label)):
            p = prov.get(side)
            if not p:
                lines.append(f"- `{label}`: no server-flags.json yet")
                continue
            lines.append(
                f"- `{label}`: tier `{p.get('tier')}`, ctx `{p.get('ctx')}`, "
                f"n-cpu-moe `{p.get('n_cpu_moe')}`, cache `{p.get('cache_type_k')}/{p.get('cache_type_v')}`, "
                f"flash-attn `{p.get('flash_attn')}`, device `{p.get('device')}`, "
                f"--ple-reader `{p.get('ple_reader')}`"
            )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(text)
    if compared == 0:
        print("no completed runs found yet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
