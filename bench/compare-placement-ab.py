#!/usr/bin/env python3
"""Compare two Stage-1 placement matrices (BAS-76).

Reads the ``matrix.json`` written by ``bench/harness.py`` for two configurations
(for example the byte-budget ``-ot`` placement as ``--a`` and the pinned Stage 0
``--n-cpu-moe 16`` baseline as ``--b``), plus the optional ``prefix-cache.json``
written by ``bench/measure-prefix-cache.py`` for the same two runs.  Emits
``placement-ab.json`` + ``placement-ab.md``.

This exists so the Step-1 acceptance check ("+4-8 pp coverage at the same VRAM
budget, with no 128K decode regression") is computed from the raw files instead
of hand-copied numbers, and so the delta is reproducible by anyone with the
recorded matrix.json.

Every row carries the engine revision and the exact llama-server argv taken from
each matrix, so a number can be traced to the binary and flags that produced it.

Stdlib Python 3 only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

GiB = 1024**3


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:  # noqa: BLE001
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return None


def ctx_metrics(matrix, ctx):
    """Per-context medians + memory from a harness matrix.json."""
    for r in matrix.get("results", []):
        if r.get("target_context") != ctx:
            continue
        s = r.get("summary", {}) or {}
        mem = r.get("memory", {}) or {}
        return {
            "status": r.get("status"),
            "prompt_tokens": (s.get("prompt_tokens") or {}).get("median"),
            "prompt_tps": (s.get("prompt_tps") or {}).get("median"),
            "output_tps": (s.get("output_tps") or {}).get("median"),
            "ttft_ms": (s.get("ttft_ms") or {}).get("median"),
            "vram_peak_bytes": mem.get("vram_peak_bytes"),
            "ram_peak_bytes": mem.get("system_ram_peak_bytes"),
        }
    return None


def pct_delta(a, b):
    """(a - b) / b * 100, or None when either side is missing/zero."""
    if a is None or b is None or not b:
        return None
    return (a - b) / b * 100.0


def server_record(matrix):
    srv = matrix.get("server") or {}
    build = (
        srv.get("llama_cpp_build_info")
        or srv.get("build")
        or srv.get("version")
        or srv.get("llama_build")
    )
    flags = srv.get("flags") or srv.get("argv") or srv.get("cmdline")
    return {"build": build, "flags": flags}


def matrix_summary(matrix, contexts):
    out = {"contexts": {str(ctx): (ctx_metrics(matrix, ctx) or {}) for ctx in contexts}}
    needle = matrix.get("needle") or {}
    out["needle"] = {
        "status": needle.get("status"),
        "prompt_tokens": needle.get("prompt_tokens"),
    }
    out["verdict"] = matrix.get("verdict")
    out["server"] = server_record(matrix)
    return out


def prefix_runs(prefix):
    out = {}
    for run in (prefix or {}).get("runs", []):
        label = run.get("label")
        if label and label != "warmup":
            out[label] = {
                "prompt_tokens": run.get("prompt_tokens"),
                "cache_n": run.get("cache_n"),
                "prompt_tps": run.get("prompt_tps"),
                "ttft_ms": run.get("ttft_ms"),
                "output_tps": run.get("output_tps"),
                "status": run.get("status"),
            }
    return out


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def fmt_pct(v, nd=2):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}%"


def render_md(rep):
    a, b = rep["a_label"], rep["b_label"]
    lines = [
        "# Placement A/B — byte-budget `-ot` vs pinned baseline",
        "",
        f"A = `{a}`; B = `{b}`. Delta is `(A - B) / B`, so a positive prompt tok/s",
        "or output tok/s delta is an improvement and a positive TTFT delta is a",
        "regression.",
        "",
        "## Engine / flags",
        "",
        f"- A build: `{rep['a']['server'].get('build')}`",
        f"- B build: `{rep['b']['server'].get('build')}`",
        "",
        "## Throughput and memory",
        "",
        "| context | A prompt tok/s | B prompt tok/s | Δ prompt | A output tok/s | B output tok/s | Δ output | A TTFT ms | B TTFT ms | Δ TTFT | A VRAM GiB | B VRAM GiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ctx in rep["contexts"]:
        ca = rep["a"]["contexts"].get(ctx) or {}
        cb = rep["b"]["contexts"].get(ctx) or {}
        vram_a = ca.get("vram_peak_bytes")
        vram_b = cb.get("vram_peak_bytes")
        lines.append(
            "| {ctx} | {ap} | {bp} | {dp} | {ao} | {bo} | {do} | {at} | {bt} | {dt} | {av} | {bv} |".format(
                ctx=ctx,
                ap=fmt(ca.get("prompt_tps")),
                bp=fmt(cb.get("prompt_tps")),
                dp=fmt_pct(pct_delta(ca.get("prompt_tps"), cb.get("prompt_tps"))),
                ao=fmt(ca.get("output_tps")),
                bo=fmt(cb.get("output_tps")),
                do=fmt_pct(pct_delta(ca.get("output_tps"), cb.get("output_tps"))),
                at=fmt(ca.get("ttft_ms"), 1),
                bt=fmt(cb.get("ttft_ms"), 1),
                dt=fmt_pct(pct_delta(ca.get("ttft_ms"), cb.get("ttft_ms"))),
                av=fmt((vram_a / GiB) if vram_a else None, 2),
                bv=fmt((vram_b / GiB) if vram_b else None, 2),
            )
        )
    lines += [
        "",
        f"- A needle: **{rep['a']['needle'].get('status')}**; B needle: **{rep['b']['needle'].get('status')}**",
        "",
    ]
    if rep.get("prefix"):
        lines += [
            "## Prefix-cache (agentic turn) — TTFT and decode",
            "",
            "| case | A TTFT ms | B TTFT ms | Δ TTFT | A output tok/s | B output tok/s | Δ output |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label in rep["prefix"]["labels"]:
            ca = rep["prefix"]["a"].get(label, {})
            cb = rep["prefix"]["b"].get(label, {})
            lines.append(
                "| {l} | {at} | {bt} | {dt} | {ao} | {bo} | {do} |".format(
                    l=label,
                    at=fmt(ca.get("ttft_ms"), 1),
                    bt=fmt(cb.get("ttft_ms"), 1),
                    dt=fmt_pct(pct_delta(ca.get("ttft_ms"), cb.get("ttft_ms"))),
                    ao=fmt(ca.get("output_tps")),
                    bo=fmt(cb.get("output_tps")),
                    do=fmt_pct(pct_delta(ca.get("output_tps"), cb.get("output_tps"))),
                )
            )
        lines.append("")
    lines.append("Generated by `bench/compare-placement-ab.py` from the raw matrix.json files.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="matrix.json for configuration A (byte-budget)")
    p.add_argument("--b", required=True, help="matrix.json for configuration B (pinned baseline)")
    p.add_argument("--a-prefix", default=None, help="prefix-cache.json for A")
    p.add_argument("--b-prefix", default=None, help="prefix-cache.json for B")
    p.add_argument("--a-label", default="A")
    p.add_argument("--b-label", default="B")
    p.add_argument("--contexts", default="4096,131072")
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args(argv)

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    ma = load_json(args.a)
    mb = load_json(args.b)
    if ma is None or mb is None:
        return 2

    rep = {
        "schema": "bongo.placement-ab.v1",
        "a_label": args.a_label,
        "b_label": args.b_label,
        "contexts": [str(c) for c in contexts],
        "a": matrix_summary(ma, contexts),
        "b": matrix_summary(mb, contexts),
        "sources": {"a_matrix": os.path.abspath(args.a), "b_matrix": os.path.abspath(args.b)},
    }

    if args.a_prefix and args.b_prefix:
        pa, pb = prefix_runs(load_json(args.a_prefix)), prefix_runs(load_json(args.b_prefix))
        labels = [l for l in pa if l in pb]
        rep["prefix"] = {"a": pa, "b": pb, "labels": labels}
        rep["sources"]["a_prefix"] = os.path.abspath(args.a_prefix)
        rep["sources"]["b_prefix"] = os.path.abspath(args.b_prefix)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "placement-ab.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    with open(os.path.join(args.out, "placement-ab.md"), "w") as fh:
        fh.write(render_md(rep))
    print(render_md(rep))
    print(f"wrote {args.out}/placement-ab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
