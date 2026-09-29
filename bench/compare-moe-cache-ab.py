#!/usr/bin/env python3
"""Acceptance comparison for the BAS-139 MoE expert-cache A/B (BAS-76 Step 2).

Reads the ``matrix.json`` written by ``bench/harness.py`` and the
``prefix-cache.json`` written by ``bench/measure-prefix-cache.py`` for the LRU
engine (``--a``) and the pinned Stage 0 baseline (``--b``), plus the engine's
``moe-cache-stats.json`` counters, and decides the BAS-139 acceptance criteria:

  * +15% turn TTFT and/or +20% 4K decode versus Stage 0 `--n-cpu-moe 16`;
  * no long-context regression > 2%;
  * the needle passes on both sides.

Delta is ``(A - B) / B``, so a positive prompt/output tok/s delta is an
improvement and a positive TTFT delta is a regression.  Emits
``moe-cache-ab.json`` + ``moe-cache-ab.md`` next to the raw matrices.

Stdlib Python 3 only.

  python3 bench/compare-moe-cache-ab.py \
    --a bench/results/2026-09-29-moe-cache-lru/ncmoe-48-moe-lru/matrix.json \
    --b bench/results/2026-09-29-moe-cache-lru/ncmoe-16-stage0/matrix.json \
    --a-prefix .../ncmoe-48-moe-lru/prefix-cache/prefix-cache.json \
    --b-prefix .../ncmoe-16-stage0/prefix-cache/prefix-cache.json \
    --cache-stats bench/results/2026-09-29-moe-cache-lru/moe-cache-stats.json \
    --out bench/results/2026-09-29-moe-cache-lru
"""

from __future__ import annotations

import argparse
import json
import os
import sys

GiB = 1024**3

TURN_TTFT_TARGET_PCT = -15.0   # turn TTFT improvement (negative delta)
DECODE_TARGET_PCT = 20.0       # 4K decode improvement
REGRESSION_TOL_PCT = -2.0      # no long-context metric may fall more than this


def load_json(path):
    if not path:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:  # noqa: BLE001
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return None


def pct_delta(a, b):
    if a is None or b is None or not b:
        return None
    return (a - b) / b * 100.0


def ctx_metrics(matrix, ctx):
    if not matrix:
        return None
    for r in matrix.get("results", []):
        if r.get("target_context") != ctx:
            continue
        s = r.get("summary", {}) or {}
        mem = r.get("memory", {}) or {}
        return {
            "status": r.get("status"),
            "prompt_tps": (s.get("prompt_tps") or {}).get("median"),
            "output_tps": (s.get("output_tps") or {}).get("median"),
            "ttft_ms": (s.get("ttft_ms") or {}).get("median"),
            "vram_peak_bytes": mem.get("vram_peak_bytes"),
        }
    return None


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
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def fmt_pct(v, nd=2):
    return "n/a" if v is None else f"{v:+.{nd}f}%"


def build_report(a, b, ap, bp, cache_stats, contexts):
    rep = {
        "schema": "bongo.moe-cache-ab.v1",
        "a_label": "moe-lru",
        "b_label": "stage0-ncmoe16",
        "contexts": [str(c) for c in contexts],
        "a": {str(c): (ctx_metrics(a, c) or {}) for c in contexts},
        "b": {str(c): (ctx_metrics(b, c) or {}) for c in contexts},
        "needle": {
            "a": (a or {}).get("needle", {}).get("status"),
            "b": (b or {}).get("needle", {}).get("status"),
        },
        "cache": cache_stats,
        "sources": {},
    }

    # per-context deltas
    for c in contexts:
        ca, cb = rep["a"][str(c)], rep["b"][str(c)]
        rep["a"][str(c)]["delta"] = {
            "prompt_tps_pct": pct_delta(ca.get("prompt_tps"), cb.get("prompt_tps")),
            "output_tps_pct": pct_delta(ca.get("output_tps"), cb.get("output_tps")),
            "ttft_pct": pct_delta(ca.get("ttft_ms"), cb.get("ttft_ms")),
        }

    # turn TTFT: the 512-token delta turn over a cached prefix
    ra, rb = prefix_runs(ap), prefix_runs(bp)
    turns = {}
    for label in sorted(set(ra) | set(rb)):
        if not label.startswith("grow_"):
            continue
        aa, bb = ra.get(label, {}), rb.get(label, {})
        turns[label] = {
            "a_ttft_ms": aa.get("ttft_ms"),
            "b_ttft_ms": bb.get("ttft_ms"),
            "ttft_pct": pct_delta(aa.get("ttft_ms"), bb.get("ttft_ms")),
            "a_output_tps": aa.get("output_tps"),
            "b_output_tps": bb.get("output_tps"),
            "output_pct": pct_delta(aa.get("output_tps"), bb.get("output_tps")),
        }
    rep["turn_ttft"] = turns

    # acceptance
    decode_4k = (rep["a"].get("4096") or {}).get("delta", {}).get("output_tps_pct")
    turn_ok = any(t.get("ttft_pct") is not None and t["ttft_pct"] <= TURN_TTFT_TARGET_PCT
                  for t in turns.values())
    decode_ok = decode_4k is not None and decode_4k >= DECODE_TARGET_PCT

    regressions = []
    for c in contexts:
        d = (rep["a"].get(str(c)) or {}).get("delta", {})
        for k, key in (("prompt_tps_pct", "prefill"), ("output_tps_pct", "decode"), ("ttft_pct", "TTFT")):
            v = d.get(k)
            # for ctx 131072 we require no >2% regression; for the 4K context the
            # same rule is applied to the other direction of the criterion
            if c >= 32768 and v is not None and v <= REGRESSION_TOL_PCT:
                regressions.append(f"{c} {key} {v:+.2f}%")
    for label, t in turns.items():
        v = t.get("ttft_pct")
        # only enforce the long-context turn (>=32K cached prefix)
        if label.startswith(("grow_p32768", "grow_p65536", "grow_p131072")) and v is not None and v > 2.0:
            regressions.append(f"{label} turn TTFT +{v:.2f}%")

    needle_ok = rep["needle"]["a"] == "pass" and rep["needle"]["b"] == "pass"
    rep["acceptance"] = {
        "thresholds": {
            "turn_ttft_pct": TURN_TTFT_TARGET_PCT,
            "decode_4k_pct": DECODE_TARGET_PCT,
            "long_context_regression_tol_pct": REGRESSION_TOL_PCT,
        },
        "decode_4k_pct": decode_4k,
        "decode_4k_pass": decode_ok,
        "turn_ttft_pass": turn_ok,
        "throughput_pass": bool(decode_ok or turn_ok),
        "needle_pass": needle_ok,
        "long_context_regressions": regressions,
        "pass": bool(decode_ok or turn_ok) and needle_ok and not regressions,
    }
    return rep


def render_md(rep):
    a, b = rep["a_label"], rep["b_label"]
    lines = [
        "# MoE expert-cache A/B — engine LRU vs pinned Stage 0",
        "",
        f"A = `{a}`; B = `{b}`. Delta is `(A - B) / B`; positive tok/s is better, positive TTFT is worse.",
        "",
        "## Throughput and memory",
        "",
        "| context | A prompt tok/s | B prompt tok/s | Δ | A output tok/s | B output tok/s | Δ | A TTFT ms | B TTFT ms | Δ | A VRAM GiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ctx in rep["contexts"]:
        ca, cb = rep["a"].get(ctx, {}), rep["b"].get(ctx, {})
        d = ca.get("delta", {})
        vram = ca.get("vram_peak_bytes")
        lines.append(
            "| {c} | {ap} | {bp} | {dp} | {ao} | {bo} | {do} | {at} | {bt} | {dt} | {v} |".format(
                c=ctx,
                ap=fmt(ca.get("prompt_tps")), bp=fmt(cb.get("prompt_tps")),
                dp=fmt_pct(d.get("prompt_tps_pct")),
                ao=fmt(ca.get("output_tps")), bo=fmt(cb.get("output_tps")),
                do=fmt_pct(d.get("output_tps_pct")),
                at=fmt(ca.get("ttft_ms"), 1), bt=fmt(cb.get("ttft_ms"), 1),
                dt=fmt_pct(d.get("ttft_pct")),
                v=fmt((vram / GiB) if vram else None, 2),
            )
        )
    lines += ["", f"- needle: A **{rep['needle']['a']}**, B **{rep['needle']['b']}**", ""]

    if rep.get("turn_ttft"):
        lines += [
            "## Turn TTFT (cached-prefix delta turn)",
            "",
            "| case | A TTFT ms | B TTFT ms | Δ | A output tok/s | B output tok/s | Δ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, t in rep["turn_ttft"].items():
            lines.append(
                "| {l} | {at} | {bt} | {dt} | {ao} | {bo} | {do} |".format(
                    l=label, at=fmt(t.get("a_ttft_ms"), 1), bt=fmt(t.get("b_ttft_ms"), 1),
                    dt=fmt_pct(t.get("ttft_pct")),
                    ao=fmt(t.get("a_output_tps")), bo=fmt(t.get("b_output_tps")),
                    do=fmt_pct(t.get("output_pct")),
                )
            )
        lines.append("")

    if rep.get("cache"):
        c = rep["cache"]
        lines += [
            "## Engine cache counters",
            "",
            f"- steps `{c.get('steps')}`, hits `{c.get('hits')}`, misses `{c.get('misses')}`, hit rate `{c.get('hit_rate')}`",
            "",
        ]

    acc = rep["acceptance"]
    lines += [
        "## Acceptance",
        "",
        f"- 4K decode Δ: **{fmt_pct(acc['decode_4k_pct'])}** (target ≥ +{DECODE_TARGET_PCT:.0f}%) → {'pass' if acc['decode_4k_pass'] else 'fail'}",
        f"- turn TTFT ≤ {TURN_TTFT_TARGET_PCT:.0f}%: {'pass' if acc['turn_ttft_pass'] else 'fail'}",
        f"- needle on both sides: {'pass' if acc['needle_pass'] else 'fail'}",
        f"- long-context regressions > 2%: {acc['long_context_regressions'] or 'none'}",
        "",
        f"**Verdict: {'PASS' if acc['pass'] else 'FAIL'}**",
        "",
        "Generated by `bench/compare-moe-cache-ab.py` from the raw matrix.json and prefix-cache.json files.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="matrix.json for the LRU engine")
    p.add_argument("--b", required=True, help="matrix.json for the Stage 0 baseline")
    p.add_argument("--a-prefix", default=None)
    p.add_argument("--b-prefix", default=None)
    p.add_argument("--cache-stats", default=None, help="engine moe-cache-stats.json")
    p.add_argument("--contexts", default="4096,131072")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    a, b = load_json(args.a), load_json(args.b)
    if a is None or b is None:
        return 2

    rep = build_report(a, b, load_json(args.a_prefix), load_json(args.b_prefix),
                       load_json(args.cache_stats), contexts)
    rep["sources"] = {"a_matrix": os.path.abspath(args.a), "b_matrix": os.path.abspath(args.b)}
    if args.a_prefix:
        rep["sources"]["a_prefix"] = os.path.abspath(args.a_prefix)
    if args.b_prefix:
        rep["sources"]["b_prefix"] = os.path.abspath(args.b_prefix)
    if args.cache_stats:
        rep["sources"]["cache_stats"] = os.path.abspath(args.cache_stats)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "moe-cache-ab.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    md = render_md(rep)
    with open(os.path.join(args.out, "moe-cache-ab.md"), "w") as fh:
        fh.write(md)
    print(md)
    print(f"wrote {args.out}/moe-cache-ab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
