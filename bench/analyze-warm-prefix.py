#!/usr/bin/env python3
"""Summarise and rank the BAS-130 warm-prefix profile results.

Reads every ``<label>/profile.json`` under a results root (written by
``bench/run-warm-prefix-profile.sh``) and prints/serialises:

* the baseline warm delta-turn table per prefix (hit overhead, grow prefill,
  marginal ms/new-token, decode tok/s);
* the ablation table at a chosen prefix and delta: how much the delta-turn
  prefill moves when one component is taken off the GPU, ranked by magnitude.

The ranking is a critical-path probe, not a from-first-principles cost model:
an ablation that moves a component from the fast GPU to the slow CPU changes
the turn time by the amount that component contributes to the turn.  A large
positive change names a component that matters; a change near zero names one
that is hidden behind other work.  The raw per-op Vulkan timings (parsed
separately by ``bench/parse-vk-perf.py``) name the GPU op classes directly.

Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def load_profiles(root):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "profile.json"))):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        label = data.get("label") or os.path.basename(os.path.dirname(path))
        data["_path"] = path
        out[label] = data
    return out


def runs_by_label(point):
    return {r.get("label"): r for r in point.get("runs", [])}


def get_run(point, prefix, kind_like):
    for r in point.get("runs", []):
        if kind_like in (r.get("label") or ""):
            return r
    return None


def prefix_points(profile, prefix_target):
    for p in profile.get("points", []):
        if p.get("prefix_target") == prefix_target:
            return p
    return None


def marginal_ms_per_token(point, delta):
    by = runs_by_label(point)
    hit = by.get(f"hit_p{point['prefix_tokens']}")
    grow = by.get(f"grow_p{point['prefix_tokens']}_d{delta}")
    if not hit or not grow:
        return None, hit, grow
    gpn = grow.get("prompt_tokens")
    hpn = hit.get("prompt_tokens")
    gms = grow.get("prompt_ms")
    hms = hit.get("prompt_ms")
    if None in (gpn, hpn, gms, hms) or (gpn - hpn) <= 0:
        return None, hit, grow
    return (gms - hms) / (gpn - hpn), hit, grow


def decode_tps(point):
    by = runs_by_label(point)
    for label, rec in by.items():
        if label.startswith("decode_") and rec.get("output_tps"):
            return rec.get("output_tps"), rec.get("output_tokens"), rec.get("output_ms")
    return None, None, None


def fmt(v, d=1):
    return "n/a" if v is None else f"{v:.{d}f}"


def md_baseline(profiles, baseline, prefixes):
    p = profiles.get(baseline)
    lines = []
    lines.append(f"### {baseline} — warm delta-turn cost by prefix")
    lines.append("")
    lines.append("| prefix | hit (fixed) ms | delta 128 ms | delta 512 ms | delta 1024 ms | 512 marginal ms/tok | 512 cache_n | decode tok/s |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pref in prefixes:
        point = prefix_points(p, pref) if p else None
        if not point:
            continue
        by = runs_by_label(point)
        hit = by.get(f"hit_p{point['prefix_tokens']}")
        row = [
            str(pref),
            fmt(hit.get("prompt_ms") if hit else None),
        ]
        for d in (128, 512, 1024):
            g = by.get(f"grow_p{point['prefix_tokens']}_d{d}")
            row.append(fmt(g.get("prompt_ms") if g else None))
        slope, _, grow512 = marginal_ms_per_token(point, 512)
        row.append(fmt(slope, 2))
        row.append(str(grow512.get("cache_n") if grow512 else "n/a"))
        tps, _, _ = decode_tps(point)
        row.append(fmt(tps, 1))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def md_ablations(profiles, baseline, prefix_target, delta):
    base = profiles.get(baseline)
    base_point = prefix_points(base, prefix_target) if base else None
    if not base_point:
        return ["_no baseline point for this prefix_", ""]
    _, _, base_grow = marginal_ms_per_token(base_point, delta)
    if not base_grow:
        return ["_no baseline grow for this delta_", ""]
    base_ms = base_grow.get("prompt_ms")
    rows = []
    for label, prof in profiles.items():
        if label == baseline or not prof.get("points"):
            continue
        point = prefix_points(prof, prefix_target)
        if not point:
            continue
        _, _, grow = marginal_ms_per_token(point, delta)
        if not grow or grow.get("prompt_ms") is None:
            continue
        dms = grow["prompt_ms"] - base_ms
        dpct = 100.0 * dms / base_ms if base_ms else None
        rows.append(
            {
                "label": label,
                "flags_note": prof.get("flags_note", ""),
                "delta512_ms": grow["prompt_ms"],
                "dms": dms,
                "dpct": dpct,
                "cache_n": grow.get("cache_n"),
            }
        )
    rows.sort(key=lambda r: abs(r["dms"]), reverse=True)
    lines = []
    lines.append(f"### Ablations at prefix {prefix_target}, delta {delta} (vs `{baseline}`)")
    lines.append("")
    lines.append(f"Baseline delta-turn prefill: **{fmt(base_ms)} ms**. A positive change means taking that component off the GPU made the turn slower (the component contributes to the critical path).")
    lines.append("")
    lines.append("| rank | config | Δ vs baseline ms | Δ % | delta-turn ms | note |")
    lines.append("| ---: | --- | ---: | ---: | ---: | --- |")
    for i, r in enumerate(rows, 1):
        note = (r["flags_note"] or "").split("note=")[-1]
        lines.append(
            f"| {i} | `{r['label']}` | {r['dms']:+.1f} | {r['dpct']:+.1f}% | {fmt(r['delta512_ms'])} | {note} |"
        )
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="bench/results/2026-09-28-warm-prefix-profile")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--prefix", type=int, default=16384)
    ap.add_argument("--delta", type=int, default=512)
    ap.add_argument("--out", default=None, help="write markdown here (default stdout)")
    ap.add_argument("--json-out", default=None, help="write a machine-readable summary here")
    args = ap.parse_args()

    profiles = load_profiles(args.root)
    prefixes = []
    base = profiles.get(args.baseline, {})
    for p in base.get("points", []):
        prefixes.append(p.get("prefix_target"))
    if not prefixes:
        for prof in profiles.values():
            for p in prof.get("points", []):
                if p.get("prefix_target") not in prefixes:
                    prefixes.append(p.get("prefix_target"))
    prefixes = [p for p in prefixes if p]

    md = []
    md.append("## Warm-prefix profile summary")
    md.append("")
    md.append(f"Configs found: {', '.join(sorted(profiles)) or 'none'}")
    md.append("")
    md += md_baseline(profiles, args.baseline, prefixes)
    md += md_ablations(profiles, args.baseline, args.prefix, args.delta)

    out = "\n".join(md) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)

    if args.json_out:
        summary = {"baseline": args.baseline, "prefixes": prefixes, "configs": sorted(profiles)}
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
