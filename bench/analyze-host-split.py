#!/usr/bin/env python3
"""Render the M4.1 host/CPU decomposition from ``bench/profile-host-split.py``.

Reads ``<label>/profile-host-split.json`` under a results root and prints /
serialises, for the measured delta turn:

  * the cpu-time split of the turn (CPU backend worker threads vs main/host
    thread vs other threads), in ms and as a % of the turn wall;
  * the storage read and page-fault delta of the turn -- the host-resident MoE
    expert weights are mmap'd from the GGUF, so this names "re-read from the
    page cache / SSD" if it is present;
  * the residual "not on-CPU" time (GPU execution + fence wait + I/O wait);
  * the same rows for the ``hit`` case, so the fixed per-request floor can be
    subtracted from the marginal delta-turn cost.

Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def load(root):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "profile-host-split.json"))):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        label = data.get("label") or os.path.basename(os.path.dirname(path))
        data["_path"] = path
        out[label] = data
    return out


def cases_by_label(point):
    return {c.get("label"): c for c in point.get("cases", [])}


def fmt(v, d=1):
    return "n/a" if v is None else f"{v:.{d}f}"


def gib(n):
    if n is None:
        return "n/a"
    return f"{n / (1024 ** 3):.2f}"


def decompose(case):
    """Named components of one measured request, in ms/% of the turn wall."""
    proc = case.get("proc") or {}
    wall = case.get("client_wall_ms") or proc.get("wall_ms")
    cpu_main = proc.get("cpu_main_ms")
    cpu_workers = proc.get("cpu_workers_ms")
    cpu_other = proc.get("cpu_other_ms")
    cpu_total = proc.get("cpu_total_ms")
    read_bytes = (proc.get("io") or {}).get("read_bytes")
    not_cpu = None
    if wall is not None and cpu_total is not None:
        not_cpu = wall - cpu_total
    pct = lambda v: (100.0 * v / wall) if (v is not None and wall) else None
    return {
        "label": case.get("label"),
        "wall_ms": wall,
        "prompt_ms": case.get("prompt_ms"),
        "cpu_workers_ms": cpu_workers,
        "cpu_workers_pct": pct(cpu_workers),
        "cpu_main_ms": cpu_main,
        "cpu_main_pct": pct(cpu_main),
        "cpu_other_ms": cpu_other,
        "cpu_other_pct": pct(cpu_other),
        "cpu_total_ms": cpu_total,
        "cpu_total_pct": pct(cpu_total),
        "not_on_cpu_ms": not_cpu,
        "not_on_cpu_pct": pct(not_cpu),
        "read_bytes": read_bytes,
        "read_gib": (read_bytes / (1024 ** 3)) if isinstance(read_bytes, (int, float)) else None,
        "majflt": proc.get("majflt"),
        "minflt": proc.get("minflt"),
        "prompt_tokens": case.get("prompt_tokens"),
        "cache_n": case.get("cache_n"),
    }


def md_decomposition(profiles, baseline, prefix):
    p = profiles.get(baseline)
    if not p:
        return ["_no baseline profile_", ""]
    point = None
    for pt in p.get("points", []):
        if pt.get("prefix_target") == prefix:
            point = pt
            break
    if not point:
        return [f"_no prefix {prefix} in {baseline}_", ""]
    by = cases_by_label(point)
    delta = p.get("deltas", [512])[0]
    grow_label = f"grow_p{point['prefix_tokens']}_d{delta}"
    hit_label = f"hit_p{point['prefix_tokens']}"
    rows = [decompose(by[grow_label])] if grow_label in by else []
    hit = decompose(by[hit_label]) if hit_label in by else None
    if not rows:
        return ["_no grow case_", ""]

    md = []
    md.append(f"### Host/CPU decomposition — `{baseline}`, prefix {prefix}, delta {delta}")
    md.append("")
    md.append("| case | wall ms | prompt_ms | CPU workers ms | % | main host ms | % | other CPU ms | % | not-on-CPU ms | % | storage read | majflt |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in (rows + ([hit] if hit else [])):
        md.append(
            "| {label} | {wall} | {prompt} | {w} | {wp} | {m} | {mp} | {o} | {op} | {b} | {bp} | {r} GiB | {mf} |".format(
                label=d["label"],
                wall=fmt(d["wall_ms"]),
                prompt=fmt(d["prompt_ms"]),
                w=fmt(d["cpu_workers_ms"]),
                wp=fmt(d["cpu_workers_pct"]),
                m=fmt(d["cpu_main_ms"]),
                mp=fmt(d["cpu_main_pct"]),
                o=fmt(d["cpu_other_ms"]),
                op=fmt(d["cpu_other_pct"]),
                b=fmt(d["not_on_cpu_ms"]),
                bp=fmt(d["not_on_cpu_pct"]),
                r=gib(d["read_bytes"]),
                mf=d["majflt"],
            )
        )
    md.append("")
    md.append(
        "`not-on-CPU` = turn wall − sum of all threads' `/proc` CPU time; it holds GPU "
        "execution, fence/sync waits and storage I/O wait. Cross-check with the "
        "`GGML_VK_PERF_LOGGER` GPU-busy number for the same case."
    )
    md.append("")
    return md


def md_lever_table(profiles, baseline, prefix, delta):
    """Before/after of each config in the root: same point, same delta."""
    p = profiles.get(baseline)
    if not p:
        return []
    base_point = next((pt for pt in p.get("points", []) if pt.get("prefix_target") == prefix), None)
    if not base_point:
        return []
    base_by = cases_by_label(base_point)
    base_case = base_by.get(f"grow_p{base_point['prefix_tokens']}_d{delta}")
    if not base_case:
        return []
    base_d = decompose(base_case)
    md = []
    md.append(f"### Levers at prefix {prefix}, delta {delta} (vs `{baseline}`)")
    md.append("")
    md.append("| config | prompt_ms | Δ ms | Δ % | CPU workers ms | CPU main ms | read GiB | not-on-CPU ms |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, prof in sorted(profiles.items()):
        pt = next((x for x in prof.get("points", []) if x.get("prefix_target") == prefix), None)
        if not pt:
            continue
        case = cases_by_label(pt).get(f"grow_p{pt['prefix_tokens']}_d{delta}")
        if not case:
            continue
        d = decompose(case)
        dms = None
        dpct = None
        if base_d["prompt_ms"] and d["prompt_ms"]:
            dms = d["prompt_ms"] - base_d["prompt_ms"]
            dpct = 100.0 * dms / base_d["prompt_ms"]
        md.append(
            "| {l} | {p} | {dms} | {dp} | {w} | {m} | {r} | {b} |".format(
                l=f"`{label}`",
                p=fmt(d["prompt_ms"]),
                dms=("n/a" if dms is None else f"{dms:+.1f}"),
                dp=("n/a" if dpct is None else f"{dpct:+.1f}"),
                w=fmt(d["cpu_workers_ms"]),
                m=fmt(d["cpu_main_ms"]),
                r=gib(d["read_bytes"]),
                b=fmt(d["not_on_cpu_ms"]),
            )
        )
    md.append("")
    return md


def md_threads(profiles, baseline, prefix, delta, top=14):
    p = profiles.get(baseline)
    if not p:
        return []
    point = next((pt for pt in p.get("points", []) if pt.get("prefix_target") == prefix), None)
    if not point:
        return []
    by = cases_by_label(point)
    grow = by.get(f"grow_p{point['prefix_tokens']}_d{delta}")
    if not grow:
        return []
    threads = sorted(grow["proc"]["threads"], key=lambda t: t["cpu_ms"], reverse=True)[:top]
    md = ["", f"#### `{baseline}` delta-turn per-thread CPU (top {top})", ""]
    md.append("| tid | comm | role | CPU ms | majflt |")
    md.append("| ---: | --- | --- | ---: | ---: |")
    for t in threads:
        md.append(f"| {t['tid']} | `{t['comm']}` | {t.get('role')} | {t['cpu_ms']} | {t['majflt']} |")
    md.append("")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="bench/results/2026-09-29-host-cpu")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--prefix", type=int, default=16384)
    ap.add_argument("--delta", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    profiles = load(args.root)
    md = ["## Host/CPU decomposition of the cached delta turn", "",
          f"Configs found: {', '.join(sorted(profiles)) or 'none'}", ""]
    md += md_decomposition(profiles, args.baseline, args.prefix)
    md += md_lever_table(profiles, args.baseline, args.prefix, args.delta)
    md += md_threads(profiles, args.baseline, args.prefix, args.delta)
    out = "\n".join(md) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(out)
    print(out)
    if args.json_out:
        summary = {label: [decompose(c) for pt in prof.get("points", []) for c in pt.get("cases", [])]
                   for label, prof in profiles.items()}
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
