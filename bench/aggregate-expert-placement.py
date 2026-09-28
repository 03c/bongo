#!/usr/bin/env python3
"""Aggregate the expert-placement sweep into one matrix.

Reads every ``ncmoe-<N>/matrix.json`` written by ``bench/sweep-expert-placement.sh``
and emits ``sweep-matrix.json`` + ``sweep-matrix.md`` next to them.

For each ``--n-cpu-moe N`` it reports, at 4096 and 131072 context:

* prompt tok/s and output tok/s (from the harness's median summary),
* peak VRAM (harness fdinfo sampler) and VRAM resident after model load,
* peak process RSS and the host free-memory floor,
* expert bytes on the GPU and on the CPU (from the GGUF tensor table),
* whether the configuration loaded and served at all.

Stdlib Python 3 only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

GiB = 1024**3


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def expert_split(expert_bytes, n_cpu_moe):
    total = sum(expert_bytes)
    cpu = sum(expert_bytes[:n_cpu_moe])
    return {"total": total, "cpu": cpu, "gpu": total - cpu, "gpu_gib": (total - cpu) / GiB, "cpu_gib": cpu / GiB}


def ctx_metrics(matrix, ctx):
    for r in matrix.get("results", []):
        if r.get("target_context") != ctx:
            continue
        s = r.get("summary", {})
        mem = r.get("memory", {}) or {}
        out = {
            "status": r.get("status"),
            "prompt_tokens": (s.get("prompt_tokens") or {}).get("median"),
            "prompt_tps": (s.get("prompt_tps") or {}).get("median"),
            "output_tps": (s.get("output_tps") or {}).get("median"),
            "ttft_ms": (s.get("ttft_ms") or {}).get("median"),
            "vram_peak_bytes": mem.get("vram_peak_bytes"),
            "ram_peak_bytes": mem.get("system_ram_peak_bytes"),
        }
        if r.get("status") != "ok":
            out["error"] = (r.get("error") or "")[:300]
        return out
    return None


def collect(root, contexts):
    expert_meta = load_json(os.path.join(root, "expert-bytes-iq2_xs.json")) or {}
    expert_bytes = expert_meta.get("expert_bytes_by_layer") or []
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "ncmoe-*", "matrix.json"))):
        matrix = load_json(path)
        if not matrix:
            continue
        n = matrix.get("n_cpu_moe")
        if n is None:
            n = int(os.path.basename(os.path.dirname(path)).split("ncmoe-")[1])
        dir_ = os.path.dirname(path)
        after_load = load_json(os.path.join(dir_, "placement-after-load.json")) or {}
        io = load_json(os.path.join(dir_, "server-io.json")) or {}
        ctx = {c: ctx_metrics(matrix, c) for c in contexts}
        # A config "fits" only when it loaded AND every measured context served OK.
        loaded = "fatal_error" not in matrix
        ctx_ok = {c: bool(ctx[c] and ctx[c].get("status") == "ok") for c in contexts}
        component_ok = any(ctx[c] is not None for c in contexts)
        fit = loaded and component_ok and all(ctx_ok.values())
        row = {
            "n_cpu_moe": n,
            "loaded": loaded,
            "fit": fit,
            "fit_by_context": ctx_ok,
            "fatal_error": (matrix.get("fatal_error") or {}).get("message") if "fatal_error" in matrix else None,
            "vram_after_load_bytes": after_load.get("vram_resident_bytes_after_load"),
            "server_read_bytes_delta": (
                (io.get("server_read_bytes_after") or 0) - (io.get("server_read_bytes_before") or 0)
                if io.get("server_read_bytes_after") and io.get("server_read_bytes_before")
                else None
            ),
            "contexts": ctx,
        }
        if expert_bytes:
            row["experts"] = expert_split(expert_bytes, n)
        rows.append(row)
    rows.sort(key=lambda r: r["n_cpu_moe"])
    return rows


def fmt(v, nd=3, dash="n/a"):
    return dash if v is None else f"{v:.{nd}f}"


def render_md(rows, contexts):
    lines = []
    lines.append("# Expert placement sweep — Arc Pro B70 / IQ2_XS")
    lines.append("")
    lines.append("`--n-cpu-moe N` keeps the routed experts of the first N layers on the CPU;")
    lines.append("layers N..47 keep their experts on the GPU. One repeat per context.")
    lines.append("")
    lines.append("`fit` means the config loaded **and** every measured context served OK.")
    lines.append("`fit_by_context` shows which contexts passed. A failed context still records")
    lines.append("the peak VRAM reached before the failure.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    header = ["n-cpu-moe", "loaded", "fit", "GPU experts GiB", "CPU experts GiB", "VRAM after load GiB"]
    for ctx in contexts:
        header += [f"{ctx} prompt tok/s", f"{ctx} output tok/s", f"{ctx} VRAM GiB", f"{ctx} RAM GiB"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---:"] * len(header)) + " |")
    for r in rows:
        ex = r.get("experts") or {}
        if not r["loaded"]:
            fit_label = "**no (load OOM)**"
        elif r["fit"]:
            fit_label = "yes"
        else:
            failed = [str(c) for c, ok in (r.get("fit_by_context") or {}).items() if not ok]
            fit_label = "**no (" + ", ".join(failed) + " failed)**"
        cells = [
            str(r["n_cpu_moe"]),
            "yes" if r["loaded"] else "**no**",
            fit_label,
            fmt(ex.get("gpu_gib"), 2),
            fmt(ex.get("cpu_gib"), 2),
            fmt((r.get("vram_after_load_bytes") or 0) / GiB if r.get("vram_after_load_bytes") else None, 2),
        ]
        for ctx in contexts:
            c = r["contexts"].get(ctx) or {}
            cells += [
                fmt(c.get("prompt_tps")),
                fmt(c.get("output_tps")),
                fmt((c.get("vram_peak_bytes") or 0) / GiB if c.get("vram_peak_bytes") else None, 2),
                fmt((c.get("ram_peak_bytes") or 0) / GiB if c.get("ram_peak_bytes") else None, 2),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## CPU-expert share and marginal 128K throughput")
    lines.append("")
    lines.append("Only configs that served 128K are comparable. The marginal column is")
    lines.append("`d(output tok/s at 128K) / d(GPU expert GiB)` between adjacent measured")
    lines.append("configs, i.e. how many output tok/s each extra GiB of GPU-resident experts buys.")
    lines.append("")
    lines.append("| n-cpu-moe | CPU expert share | GPU experts GiB | 128K output tok/s | marginal tok/s per GPU GiB |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    ok_rows = [r for r in rows if r["fit"]]
    for i, r in enumerate(rows):
        ex = r.get("experts") or {}
        cpu_share = (ex.get("cpu_gib") / (ex.get("cpu_gib") + ex.get("gpu_gib"))) if ex else None
        o128 = (r["contexts"].get(131072) or {}).get("output_tps")
        marginal = None
        if r["fit"] and o128:
            for prev in reversed(ok_rows[: ok_rows.index(r)]):
                p_o = (prev["contexts"].get(131072) or {}).get("output_tps")
                p_g = (prev.get("experts") or {}).get("gpu_gib")
                g = ex.get("gpu_gib")
                if p_o and p_g and g and (p_g - g) != 0:
                    marginal = (o128 - p_o) / (p_g - g)
                    break
        lines.append(
            f"| {r['n_cpu_moe']} | {fmt(cpu_share,3) if cpu_share is not None else 'n/a'} | "
            f"{fmt(ex.get('gpu_gib'),2)} | {fmt(o128)} | {fmt(marginal,4)} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="bench/results/2026-09-27-expert-placement")
    p.add_argument("--contexts", default="4096,131072")
    args = p.parse_args(argv)
    contexts = [int(x) for x in args.contexts.split(",") if x]
    rows = collect(args.root, contexts)
    if not rows:
        print("no results found", file=sys.stderr)
        return 1
    out = {"schema": "bongo.expert-placement.sweep.v1", "contexts": contexts, "rows": rows}
    with open(os.path.join(args.root, "sweep-matrix.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    md = render_md(rows, contexts)
    with open(os.path.join(args.root, "sweep-matrix.md"), "w") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
