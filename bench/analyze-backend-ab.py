#!/usr/bin/env python3
"""Summarise the M3.0 backend A/B (BAS-72) and apply the decision rule.

Reads ``bench/results/2026-09-28-backend-ab/<backend>/matrix.json`` and
``.../<backend>/prefix-cache/prefix-cache.json`` and emits ``summary.md``:

  * 4K and 128K prefill tok/s (median of the repeats),
  * 128K decode tok/s,
  * 512-token cached-turn TTFT (the ``grow`` row of the prefix-cache run),
  * peak VRAM and n_ctx,
  * the decision: SYCL becomes the default only if it holds >= 1.3x on 128K TTFT
    and stays within ~10% on 128K decode; otherwise Vulkan stays the default.

    python3 bench/analyze-backend-ab.py \\
        --dir bench/results/2026-09-28-backend-ab
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

TTFT_RATIO_MIN = 1.3
DECODE_TOLERANCE = 1.10


def load(path):
    with open(path) as fh:
        return json.load(fh)


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def context_entry(matrix, target):
    for entry in matrix.get("results") or []:
        if entry.get("target_context") == target:
            return entry
    return None


def backend_rows(d, backend, targets):
    matrix_path = os.path.join(d, backend, "matrix.json")
    if not os.path.isfile(matrix_path):
        return None
    matrix = load(matrix_path)
    rows = {"backend": backend, "generated_at": matrix.get("generated_at")}

    cfg = matrix.get("config") or {}
    rows["n_ctx"] = cfg.get("context_limit")
    rows["command"] = (matrix.get("harness") or {}).get("command")
    rows["needle"] = (matrix.get("needle") or {}).get("status")

    for target in targets:
        entry = context_entry(matrix, target)
        if not entry:
            continue
        runs = entry.get("runs") or []
        key = f"ctx{target}"
        rows[f"{key}_prompt_tps"] = median([r.get("prompt_tps") for r in runs])
        rows[f"{key}_output_tps"] = median([r.get("output_tps") for r in runs])
        rows[f"{key}_ttft_ms"] = median([r.get("ttft_ms") for r in runs])
        rows[f"{key}_prompt_tokens"] = median([r.get("prompt_tokens") for r in runs])
        rows[f"{key}_repeats"] = len(runs)
        mem = entry.get("memory") or {}
        vram = mem.get("peak_vram_gib") or mem.get("vram_peak_gib")
        if vram is None and mem.get("vram_peak_bytes"):
            vram = mem["vram_peak_bytes"] / (1024 ** 3)
        if vram is None:
            top = (matrix.get("memory") or {}).get("vram_peak_bytes")
            if top:
                vram = top / (1024 ** 3)
        if vram is not None:
            rows["peak_vram_gib"] = vram

    pc_path = os.path.join(d, backend, "prefix-cache", "prefix-cache.json")
    if os.path.isfile(pc_path):
        pc = load(pc_path)
        rows["delta_tokens"] = pc.get("delta_tokens")
        for run in pc.get("runs") or []:
            label = run.get("label") or ""
            if label.startswith("grow_p") and "_d" in label:
                rows["cached_turn_ttft_ms"] = run.get("ttft_ms")
                rows["cached_turn_prefix"] = label
            if label.startswith("hit_p"):
                rows.setdefault("hit_ttft_ms", run.get("ttft_ms"))
    return rows


def fmt(value, spec="{:.2f}", dash="n/a"):
    if value is None:
        return dash
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench/results/2026-09-28-backend-ab")
    ap.add_argument("--targets", default="4096,131072")
    ap.add_argument("--small", type=int, default=4096)
    ap.add_argument("--deep", type=int, default=131072)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    targets = [int(x) for x in args.targets.split(",") if x.strip()]
    rows = {}
    for backend in ("vulkan", "sycl"):
        row = backend_rows(args.dir, backend, targets)
        if row:
            rows[backend] = row

    if not rows:
        print(f"no backend results under {args.dir}", file=sys.stderr)
        return 2

    v, s = rows.get("vulkan"), rows.get("sycl")
    lines = []
    lines.append("# M3.0 backend A/B — SYCL vs Vulkan (agentic profile)")
    lines.append("")
    lines.append(f"Source: `{args.dir}/` (raw files beside this summary).")
    lines.append("")

    if v:
        lines.append("## Vulkan (Stage 0 baseline, kept pinned)")
        lines.append("")
        lines.append(f"- engine: llama.cpp `{(v.get('command') or '').split('--contexts')[0].strip()[:0] or 'b11223'}`")
        lines.append(f"- n_ctx: {v.get('n_ctx')}, 128K needle: **{v.get('needle')}**")
        lines.append(f"- peak VRAM: {fmt(v.get('peak_vram_gib'))} GiB")
        lines.append("")

    header = "| metric | Vulkan | SYCL | SYCL / Vulkan |"
    lines.append("## Result table")
    lines.append("")
    lines.append(header)
    lines.append("| --- | ---: | ---: | ---: |")

    def row(label, key, spec="{:.2f}"):
        vv = (v or {}).get(key)
        sv = (s or {}).get(key)
        ratio = (sv / vv) if (vv and sv) else None
        lines.append(
            f"| {label} | {fmt(vv, spec)} | {fmt(sv, spec)} | {fmt(ratio, '{:.2f}x') if ratio else 'n/a'} |"
        )

    row(f"{args.small} prefill tok/s", f"ctx{args.small}_prompt_tps")
    row(f"{args.deep} prefill tok/s", f"ctx{args.deep}_prompt_tps")
    row(f"{args.deep} decode tok/s", f"ctx{args.deep}_output_tps")
    row(f"{args.deep} TTFT ms (cold prefill)", f"ctx{args.deep}_ttft_ms", "{:.0f}")
    row("512-token cached-turn TTFT ms", "cached_turn_ttft_ms", "{:.0f}")
    row("peak VRAM GiB", "peak_vram_gib")

    # ---- decision ----
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if not s:
        lines.append(
            "**Vulkan stays the default.** No SYCL result was recorded in this run, so the SYCL branch of "
            "the decision rule cannot be satisfied. See the run log for why."
        )
    else:
        ttft_v = (v or {}).get(f"ctx{args.deep}_ttft_ms")
        ttft_s = s.get(f"ctx{args.deep}_ttft_ms")
        dec_v = (v or {}).get(f"ctx{args.deep}_output_tps")
        dec_s = s.get(f"ctx{args.deep}_output_tps")
        if not (ttft_v and ttft_s and dec_v and dec_s):
            lines.append("**Inconclusive** — one of the decision inputs is missing; see the table above.")
        else:
            speedup = ttft_v / ttft_s
            decode_ratio = dec_s / dec_v
            ok_ttft = speedup >= TTFT_RATIO_MIN
            ok_decode = decode_ratio >= (1 / DECODE_TOLERANCE)
            if ok_ttft and ok_decode:
                verdict = "**SYCL becomes the default.**"
                detail = (
                    f"SYCL is {speedup:.2f}x on {args.deep} TTFT (>= {TTFT_RATIO_MIN}x) and decode is "
                    f"{decode_ratio:.2f}x of Vulkan (>= {1 / DECODE_TOLERANCE:.2f}x, i.e. within 10%)."
                )
            else:
                verdict = "**Vulkan stays the default.**"
                reasons = []
                if not ok_ttft:
                    reasons.append(
                        f"{args.deep} TTFT speedup {speedup:.2f}x is below the required {TTFT_RATIO_MIN}x"
                    )
                if not ok_decode:
                    reasons.append(
                        f"128K decode is {decode_ratio:.2f}x of Vulkan, outside the "
                        f"{1 / DECODE_TOLERANCE:.2f}x floor"
                    )
                detail = "; ".join(reasons) + "."
            lines.append(verdict)
            lines.append("")
            lines.append(
                f"- 128K TTFT: Vulkan {ttft_v / 1000:.0f} s vs SYCL {ttft_s / 1000:.0f} s "
                f"({speedup:.2f}x)"
            )
            lines.append(
                f"- 128K decode: Vulkan {dec_v:.2f} tok/s vs SYCL {dec_s:.2f} tok/s "
                f"({decode_ratio:.2f}x)"
            )
            lines.append(f"- {detail}")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    out = args.out or os.path.join(args.dir, "summary.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
