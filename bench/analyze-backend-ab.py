#!/usr/bin/env python3
"""Summarise the M3.0 backend A/B (BAS-72) and apply the decision rule.

Reads ``bench/results/2026-09-28-backend-ab/<backend>/matrix.json``,
``.../<backend>/prefix-cache/prefix-cache.json`` and
``.../<backend>/bongo-config.json`` and writes two artefacts:

``summary.md``
    the human-readable result table and the decision;
``adr-amendment.md``
    the block to paste into ADR-0002, carrying the engine revision, tier,
    flags and raw file for every number so the row is reproducible.

Metrics per backend:

  * 4K and 128K prefill tok/s (median of the repeats),
  * 128K decode tok/s,
  * 128K cold TTFT (median), and
  * the 512-token cached-turn TTFT from the prefix-cache run: one number per
    warmed prefix length plus the steady-state (``repeat``) cached turn.

Decision rule, verbatim from the ticket:

    SYCL becomes the default only if it holds >= 1.3x on 128K TTFT and stays
    within ~10% on 128K decode; otherwise Vulkan stays the default.

Because the product metric is the *agentic* turn under prefix reuse, the
1.3x gate is applied to both the 128K cold TTFT and the 512-token cached-turn
TTFT, and it is all-or-nothing: failing any gate leaves Vulkan as the default.

    python3 bench/analyze-backend-ab.py \\
        --dir bench/results/2026-09-28-backend-ab
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

TTFT_RATIO_MIN = 1.3
DECODE_TOLERANCE = 1.10
BACKENDS = ("vulkan", "sycl")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def maybe_load(path):
    return load(path) if os.path.isfile(path) else None


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def _identity_mismatch(row):
    """Text describing a backend-identity problem, or None if the build agrees.

    Guards the whole A/B: measuring the Vulkan build twice would 'confirm' SYCL.
    """
    if not row:
        return None
    ev = row.get("evidence") or {}
    if ev.get("error"):
        return f"could not inspect {ev.get('binary')}: {ev['error']}"
    want = row["backend"]
    if want == "sycl" and not ev.get("is_sycl_build"):
        return f"expected a SYCL build (libggml-sycl.so beside {ev.get('binary')})"
    if want == "vulkan" and not ev.get("is_vulkan_build"):
        return f"expected a Vulkan build (libggml-vulkan.so beside {ev.get('binary')})"
    other = "vulkan" if want == "sycl" else "sycl"
    if ev.get(f"is_{other}_build"):
        return f"the {other} backend library is also present; the A/B is not isolated"
    return None


def prefix_len(key):
    """Sort key for a cached-turn key like ``p31744_grow``."""
    m = re.match(r"p(\d+)", key)
    return int(m.group(1)) if m else -1


def prefix_tokens(key):
    m = re.match(r"p(\d+)", key)
    return int(m.group(1)) if m else 0


def context_entry(matrix, target):
    for entry in matrix.get("results") or []:
        if entry.get("target_context") == target:
            return entry
    return None


def peak_vram_gib(memory):
    """Peak VRAM in GiB from a harness ``memory`` block, whatever it calls it."""
    if not isinstance(memory, dict):
        return None
    vram = memory.get("peak_vram_gib") or memory.get("vram_peak_gib")
    if vram is None and memory.get("vram_peak_bytes"):
        vram = memory["vram_peak_bytes"] / (1024 ** 3)
    return vram


def backend_evidence(binary, declared):
    """Evidence that the binary under test really is the backend it claims.

    The `--llama-bin` directory is named after the backend, so a mis-pointed
    path would silently measure one backend twice and quietly "confirm" SYCL.
    ggml keeps its backend in a sibling ``libggml-<backend>.so*``, so that file
    is the check; the ldd view is recorded beside it as corroboration.
    """
    ev = {"binary": binary, "declared_backend": declared}
    if not binary or not os.path.isfile(binary):
        ev["error"] = "binary not found"
        return ev
    bindir = os.path.dirname(binary)
    present = []
    try:
        names = os.listdir(bindir)
    except OSError as exc:
        ev["error"] = str(exc)
        return ev
    for backend in BACKENDS:
        matches = sorted(
            n for n in names if n.startswith(f"libggml-{backend}.so")
        )
        if matches:
            present.append(f"{backend}:{matches[0]}")
    ev["backend_libs"] = present
    ev["is_sycl_build"] = any(p.startswith("sycl:") for p in present)
    ev["is_vulkan_build"] = any(p.startswith("vulkan:") for p in present)
    try:
        import subprocess

        out = subprocess.run(
            ["ldd", binary], capture_output=True, text=True, timeout=30
        ).stdout
        ev["linked"] = sorted(
            os.path.basename(line.split("=>")[0].strip())
            for line in out.splitlines()
            if "=>" in line and re.search(r"lib(ggml|sycl|vulkan)", line)
        )
        ev["unresolved"] = sum(1 for line in out.splitlines() if "not found" in line)
    except (OSError, subprocess.SubprocessError):
        ev["linked"] = None
    return ev


def backend_rows(d, backend, targets):
    matrix_path = os.path.join(d, backend, "matrix.json")
    if not os.path.isfile(matrix_path):
        return None
    matrix = load(matrix_path)
    rows = {"backend": backend, "generated_at": matrix.get("generated_at")}

    bongo_cfg = maybe_load(os.path.join(d, backend, "bongo-config.json"))
    llama = (bongo_cfg or {}).get("llama_cpp") or {}
    rows["revision"] = llama.get("revision")
    rows["commit"] = llama.get("commit")
    rows["declared_backend"] = llama.get("backend")
    rows["binary"] = llama.get("binary")
    rows["evidence"] = backend_evidence(rows["binary"], rows["declared_backend"])
    rows["tier"] = matrix.get("tier") or ((bongo_cfg or {}).get("model") or {}).get("tier")
    rows["gpu"] = ((bongo_cfg or {}).get("gpu") or {}).get("name")
    rows["runtime"] = ((bongo_cfg or {}).get("runtime") or {}).get("dir")
    rows["flags"] = ((bongo_cfg or {}).get("server") or {}).get("flags")

    cfg = matrix.get("config") or {}
    rows["n_ctx"] = cfg.get("context_limit")
    rows["repeats"] = cfg.get("repeats")
    rows["cache_prompt"] = cfg.get("cache_prompt")
    rows["needle_context"] = cfg.get("needle_context")
    rows["needle"] = (matrix.get("needle") or {}).get("status")
    rows["harness_argv"] = (matrix.get("harness") or {}).get("argv")

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
        rows[f"{key}_status"] = entry.get("status")
        rows[f"{key}_ttft_cv"] = _cv([r.get("ttft_ms") for r in runs])
        vram = peak_vram_gib(entry.get("memory"))
        if vram is not None:
            rows[f"{key}_peak_vram_gib"] = vram

    pc_path = os.path.join(d, backend, "prefix-cache", "prefix-cache.json")
    pc = maybe_load(pc_path)
    if pc:
        rows["delta_tokens"] = pc.get("delta_tokens")
        # Keys are "p<prefix>_grow" (a turn that grew the cache by delta) and
        # "p<prefix>_repeat" (the next turn, served from a full cache hit).
        rows["cached_turns"] = {}
        for run in pc.get("runs") or []:
            label = run.get("label") or ""
            m = re.fullmatch(r"grow_p(\d+)_(d\d+|repeat)", label)
            if not m:
                continue
            rows["cached_turns"][f"p{m.group(1)}_{m.group(2)}"] = run.get("ttft_ms")
        grows = [k for k in rows["cached_turns"] if not k.endswith("_repeat")]
        rows["cached_turn_prefixes"] = sorted(grows)
        # Headline cached turn: the longest warmed prefix, i.e. the closest to
        # the shipped 131072 context.  Falls back to the longest available.
        if grows:
            deepest = max(grows, key=prefix_len)
            rows["cached_turn_prefix"] = deepest
            rows["cached_turn_ttft_ms"] = rows["cached_turns"][deepest]
        steady = [k for k in rows["cached_turns"] if k.endswith("_repeat")]
        if steady:
            rows["steady_cached_ttft_ms"] = max(steady, key=prefix_len)

    rows["raw_files"] = sorted(
        os.path.join(backend, "raw", f)
        for f in (os.listdir(os.path.join(d, backend, "raw"))
                  if os.path.isdir(os.path.join(d, backend, "raw")) else [])
    )
    return rows


def _cv(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if not mean:
        return None
    return statistics.pstdev(values) / mean


def fmt(value, spec="{:.2f}", dash="n/a"):
    if value is None:
        return dash
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def flags_line(flags, keep_device=False):
    """`--n-cpu-moe 16 --flash-attn on ...` — the flags that define the profile."""
    if not flags:
        return "n/a"
    # valueless switches, dropped along with the plumbing they imply
    drop_bare = {"--jinja", "--metrics"}
    # flags dropped together with the value that follows
    drop_value = {"--model", "--host", "--port", "--alias", "--slot-save-path", "--parallel"}
    if not keep_device:
        drop_value = drop_value | {"--device"}
    out, skip = [], False
    for tok in flags:
        if skip:
            skip = False
            continue
        if tok in drop_bare:
            continue
        if tok in drop_value:
            skip = True
            continue
        out.append(tok)
    return " ".join(out)


def device_of(row):
    flags = (row or {}).get("flags") or []
    for i, tok in enumerate(flags):
        if tok == "--device" and i + 1 < len(flags):
            return flags[i + 1]
    return (row or {}).get("declared_backend")


def evaluate(v, s, deep):
    """Apply the decision rule.  Returns (passes, list of gate dicts)."""
    gates = []

    def gate(name, need, ok, detail):
        # ok stays None when the input is missing, so the summary can say
        # "not measured" instead of implying SYCL lost the gate.
        gates.append({"name": name, "need": need, "detail": detail, "ok": ok})

    # Precondition: each leg must have run the build it claims to have run.
    for backend, row in (("vulkan", v), ("sycl", s)):
        if not row:
            continue
        bad = _identity_mismatch(row)
        gate(f"{backend} build identity", "matches the leg",
             False if bad else True,
             bad or f"{backend} build carries the expected backend library")

    cold_v, cold_s = (v or {}).get(f"ctx{deep}_ttft_ms"), (s or {}).get(f"ctx{deep}_ttft_ms")
    if cold_v and cold_s:
        gate(f"{deep} cold TTFT", f">= {TTFT_RATIO_MIN}x",
             cold_v / cold_s >= TTFT_RATIO_MIN,
             f"Vulkan {cold_v:.0f} ms vs SYCL {cold_s:.0f} ms = {cold_v / cold_s:.2f}x")
    else:
        gate(f"{deep} cold TTFT", f">= {TTFT_RATIO_MIN}x", None, "missing")

    # The product metric: an agentic turn that reuses a long warmed prefix.
    turn_v = turn_s = None
    if v and s:
        prefixes = [p for p in (s.get("cached_turn_prefixes") or [])
                    if p in (v.get("cached_turn_prefixes") or [])]
        if prefixes:
            deepest = max(prefixes, key=prefix_len)
            turn_v = (v.get("cached_turns") or {}).get(deepest)
            turn_s = (s.get("cached_turns") or {}).get(deepest)
            if turn_v and turn_s:
                gate("512-token cached-turn TTFT", f">= {TTFT_RATIO_MIN}x",
                     turn_v / turn_s >= TTFT_RATIO_MIN,
                     f"Vulkan {turn_v:.0f} ms vs SYCL {turn_s:.0f} ms = {turn_v / turn_s:.2f}x")
            else:
                gate("512-token cached-turn TTFT", f">= {TTFT_RATIO_MIN}x", None,
                     f"missing (prefix {deepest})")
        else:
            gate("512-token cached-turn TTFT", f">= {TTFT_RATIO_MIN}x", None, "missing")
    else:
        gate("512-token cached-turn TTFT", f">= {TTFT_RATIO_MIN}x", None, "no SYCL result")

    dec_v, dec_s = (v or {}).get(f"ctx{deep}_output_tps"), (s or {}).get(f"ctx{deep}_output_tps")
    if dec_v and dec_s:
        gate(f"{deep} decode tok/s", f">= {1 / DECODE_TOLERANCE:.2f}x",
             dec_s / dec_v >= 1 / DECODE_TOLERANCE,
             f"Vulkan {dec_v:.2f} vs SYCL {dec_s:.2f} tok/s = {dec_s / dec_v:.2f}x")
    else:
        gate(f"{deep} decode tok/s", f">= {1 / DECODE_TOLERANCE:.2f}x", None, "missing")

    decided = all(g["ok"] for g in gates) and gates
    return decided, gates


def build_summary(args, rows, v, s, gates, decided):
    L = []
    L.append("# M3.0 backend A/B — SYCL vs Vulkan (agentic profile)")
    L.append("")
    L.append(f"Source: `{args.dir}/` — raw files beside this summary, listed per backend below.")
    L.append("")
    L.append("Profile (shipped): `--n-cpu-moe 16 --flash-attn on --cache-type-k q8_0 "
             "--cache-type-v q8_0`, IQ2_XS, `n_ctx 131072`, one slot, "
             "`bench/harness.py --no-cache-prompt` (cold prefill) x3 plus "
             "`measure-prefix-cache.py --delta 512` for the cached turn.")
    L.append("")

    for name, row in (("vulkan", v), ("sycl", s)):
        if not row:
            continue
        L.append(f"## {name} — run identity")
        L.append("")
        L.append(f"- engine: llama.cpp `{row.get('revision') or 'unknown'}` "
                 f"(`{(row.get('commit') or 'unknown')[:12]}`), build backend "
                 f"`{row.get('declared_backend') or 'unknown'}`")
        ev = row.get("evidence") or {}
        if ev.get("backend_libs"):
            L.append(f"- backend library present: {', '.join(ev['backend_libs'])}")
        if ev.get("linked"):
            L.append(f"- linked: {', '.join(ev['linked'])}")
        if ev.get("error"):
            L.append(f"- **identity check unavailable: {ev['error']}**")
        mism = _identity_mismatch(row)
        if mism:
            L.append(f"- **IDENTITY MISMATCH: {mism}**")
        L.append(f"- binary: `{row.get('binary') or 'unknown'}`")
        L.append(f"- tier: `{row.get('tier') or 'unknown'}`, runtime `{row.get('runtime') or 'unknown'}`, "
                 f"GPU {row.get('gpu') or 'unknown'}")
        L.append(f"- server flags: `{flags_line(row.get('flags'), keep_device=True)}`")
        L.append(f"- n_ctx: {row.get('n_ctx')}, repeats: {row.get('repeats')}, "
                 f"needle at {row.get('needle_context')}: **{row.get('needle')}**")
        L.append(f"- peak VRAM at {args.deep}: "
                 f"{fmt(row.get(f'ctx{args.deep}_peak_vram_gib'))} GiB "
                 f"(at {args.small}: {fmt(row.get(f'ctx{args.small}_peak_vram_gib'))} GiB)")
        if row.get("raw_files"):
            L.append(f"- raw: {', '.join(f'`{f}`' for f in row['raw_files'])}")
        L.append("")

    L.append("## Result table")
    L.append("")
    L.append("| metric | Vulkan | SYCL | SYCL / Vulkan |")
    L.append("| --- | ---: | ---: | ---: |")

    def row(label, getter, spec="{:.2f}", better="lower"):
        vv, sv = getter(v), getter(s)
        ratio = (sv / vv) if (vv and sv) else None
        if better == "lower":
            cell = fmt(ratio, "{:.2f}x") if ratio else "n/a"
        else:
            cell = fmt((vv / sv), "{:.2f}x") if ratio else "n/a"
        L.append(f"| {label} | {fmt(vv, spec)} | {fmt(sv, spec)} | {cell} |")

    row(f"{args.small} prefill tok/s", lambda r: (r or {}).get(f"ctx{args.small}_prompt_tps"),
        better="higher")
    row(f"{args.deep} prefill tok/s", lambda r: (r or {}).get(f"ctx{args.deep}_prompt_tps"),
        better="higher")
    row(f"{args.deep} decode tok/s", lambda r: (r or {}).get(f"ctx{args.deep}_output_tps"),
        better="higher")
    row(f"{args.deep} cold TTFT ms", lambda r: (r or {}).get(f"ctx{args.deep}_ttft_ms"), "{:.0f}")

    keys = {p for r in (v, s) if r for p in (r.get("cached_turns") or {})}
    grows = sorted((k for k in keys if not k.endswith("_repeat")), key=prefix_len)
    steadies = sorted((k for k in keys if k.endswith("_repeat")), key=prefix_len)
    for prefix in grows:
        row(f"512-token cached-turn TTFT ms (prefix {prefix_tokens(prefix)})",
            lambda r, p=prefix: ((r or {}).get("cached_turns") or {}).get(p), "{:.0f}")
    for prefix in steadies:
        row(f"512-token steady cached turn TTFT ms (prefix {prefix_tokens(prefix)})",
            lambda r, p=prefix: ((r or {}).get("cached_turns") or {}).get(p), "{:.0f}")
    row(f"peak VRAM GiB at {args.deep}",
        lambda r: (r or {}).get(f"ctx{args.deep}_peak_vram_gib"))

    L.append("")
    L.append("TTFT ratio is `Vulkan / SYCL` (> 1 means SYCL is faster); tok/s ratios are "
             "`SYCL / Vulkan` (> 1 means SYCL is faster).")
    L.append("")

    L.append("## Decision rule")
    L.append("")
    L.append("> SYCL becomes the default only if it holds >= "
             f"{TTFT_RATIO_MIN}x on {args.deep} TTFT and stays within "
             f"{int((DECODE_TOLERANCE - 1) * 100)}% on {args.deep} decode; "
             "otherwise Vulkan stays the default.")
    L.append("")
    L.append("The ticket's product metric is the agentic turn under prefix reuse, so the "
             f"{TTFT_RATIO_MIN}x gate is applied to the {args.deep} cached-turn TTFT as well as "
             "the cold TTFT, and all gates must pass.")
    L.append("")
    L.append("| gate | required | measured | result |")
    L.append("| --- | --- | --- | --- |")
    for g in gates:
        verdict = "**pass**" if g["ok"] else ("**fail**" if g["ok"] is False else "not measured")
        L.append(f"| {g['name']} | {g['need']} | {g['detail']} | {verdict} |")
    L.append("")

    L.append("## Decision")
    L.append("")
    if not s:
        L.append("**Vulkan stays the default.** No SYCL result was recorded in this run, so the "
                 "SYCL branch of the decision rule cannot be satisfied. See the run log for why.")
    elif decided:
        L.append("**SYCL becomes the default.** Every gate passed:")
        L.append("")
        for g in gates:
            L.append(f"- {g['name']}: {g['detail']} (required {g['need']})")
    else:
        L.append("**Vulkan stays the default.** The SYCL branch of the decision rule was not met:")
        L.append("")
        for g in gates:
            if g["ok"]:
                L.append(f"- {g['name']}: {g['detail']} (required {g['need']}) — passed")
            else:
                L.append(f"- {g['name']}: {g['detail']} (required {g['need']}) — **failed**")
    L.append("")
    return "\n".join(L)


def build_adr_block(args, rows, v, s, gates, decided):
    L = []
    L.append("## Amendment (2026-09-28, [BAS-72](/BAS/issues/BAS-72) — M3.0 backend A/B)")
    L.append("")
    L.append("The default backend is now decided on measurement instead of assumption. Both backends "
             "were run with the shipped Stage-0 profile on the same pinned engine build, on an "
             "otherwise idle Arc Pro B70 (the single-GPU lock from "
             "[BAS-80](/BAS/issues/BAS-80) was held for the whole run):")
    L.append("")
    L.append(f"- engine: llama.cpp `{(v or s or {}).get('revision') or 'unknown'}` "
             f"(`{((v or s or {}).get('commit') or 'unknown')[:12]}`)")
    L.append(f"- tier `{((v or s or {}).get('tier') or 'unknown')}`, "
             f"`n_ctx {((v or s or {}).get('n_ctx') or 'unknown')}`, "
             f"3 repeats, cold prefill (`--no-cache-prompt`) plus a "
             "`--delta 512` cached-turn run")
    L.append(f"- flags: `{flags_line((v or s or {}).get('flags'))}`, device "
             f"`{device_of(v)}` / `{device_of(s)}`")
    L.append(f"- raw: `{args.dir}/<backend>/raw/`")
    L.append("")
    L.append("| metric | Vulkan1 | SYCL0 | SYCL / Vulkan |")
    L.append("| --- | ---: | ---: | ---: |")

    def row(label, getter, spec="{:.2f}", invert=False):
        vv, sv = getter(v), getter(s)
        if not (vv and sv):
            L.append(f"| {label} | {fmt(vv, spec)} | {fmt(sv, spec)} | n/a |")
            return
        L.append(f"| {label} | {fmt(vv, spec)} | {fmt(sv, spec)} | "
                 f"{fmt((vv / sv) if invert else (sv / vv), '{:.2f}x')} |")

    row(f"{args.small} prefill tok/s", lambda r: (r or {}).get(f"ctx{args.small}_prompt_tps"))
    row(f"{args.deep} prefill tok/s", lambda r: (r or {}).get(f"ctx{args.deep}_prompt_tps"))
    row(f"{args.deep} decode tok/s", lambda r: (r or {}).get(f"ctx{args.deep}_output_tps"))
    row(f"{args.deep} cold TTFT ms", lambda r: (r or {}).get(f"ctx{args.deep}_ttft_ms"),
        "{:.0f}", invert=True)
    turn = "cached_turn_ttft_ms"
    row(f"512-token cached-turn TTFT ms (prefix "
        f"{prefix_tokens((v or s or {}).get('cached_turn_prefix') or '')})",
        lambda r: (r or {}).get(turn), "{:.0f}", invert=True)
    row(f"peak VRAM GiB at {args.deep}",
        lambda r: (r or {}).get(f"ctx{args.deep}_peak_vram_gib"))
    L.append("")
    if not s:
        L.append("**SYCL does not become the default; Vulkan stays the default.** No SYCL measurement "
                 "completed in this run, so the rule below cannot be satisfied in SYCL's favour.")
    elif decided:
        L.append("**SYCL becomes the default** — every gate passed:")
    else:
        L.append("**SYCL does not become the default; Vulkan stays the default** — the rule "
                 f"(SYCL holds >= {TTFT_RATIO_MIN}x on {args.deep} TTFT, cold and cached-turn, and "
                 f"stays within {int((DECODE_TOLERANCE - 1) * 100)}% on {args.deep} decode) "
                 "was not met:")
    L.append("")
    for g in gates:
        verdict = "pass" if g["ok"] else ("**fail**" if g["ok"] is False else "not measured")
        L.append(f"- {g['name']}: {g['detail']} (required {g['need']}) — {verdict}")
    L.append("")
    return "\n".join(L)


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
    for backend in BACKENDS:
        row = backend_rows(args.dir, backend, targets)
        if row:
            rows[backend] = row
    if not rows:
        print(f"no backend results under {args.dir}", file=sys.stderr)
        return 2

    v, s = rows.get("vulkan"), rows.get("sycl")
    decided, gates = evaluate(v, s, args.deep)

    text = build_summary(args, rows, v, s, gates, decided)
    out = args.out or os.path.join(args.dir, "summary.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"wrote {out}")

    adr = os.path.join(args.dir, "adr-amendment.md")
    with open(adr, "w") as fh:
        fh.write(build_adr_block(args, rows, v, s, gates, decided) + "\n")
    print(f"wrote {adr}")
    print(f"decision: {'SYCL' if decided else 'Vulkan'} default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
