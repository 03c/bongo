#!/usr/bin/env python3
"""Measure suffix/n-gram speculative decoding (llama.cpp ``--spec-type ngram-*``).

The published GGUF has no MTP head, so the only decode multiplier available to
bongo is self-speculation: a suffix/n-gram drafter proposes tokens from the
context and the target model verifies the longest greedy prefix in one forward
pass.  The pinned engine (llama.cpp ``b11223``) already ships this machinery
(``--spec-type ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache``),
so this tool *configures and measures* it rather than patching the engine.

What it records, per context length:

* the **equivalence** generation (greedy, ``temperature=0``, ``cache_prompt=false``)
  so two runs -- spec off and spec on -- can be compared byte for byte;
* the **decode** path (streaming, ``cache_prompt=true``) with server timings,
  TTFT, and the draft counters the engine reports;
* the **acceptance** statistics, from ``timings`` when present and otherwise
  parsed from the llama-server log line
  ``draft acceptance = <ratio> (<acc> accepted / <gen> generated), mean len = <n>``.

Two subcommands:

    measure-speculation.py run     --label baseline --out baseline.json [...]
    measure-speculation.py compare --baseline baseline.json --spec spec.json
    measure-speculation.py self-test

``run`` is executed once per server configuration (spec off, then spec on); the
server config lives outside this tool because ``--spec-type`` is a server flag.
``compare`` is pure -- it never contacts a server -- so it works as a unit test
and as the equivalence gate.

Stdlib only; reuses ``bench_lib``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import (  # noqa: E402
    CORPUS,
    Tokenizer,
    get_json,
    now_iso,
    post_json,
    server_root,
    stream_completion,
)

BASE = os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BONGO_MODEL", "bongo-iq2_xs")
SCHEMA_RUN = "bongo.speculation-run.v1"
SCHEMA_CMP = "bongo.speculation-compare.v1"

DRAFT_RE = re.compile(
    r"draft acceptance\s*=\s*(?P<ratio>[0-9.]+)\s*"
    r"\(\s*(?P<acc>\d+)\s+accepted\s*/\s*(?P<gen>\d+)\s+generated\s*\)\s*,?\s*"
    r"mean len\s*=\s*(?P<mean>[0-9.]+)"
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def draft_fields(timings, usage):
    """Pull cache-reuse and speculative counters out of one response.

    llama.cpp reports the accepted/verified counts under several names across
    builds, so fold every plausible key into one record instead of guessing.
    """
    t = timings if isinstance(timings, dict) else {}
    u = usage if isinstance(usage, dict) else {}
    out = {
        "cache_n": t.get("cache_n") if t.get("cache_n") is not None else u.get("prompt_tokens_details", {}).get("cached_tokens"),
        "prompt_n": t.get("prompt_n"),
        "predicted_n": t.get("predicted_n"),
        "draft_n": None,
        "draft_n_accepted": None,
        "draft_n_verif_steps": None,
        "draft_ratio": None,
        "mean_acc_len": None,
    }
    for key in ("draft_n", "n_draft", "draft_n_total", "draft_tokens"):
        if t.get(key) is not None:
            out["draft_n"] = t[key]
            break
    for key in ("draft_n_accepted", "n_draft_accepted", "draft_tokens_accepted"):
        if t.get(key) is not None:
            out["draft_n_accepted"] = t[key]
            break
    for key in ("draft_n_verif_steps", "n_draft_verif_steps", "draft_verif_steps"):
        if t.get(key) is not None:
            out["draft_n_verif_steps"] = t[key]
            break
    for key in ("draft_ratio", "draft_acceptance", "acceptance"):
        if t.get(key) is not None:
            out["draft_ratio"] = t[key]
            break
    for key in ("mean_acc_len", "draft_mean_len", "mean_accepted_len"):
        if t.get(key) is not None:
            out["mean_acc_len"] = t[key]
            break
    if out["draft_ratio"] is None and out["draft_n"] and out["draft_n_accepted"] is not None:
        out["draft_ratio"] = out["draft_n_accepted"] / out["draft_n"]
    return out


def tail_new_lines(path, offset):
    """Return (new_offset, lines added since ``offset``).  Never raises."""
    if not path or not os.path.isfile(path):
        return offset, []
    try:
        with open(path, "r", errors="replace") as fh:
            fh.seek(offset)
            data = fh.read()
            new_offset = fh.tell()
        return new_offset, data.splitlines()
    except OSError:
        return offset, []


def parse_acceptance(lines):
    """Fold ``draft acceptance`` log lines into per-run acceptance numbers."""
    ratio = acc = gen = 0.0
    mean = None
    steps = 0
    for line in lines:
        m = DRAFT_RE.search(line)
        if not m:
            continue
        r = float(m.group("ratio"))
        a = int(m.group("acc"))
        g = int(m.group("gen"))
        ratio += r * g
        acc += a
        gen += g
        steps += 1
        if mean is None:
            mean = float(m.group("mean"))
    if gen <= 0:
        return None
    return {
        "draft_n": int(gen),
        "draft_n_accepted": int(acc),
        "draft_ratio": acc / gen,
        "mean_acc_len": mean,
        "verif_steps": steps,
        "tokens_per_round": (acc + steps) / steps if steps else None,
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def greedy_completion(base_url, model, prompt, max_tokens, timeout, cache_prompt=True):
    """Deterministic greedy completion for the equivalence gate."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": bool(cache_prompt),
        "ignore_eos": True,
    }
    res = post_json(base_url, "/completions", payload, timeout)
    out = {"status": res.status, "text": None, "prompt_tokens": None, "completion_tokens": None}
    if isinstance(res.json, dict):
        t = res.json.get("timings") or {}
        u = res.json.get("usage") or {}
        out["prompt_tokens"] = t.get("prompt_n") or u.get("prompt_tokens")
        out["completion_tokens"] = t.get("predicted_n") or u.get("completion_tokens")
        choices = res.json.get("choices") or []
        if choices and isinstance(choices[0], dict):
            out["text"] = choices[0].get("text") or ""
    else:
        out["error"] = res.body.decode("utf-8", "replace")[:500]
    return out


def stream_measure(base_url, model, prompt, max_tokens, timeout):
    """One streaming decode request; returns timings, TTFT, and draft counters."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "cache_prompt": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    ttft_ms, _ttfb_ms, total_ms, text, meta = stream_completion(base_url, payload, timeout)
    timings = meta.get("timings") or {}
    return {
        "status": meta.get("status"),
        "wall_ms": round(total_ms, 2),
        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tps": timings.get("prompt_per_second"),
        "output_tokens": timings.get("predicted_n"),
        "output_tps": timings.get("predicted_per_second"),
        "timings": timings,
        "draft": draft_fields(timings, meta.get("usage")),
        "text": text,
        "error": meta.get("error"),
    }


def summarize(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    med = vals_sorted[n // 2] if n % 2 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
    return {"n": n, "median": med, "min": min(vals), "max": max(vals)}


def run_measurement(args):
    base_url = args.base_url.rstrip("/")
    tk = Tokenizer(base_url, timeout=args.timeout)
    if not tk.available:
        tk.count("warmup")

    props_res = get_json(server_root(base_url), "/props", args.timeout)
    props = props_res.json if isinstance(props_res.json, dict) else {}

    contexts = [int(x) for x in str(args.contexts).replace(" ", "").split(",") if x]

    # Warm the server once so the first measured request is not the shader compile.
    warm = stream_measure(base_url, args.model, tk.size_to(args.warmup_tokens), 1, args.timeout)
    # Start the log window after warmup so warmup drafts do not pollute acceptance.
    log_offset = 0
    if args.server_log and os.path.isfile(args.server_log):
        log_offset = os.path.getsize(args.server_log)

    out = {
        "schema": SCHEMA_RUN,
        "generated_at": now_iso(),
        "label": args.label,
        "spec_type": args.spec_type,
        "spec_synth": args.spec_synth,
        "base_url": base_url,
        "model": args.model,
        "tier": args.tier,
        "config": {
            "contexts": contexts,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "equivalence_tokens": args.equivalence_tokens,
            "skip_equivalence": bool(args.skip_equivalence),
            "cache_prompt": True,
        },
        "server": {
            "build_info": props.get("build_info") or props.get("build"),
            "default_generation_settings": props.get("default_generation_settings"),
            "log": args.server_log,
            "note": args.server_note,
        },
        "warmup": warm,
        "contexts": [],
    }

    for ctx in contexts:
        entry = {"context": ctx, "equiv": None, "decode": [], "summary": {}, "acceptance": None,
                 "acceptance_runs": []}
        hard_max = max(1, args.context_limit_guard - args.max_tokens if args.context_limit_guard else ctx)
        prompt = tk.size_to(ctx, CORPUS, hard_max=hard_max)
        actual = tk.count(prompt)
        entry["prompt_tokens"] = actual
        print(f"== context {ctx} (actual prompt {actual} tokens) ==", flush=True)

        if not args.skip_equivalence:
            entry["equiv"] = greedy_completion(base_url, args.model, prompt, args.equivalence_tokens, args.timeout)
            eq = entry["equiv"]
            print(f"   equiv status={eq.get('status')} tokens={eq.get('completion_tokens')}", flush=True)

        for rep in range(args.repeats):
            before = log_offset
            rec = stream_measure(base_url, args.model, prompt, args.max_tokens, args.timeout)
            log_offset, lines = tail_new_lines(args.server_log, before)
            rec["repeat"] = rep + 1
            entry["decode"].append(rec)
            print(
                f"   decode r{rep + 1}: status={rec['status']} ttft={rec['ttft_ms']} "
                f"out_tps={rec['output_tps']} cache_n={rec['draft'].get('cache_n')}",
                flush=True,
            )
            acc = parse_acceptance(lines)
            if acc:
                entry["acceptance_runs"].append(acc)

        if entry["acceptance_runs"]:
            gen = sum(a["draft_n"] for a in entry["acceptance_runs"])
            accepted = sum(a["draft_n_accepted"] for a in entry["acceptance_runs"])
            steps = sum(a["verif_steps"] for a in entry["acceptance_runs"])
            entry["acceptance"] = {
                "draft_n": gen,
                "draft_n_accepted": accepted,
                "draft_ratio": (accepted / gen) if gen else None,
                "mean_acc_len": entry["acceptance_runs"][-1].get("mean_acc_len"),
                "verif_steps": steps,
                "tokens_per_round": ((accepted + steps) / steps) if steps else None,
            }

        entry["summary"] = {
            "output_tps": summarize([r.get("output_tps") for r in entry["decode"]]),
            "ttft_ms": summarize([r.get("ttft_ms") for r in entry["decode"]]),
            "output_tokens": summarize([r.get("output_tokens") for r in entry["decode"]]),
            "cache_n": summarize([r["draft"].get("cache_n") for r in entry["decode"]]),
        }
        out["contexts"].append(entry)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}")
    return out


# ---------------------------------------------------------------------------
# compare (pure; no server)
# ---------------------------------------------------------------------------


def _decode_tps(entry):
    s = (entry.get("summary") or {}).get("output_tps")
    return s.get("median") if isinstance(s, dict) else None


def _ttft(entry):
    s = (entry.get("summary") or {}).get("ttft_ms")
    return s.get("median") if isinstance(s, dict) else None


def compare_runs(baseline, spec):
    """Compare a spec-off run and a spec-on run.  Pure function; unit-testable."""
    by_ctx = {e.get("context"): e for e in baseline.get("contexts", [])}
    spec_by_ctx = {e.get("context"): e for e in spec.get("contexts", [])}
    results = []
    for ctx in sorted(set(by_ctx) | set(spec_by_ctx)):
        b = by_ctx.get(ctx)
        s = spec_by_ctx.get(ctx)
        row = {"context": ctx, "status": "ok"}
        if b is None or s is None:
            row.update({"status": "missing", "baseline_present": b is not None, "spec_present": s is not None})
            results.append(row)
            continue
        be, se = b.get("equiv"), s.get("equiv")
        if be is not None and se is not None and be.get("status") == 200 and se.get("status") == 200:
            bt, st = be.get("text") or "", se.get("text") or ""
            row["equiv_equal"] = bt == st
            row["equiv_len"] = len(bt)
            if bt != st:
                diverge = next((i for i, (x, y) in enumerate(zip(bt, st)) if x != y), min(len(bt), len(st)))
                row["equiv_divergence_index"] = diverge
                row["equiv_baseline_excerpt"] = bt[max(0, diverge - 40):diverge + 80]
                row["equiv_spec_excerpt"] = st[max(0, diverge - 40):diverge + 80]
        else:
            row["equiv_equal"] = None
        btps, stps = _decode_tps(b), _decode_tps(s)
        row["output_tps_baseline"] = btps
        row["output_tps_spec"] = stps
        row["speedup"] = (stps / btps) if btps and stps else None
        row["ttft_baseline_ms"] = _ttft(b)
        row["ttft_spec_ms"] = _ttft(s)
        row["acceptance"] = s.get("acceptance") or (s.get("decode") or [{}])[-1].get("draft")
        results.append(row)

    identical = all(r.get("equiv_equal") is not False for r in results)
    regressions = [
        r for r in results
        if r.get("speedup") is not None and r["speedup"] < 1.0 and (r.get("acceptance") or {}).get("draft_ratio", 0) < 0.05
    ]
    ge_half = [
        r for r in results
        if (r.get("acceptance") or {}).get("draft_ratio") is not None
        and (r["acceptance"]["draft_ratio"] >= 0.5)
        and r.get("speedup") is not None
    ]
    return {
        "schema": SCHEMA_CMP,
        "generated_at": now_iso(),
        "baseline_label": baseline.get("label"),
        "spec_label": spec.get("label"),
        "spec_type": spec.get("spec_type"),
        "contexts": results,
        "verdict": {
            "outputs_identical": identical,
            "speedups_at_acceptance_ge_0.5": {str(r["context"]): r["speedup"] for r in ge_half},
            "min_speedup_at_acceptance_ge_0.5": min((r["speedup"] for r in ge_half), default=None),
            "low_acceptance_regressions": [
                {"context": r["context"], "speedup": r["speedup"]} for r in regressions
            ],
        },
    }


def cmd_run(args):
    run_measurement(args)
    return 0


def cmd_compare(args):
    with open(args.baseline) as fh:
        baseline = json.load(fh)
    with open(args.spec) as fh:
        spec = json.load(fh)
    cmp = compare_runs(baseline, spec)
    text = json.dumps(cmp, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}")
    return 0 if cmp["verdict"]["outputs_identical"] else 1


# ---------------------------------------------------------------------------
# self-test (no server)
# ---------------------------------------------------------------------------


def self_test():
    def run(label, spec_type, tps, text):
        return {
            "label": label,
            "spec_type": spec_type,
            "contexts": [
                {
                    "context": 4096,
                    "equiv": {"status": 200, "text": text, "completion_tokens": 64},
                    "summary": {"output_tps": {"median": tps}, "ttft_ms": {"median": 100.0}},
                    "decode": [{"draft": {"draft_ratio": 0.6}}],
                    "acceptance": {"draft_ratio": 0.6, "draft_n": 100, "draft_n_accepted": 60},
                }
            ],
        }

    ok = compare_runs(run("b", "none", 10.0, "hello world"), run("s", "ngram-map-k4v", 13.5, "hello world"))
    assert ok["verdict"]["outputs_identical"] is True, ok
    assert abs(ok["contexts"][0]["speedup"] - 1.35) < 1e-9, ok
    assert ok["verdict"]["min_speedup_at_acceptance_ge_0.5"] == 1.35, ok

    bad = compare_runs(run("b", "none", 10.0, "hello world"), run("s", "ngram-map-k4v", 13.5, "hello worXX"))
    assert bad["verdict"]["outputs_identical"] is False, bad
    assert bad["contexts"][0]["equiv_divergence_index"] == 9, bad

    # A low-acceptance spec run that is slower must be flagged, not hidden.
    slow = run("s", "ngram-map-k4v", 9.0, "hello world")
    slow["contexts"][0]["acceptance"] = {"draft_ratio": 0.01}
    slow["contexts"][0]["decode"][0]["draft"] = {"draft_ratio": 0.01}
    low = compare_runs(run("b", "none", 10.0, "hello world"), slow)
    assert low["verdict"]["low_acceptance_regressions"], low

    acc = parse_acceptance([
        "I slot print_timing: draft acceptance = 0.50000 (   40 accepted /    80 generated), mean len =  1.50",
        "garbage line",
        "I slot print_timing: draft acceptance = 0.75000 (   30 accepted /    40 generated), mean len =  1.75",
    ])
    assert acc["draft_n"] == 120 and acc["draft_n_accepted"] == 70, acc
    assert abs(acc["draft_ratio"] - 70 / 120) < 1e-9, acc

    fields = draft_fields({"cache_n": 100, "draft_n": 10, "draft_n_accepted": 6}, None)
    assert fields["cache_n"] == 100 and abs(fields["draft_ratio"] - 0.6) < 1e-9, fields
    print("self-test: OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="measure suffix/n-gram speculative decoding")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="measure one server configuration")
    r.add_argument("--base-url", default=BASE)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--tier", default=os.environ.get("BONGO_TIER", "iq2_xs"))
    r.add_argument("--contexts", default="4096,131072")
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--equivalence-tokens", type=int, default=64)
    r.add_argument("--warmup-tokens", type=int, default=256)
    r.add_argument("--context-limit-guard", type=int, default=0,
                   help="server n_ctx; caps the prompt so prompt+max_tokens cannot overflow")
    r.add_argument("--skip-equivalence", action="store_true")
    r.add_argument("--server-log", default=None)
    r.add_argument("--server-note", default=None,
                   help="free-text record of the exact server flags/engine/tier for reproducibility")
    r.add_argument("--timeout", type=float, default=float(os.environ.get("BONGO_TIMEOUT", "3600")))
    r.add_argument("--label", required=True)
    r.add_argument("--spec-type", default="none")
    r.add_argument("--spec-synth", default=None)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="compare spec-off and spec-on runs (pure)")
    c.add_argument("--baseline", required=True)
    c.add_argument("--spec", required=True)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("self-test", help="internal consistency test (no server)")
    t.set_defaults(func=lambda _a: self_test())

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
