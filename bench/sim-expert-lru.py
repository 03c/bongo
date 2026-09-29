#!/usr/bin/env python3
"""Held-out A/B: frozen frequency profile vs dynamic VRAM LRU (BAS-76, Step 2).

R4 (`docs/research/expert-activation-skew.md`) measures an *offline* frequency
profile at 0.88-0.99 held-out activation coverage.  R3 (llama.cpp PR #27861, a
different model) reports a *static* top-32 ranking recovering only ~10%
out-of-sample, against 67-81% for an *online LRU*.  The two cannot both be the
shipping policy, so this tool replays bongo's own captured router traces and
measures every candidate policy on the same held-out corpus, at the same byte
budget:

  static_profile   top cells by training count, frozen (R4's policy)
  static_insample  the same profile trained on the test corpus (upper bound)
  static_layer_16  llama.cpp `--n-cpu-moe 16` (Stage 0 baseline)
  byte_budget_layers  cheapest-layer-first whole-layer residency (Step 1)
  cold_lru         empty start, online LRU at (layer, expert) granularity
  profile_lru      the offline profile as LRU *initialisation* (the ship rule)
  per_layer_lru    an independent LRU per layer, equal byte share per layer

Coverage = fraction of routed `(layer, expert)` events served from the resident
set.  Leave-one-corpus-out: the profile is built on all corpora except the one
being scored.  Raw captures come from `bench/results/2026-09-28-expert-activation/`.

Stdlib only.

  python3 bench/sim-expert-lru.py \
      --raw bench/results/2026-09-28-expert-activation/raw \
      --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
      --budget-gib 22.40 \
      --corpora doc,code,chat,convo \
      --decode doc=.../doc_dec.tsv.gz,chat=.../chat_dec.tsv.gz \
      --out bench/results/2026-09-28-byte-budget-placement/lru-ab.json
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
from collections import Counter, OrderedDict

N_LAYERS = 48
N_EXPERTS = 512
GIB = 1024 ** 3

_ANALYZER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "analyze-expert-activation.py")


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("bongo_analyze_ea", _ANALYZER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def byte_fn(layer_bytes):
    return lambda key: layer_bytes[key[0]] / N_EXPERTS


def profile_set(counts: Counter, byte_of, budget: float):
    """Top cells by count until the byte budget is full (frozen set)."""
    chosen = set()
    used = 0.0
    for key, _n in counts.most_common():
        b = byte_of(key)
        if used + b <= budget:
            chosen.add(key)
            used += b
    return chosen, used


def layer_budget_set(layer_bytes, budget: float, resident_layers):
    return {(l, e) for l in resident_layers for e in range(N_EXPERTS)}


def replay_trace(trace, byte_of, budget, preload=None):
    """Standard online LRU.  Returns (hits, events)."""
    cache: "OrderedDict[tuple[int, int], int]" = OrderedDict()
    used = 0.0
    hits = 0
    events = 0
    if preload:
        for key in preload:
            b = byte_of(key)
            if used + b <= budget:
                cache[key] = 1
                used += b
    for key in trace:
        events += 1
        if key in cache:
            cache.move_to_end(key)
            hits += 1
            continue
        b = byte_of(key)
        while cache and used + b > budget:
            _old, _ = cache.popitem(last=False)
            used -= byte_of(_old)
        if b <= budget:
            cache[key] = 1
            used += b
    return hits, events


def replay_per_layer_lru(trace, byte_of, budget):
    caches: dict[int, "OrderedDict[tuple[int, int], int]"] = {}
    used: dict[int, float] = {}
    per_layer_budget = budget / N_LAYERS
    hits = events = 0
    for key in trace:
        layer = key[0]
        cache = caches.get(layer)
        if cache is None:
            cache = caches[layer] = OrderedDict()
            used[layer] = 0.0
        events += 1
        if key in cache:
            cache.move_to_end(key)
            hits += 1
            continue
        b = byte_of(key)
        while cache and used[layer] + b > per_layer_budget:
            _old, _ = cache.popitem(last=False)
            used[layer] -= byte_of(key)
        if b <= per_layer_budget:
            cache[key] = 1
            used[layer] += b
    return hits, events


def coverage_of(trace, resident: set):
    hits = sum(1 for key in trace if key in resident)
    return hits / len(trace) if trace else 0.0


def tokens_of(layers):
    return len(next(iter(layers.values())))


def trace_of(layers, analyzer):
    """Token-major access sequence: for each token, layers 0..47 in order."""
    n_tokens = tokens_of(layers)
    out = []
    for t in range(n_tokens):
        for layer in range(N_LAYERS):
            sets = layers.get(layer)
            if sets is None or t >= len(sets):
                continue
            for e in sets[t]:
                out.append((layer, e))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--expert-bytes", required=True)
    ap.add_argument("--budget-gib", type=float, default=22.40)
    ap.add_argument("--corpora", default="doc,code,chat,convo")
    ap.add_argument("--decode", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", default="iq2_xs")
    args = ap.parse_args(argv)

    analyzer = _load_analyzer()
    with open(args.expert_bytes) as fh:
        layer_bytes = json.load(fh)["expert_bytes_by_layer"]
    byte_of = byte_fn(layer_bytes)
    budget = args.budget_gib * GIB

    names = [n for n in args.corpora.split(",") if n]

    def raw_path(n):
        p = os.path.join(args.raw, n + ".tsv")
        if not os.path.exists(p) and os.path.exists(p + ".gz"):
            return p + ".gz"
        return p

    data = {n: analyzer.load_layer_sets(raw_path(n)) for n in names}
    counts = {n: analyzer.counts_of(v) for n, v in data.items()}
    traces = {n: trace_of(v, analyzer) for n, v in data.items()}

    # cheapest-layer-first whole-layer residency (Step 1)
    events_pool = [0] * N_LAYERS
    for n in names:
        c = analyzer.concentration(counts[n])
        for l in range(N_LAYERS):
            events_pool[l] += c["per_layer"][l]["events"]
    resident_layers = sorted(analyzer.static_whole_layer_greedy(byte_of, budget)[2])

    result = {
        "schema": "bongo.expert-lru-ab.v1",
        "tier": args.tier,
        "budget_gib": args.budget_gib,
        "budget_bytes": budget,
        "engine_revision": "llama.cpp b11223 (4da6337767f973e2b4d0797e5b323d77d8565e4a)",
        "capture": "bench/results/2026-09-28-expert-activation/raw (CPU router capture, R4)",
        "policy_notes": {
            "static_profile": "frozen top-count profile from the other corpora (R4 policy)",
            "static_insample": "frozen profile trained on the scored corpus (coverage ceiling)",
            "static_layer_16": "llama.cpp --n-cpu-moe 16 (Stage 0 baseline)",
            "byte_budget_layers": "cheapest-layer-first whole-layer residency (Step 1)",
            "cold_lru": "empty-start online LRU",
            "profile_lru": "held-out profile as LRU initialisation (ship rule)",
            "per_layer_lru": "independent LRU per layer, equal byte share per layer",
        },
        "corpora": {n: {"tokens": tokens_of(data[n]), "events": len(traces[n])} for n in names},
        "step1_resident_layers": resident_layers,
        "held_out": {},
        "decode_held_out": {},
    }

    print(f"budget={args.budget_gib:.2f} GiB  corpora={names}")
    print(f"Step-1 resident layers: n={len(resident_layers)} {resident_layers}")

    for test in names:
        train_counts = Counter()
        for n in names:
            if n != test:
                train_counts.update(counts[n])
        trace = traces[test]
        static_profile, _used = profile_set(train_counts, byte_of, budget)
        static_insample, _ = profile_set(counts[test], byte_of, budget)
        layer16 = layer_budget_set(layer_bytes, budget, list(range(16, N_LAYERS)))
        layer_byte = layer_budget_set(layer_bytes, budget, resident_layers)
        preload = sorted(static_profile, key=lambda k: -train_counts[k])

        row = {
            "tokens": tokens_of(data[test]),
            "events": len(trace),
            "static_profile": coverage_of(trace, static_profile),
            "static_insample": coverage_of(trace, static_insample),
            "static_layer_16": coverage_of(trace, layer16),
            "byte_budget_layers": coverage_of(trace, layer_byte),
            "cold_lru": None,
            "profile_lru": None,
            "per_layer_lru": None,
            "profile_cells": len(static_profile),
        }
        h, e = replay_trace(trace, byte_of, budget, preload=None)
        row["cold_lru"] = h / e
        h, e = replay_trace(trace, byte_of, budget, preload=preload)
        row["profile_lru"] = h / e
        h, e = replay_per_layer_lru(trace, byte_of, budget)
        row["per_layer_lru"] = h / e
        result["held_out"][test] = row
        print(f"\nheld-out {test:6s} tokens={row['tokens']:5d} events={row['events']:8d}")
        for k in ("static_insample", "static_profile", "profile_lru", "cold_lru",
                  "per_layer_lru", "byte_budget_layers", "static_layer_16"):
            print(f"  {k:20s} {row[k]:.4f}")

    # --- decode transfer (held-out profile -> real decode routing) --------- #
    for item in args.decode.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, path = item.partition("=")
        if not os.path.exists(path) and not os.path.exists(path + ".gz"):
            continue
        dec = analyzer.load_layer_sets(path)
        dec_trace = trace_of(dec, analyzer)
        train_counts = Counter()
        for n in names:
            if n != name:
                train_counts.update(counts[n])
        static_profile, _ = profile_set(train_counts, byte_of, budget)
        preload = sorted(static_profile, key=lambda k: -train_counts[k])
        h, e = replay_trace(dec_trace, byte_of, budget, preload=None)
        cold = h / e
        h, e = replay_trace(dec_trace, byte_of, budget, preload=preload)
        prof = h / e
        row = {
            "decode_tokens": tokens_of(dec),
            "decode_events": len(dec_trace),
            "static_profile": coverage_of(dec_trace, static_profile),
            "cold_lru": cold,
            "profile_lru": prof,
        }
        result["decode_held_out"][name] = row
        print(f"\ndecode held-out {name}: tokens={row['decode_tokens']} "
              f"events={row['decode_events']} static={row['static_profile']:.4f} "
              f"cold_lru={cold:.4f} profile_lru={prof:.4f}")

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
