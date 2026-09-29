#!/usr/bin/env python3
"""Generate the MoE expert-cache profile consumed by the BAS-139 engine.

The Step-2 engine (`tools/patches/moe-expert-cache.patch`) keeps a fixed VRAM
cache of expert slices per host-resident MoE layer and tracks it online with a
global LRU.  BAS-76 measured that the best *initial* state is the offline R4
frequency profile, not a cold cache and not a frozen profile: the profile
removes the cold-LRU warmup penalty while the online LRU still follows serving
drift.  This tool turns the captured router traces into that start state.

The output is the engine's profile format, one line per layer:

    L <il> <expert> <expert> ...      # experts in descending priority

The line length is the layer's slot count.  Experts are the most frequent
`(layer, expert)` cells until the byte budget is full (the `profile_lru`
preload of `bench/sim-expert-lru.py`), so the cache fits the same expert bytes
the Stage-0 `--n-cpu-moe 16` / Step-1 `-ot` rules spend on residency.

Stdlib only.

    python3 bench/gen-moe-cache-profile.py \
        --raw bench/results/2026-09-28-expert-activation/raw \
        --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
        --budget-gib 22.40 --corpora doc,code,chat,convo --tier iq2_xs \
        --out bench/results/2026-09-29-moe-cache/profile-iq2_xs-22.40.txt \
        --json-out bench/results/2026-09-29-moe-cache/profile-iq2_xs-22.40.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter

GIB = 1024 ** 3
N_LAYERS = 48
N_EXPERTS = 512

_ANALYZER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "analyze-expert-activation.py")


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("bongo_analyze_ea", _ANALYZER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def byte_fn(layer_bytes):
    return lambda key: layer_bytes[key[0]] / N_EXPERTS


def profile_cells(counts: Counter, byte_of, budget: float):
    """Top cells by count until the byte budget is full, priority order kept."""
    chosen = []
    used = 0.0
    for key, _n in counts.most_common():
        b = byte_of(key)
        if used + b <= budget:
            chosen.append((key, _n))
            used += b
    return chosen, used


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--expert-bytes", required=True)
    ap.add_argument("--budget-gib", type=float, default=22.40)
    ap.add_argument("--corpora", default="doc,code,chat,convo")
    ap.add_argument("--tier", default="iq2_xs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    analyzer = _load_analyzer()
    with open(args.expert_bytes) as fh:
        layer_bytes = json.load(fh)["expert_bytes_by_layer"]
    byte_of = byte_fn(layer_bytes)
    budget = args.budget_gib * GIB

    names = [n for n in args.corpora.split(",") if n]
    if not names:
        print("error: --corpora is empty", file=sys.stderr)
        return 2

    counts: Counter = Counter()
    total_events = 0
    per_corpus = {}
    for n in names:
        p = os.path.join(args.raw, n + ".tsv")
        if not os.path.exists(p) and os.path.exists(p + ".gz"):
            p = p + ".gz"
        if not os.path.exists(p):
            print(f"error: trace not found: {p}", file=sys.stderr)
            return 2
        c = analyzer.counts_of(analyzer.load_layer_sets(p))
        per_corpus[n] = sum(c.values())
        total_events += sum(c.values())
        counts.update(c)

    chosen, used = profile_cells(counts, byte_of, budget)
    covered = sum(n for _k, n in chosen)

    # group per layer, keep descending-count order
    per_layer: dict[int, list[int]] = {l: [] for l in range(N_LAYERS)}
    for (layer, expert), _n in chosen:
        per_layer[layer].append(expert)

    total_slots = sum(len(v) for v in per_layer.values())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("bongo-moe-cache-profile 1\n")
        fh.write(f"# tier {args.tier}\n")
        fh.write(f"# budget_gib {args.budget_gib}\n")
        fh.write(f"# corpora {','.join(names)}\n")
        fh.write(f"# raw {args.raw}\n")
        fh.write(f"# source profile_cells (top (layer,expert) by count until budget)\n")
        fh.write(f"# cells {total_slots} bytes {int(used)}\n")
        for l in range(N_LAYERS):
            experts = per_layer[l]
            if not experts:
                continue
            fh.write("L %d" % l)
            fh.write("".join(" %d" % e for e in experts))
            fh.write("\n")

    meta = {
        "schema": "bongo.moe-cache-profile.v1",
        "tier": args.tier,
        "budget_gib": args.budget_gib,
        "budget_bytes": budget,
        "raw": args.raw,
        "corpora": names,
        "events_total": total_events,
        "events_covered": covered,
        "coverage": (covered / total_events) if total_events else 0.0,
        "cells": total_slots,
        "bytes": int(used),
        "expert_bytes_by_layer": layer_bytes,
        "per_layer_slots": {str(l): len(per_layer[l]) for l in range(N_LAYERS)},
        "engine": "llama.cpp b11223 + tools/patches/moe-expert-cache.patch",
        "policy": "profile initialisation, then online global LRU",
        "per_corpus_events": per_corpus,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.write("\n")

    print(f"cells={total_slots} bytes={used/GIB:.3f} GiB "
          f"coverage={meta['coverage']:.4f} layers={sum(1 for v in per_layer.values() if v)}")
    print(f"wrote {args.out}")
    if args.json_out:
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
