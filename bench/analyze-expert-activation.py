#!/usr/bin/env python3
"""Expert-activation skew analysis for the bongo reference box (BAS-66 / R4).

Reads the raw router captures produced by bench/tools/route_capture.c
(`TOPK ffn_moe_topk-<layer> <ne0..ne3> <idx...>`, with optional `STEP <i>`
markers in decode files), builds frequency profiles, and answers the residency
question:

  - how concentrated the routing distribution is over (layer, expert);
  - coverage of a byte-budgeted resident set, frequency-ranked vs the static
    `--n-cpu-moe` layer rule vs arrival order;
  - leave-one-out transfer of a profile between prompt classes;
  - temporal persistence of an expert set across adjacent tokens;
  - transfer of a prefill-built profile to real per-token decode routing.

Standard library only.

  python3 bench/analyze-expert-activation.py \
      --raw bench/results/2026-09-28-expert-activation/raw \
      --decode doc=bench/results/2026-09-28-expert-activation/raw/doc_dec.tsv.gz \
      --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
      --out bench/results/2026-09-28-expert-activation/analysis.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter

GIB = 1024 ** 3
N_LAYERS = 48
N_EXPERTS = 512
N_ACTIVE = 10


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _parse(path: str):
    """Yield (step_or_None, layer, tuple_of_experts) per TOPK line."""
    step = None
    with _open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            if parts[0] == "STEP":
                step = int(parts[1])
                continue
            if parts[0] != "TOPK":
                continue
            name = parts[1]
            ne0, ne1, ne2, ne3 = (int(parts[2]), int(parts[3]),
                                  int(parts[4]), int(parts[5]))
            vals = [int(x) for x in parts[6:] if x != ""]
            if len(vals) != ne0 * ne1 * ne2 * ne3:
                raise ValueError(
                    f"{path}: {name} expected {ne0*ne1*ne2*ne3} values, got {len(vals)}")
            layer = int(name.rsplit("-", 1)[1])
            for t in range(ne1 * ne2 * ne3):
                yield step, layer, tuple(vals[t * ne0:(t + 1) * ne0])


def load_layer_sets(path: str):
    """{layer: [tuple(expert_ids), ...]}, one tuple per token (all steps pooled)."""
    layers: dict[int, list[tuple[int, ...]]] = {}
    for _step, layer, exps in _parse(path):
        layers.setdefault(layer, []).append(exps)
    if sorted(layers) != list(range(N_LAYERS)):
        raise ValueError(f"{path}: expected layers 0..{N_LAYERS-1}, got {sorted(layers)}")
    return layers


def counts_of(layers) -> Counter:
    c: Counter = Counter()
    for layer, sets in layers.items():
        for s in sets:
            for e in s:
                c[(layer, e)] += 1
    return c


# --------------------------------------------------------------------------- #
# residency sets under a byte budget
# --------------------------------------------------------------------------- #

def make_per_expert_bytes(layer_bytes):
    return [b / N_EXPERTS for b in layer_bytes]


def greedy(items, byte_of, budget):
    chosen = set()
    used = 0.0
    for item, _rank in items:
        b = byte_of(item)
        if used + b <= budget:
            chosen.add(item)
            used += b
    return chosen, used


def hot_by_count(counts: Counter, byte_of, budget):
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return greedy(items, byte_of, budget)


def hot_by_density(counts: Counter, byte_of, budget):
    items = sorted(counts.items(), key=lambda kv: -(kv[1] / byte_of(kv[0])))
    return greedy(items, byte_of, budget)


def arrival_order(layers, byte_of, budget):
    chosen = set()
    used = 0.0
    for layer in sorted(layers):
        for s in layers[layer]:
            for e in s:
                key = (layer, e)
                if key in chosen:
                    continue
                b = byte_of(key)
                if used + b <= budget:
                    chosen.add(key)
                    used += b
    return chosen, used


def static_contiguous(byte_of, budget):
    """llama.cpp --n-cpu-moe N: layers 0..N-1 on CPU, residents N..47."""
    for n_cpu in range(0, N_LAYERS + 1):
        used = sum(byte_of((l, 0)) * N_EXPERTS for l in range(n_cpu, N_LAYERS))
        if used <= budget:
            chosen = {(l, e) for l in range(n_cpu, N_LAYERS) for e in range(N_EXPERTS)}
            return chosen, used, n_cpu
    return set(), 0.0, N_LAYERS


def static_whole_layer_greedy(byte_of, budget):
    layer_cost = [byte_of((l, 0)) * N_EXPERTS for l in range(N_LAYERS)]
    chosen_layers = set()
    used = 0.0
    for l in sorted(range(N_LAYERS), key=lambda i: layer_cost[i]):
        if used + layer_cost[l] <= budget:
            chosen_layers.add(l)
            used += layer_cost[l]
    chosen = {(l, e) for l in chosen_layers for e in range(N_EXPERTS)}
    return chosen, used, sorted(chosen_layers)


def random_expected_coverage(counts: Counter, byte_of, budget, trials=8):
    all_cells = [(l, e) for l in range(N_LAYERS) for e in range(N_EXPERTS)]
    total_events = sum(counts.values())
    covs = []
    state = 12345
    for _ in range(trials):
        chosen = set()
        used = 0.0
        order = sorted(all_cells, key=lambda k: ((k[0] * 2654435761 + k[1] * 40503 + state) % (1 << 31)))
        for k in order:
            b = byte_of(k)
            if used + b <= budget:
                chosen.add(k)
                used += b
        state = (state * 1103515245 + 12345) % (1 << 31)
        covs.append(coverage(chosen, counts))
    return sum(covs) / len(covs)


def coverage(chosen, counts: Counter) -> float:
    total = sum(counts.values())
    hit = sum(n for k, n in counts.items() if k in chosen)
    return hit / total


# --------------------------------------------------------------------------- #
# concentration
# --------------------------------------------------------------------------- #

def concentration(counts: Counter):
    total = sum(counts.values())
    ranked = [n for _k, n in counts.most_common()]
    touched = len(counts)
    ent = 0.0
    for n in ranked:
        p = n / total
        ent -= p * math.log(p)
    thresholds = {}
    cum = 0
    targets = [0.5, 0.8, 0.9, 0.95, 0.99]
    ti = 0
    for i, n in enumerate(ranked, 1):
        cum += n
        while ti < len(targets) and cum / total >= targets[ti]:
            thresholds[targets[ti]] = i
            ti += 1
    per_layer = {}
    for layer in range(N_LAYERS):
        lc = [n for (l, _e), n in counts.items() if l == layer]
        ltotal = sum(lc)
        per_layer[layer] = {
            "events": ltotal,
            "distinct_experts": len(lc),
            "top10_share": (sum(sorted(lc, reverse=True)[:N_ACTIVE]) / ltotal) if ltotal else 0.0,
            "entropy_bits": (-sum((n / ltotal) * math.log2(n / ltotal) for n in lc)) if ltotal else 0.0,
        }
    return {
        "total_events": total,
        "touched_cells": touched,
        "grid_cells": N_LAYERS * N_EXPERTS,
        "touched_fraction": touched / (N_LAYERS * N_EXPERTS),
        "entropy_bits": ent,
        "effective_experts": math.exp(ent),
        "top_n_coverage": {f"{t:.2f}": v for t, v in thresholds.items()},
        "per_layer": per_layer,
    }


# --------------------------------------------------------------------------- #
# temporal persistence
# --------------------------------------------------------------------------- #

def temporal(layers, static_topk_by_layer=None):
    hit = tot = 0
    persist_by_layer = {}
    for layer, sets in layers.items():
        lh = lt = 0
        for t in range(len(sets) - 1):
            prev = set(sets[t])
            for e in sets[t + 1]:
                lt += 1
                if e in prev:
                    lh += 1
        persist_by_layer[layer] = lh / lt if lt else 0.0
        hit += lh
        tot += lt
    out = {"adjacent_persistence": hit / tot if tot else 0.0, "per_layer": persist_by_layer}
    if static_topk_by_layer:
        sh = st = 0
        for layer, sets in layers.items():
            top = static_topk_by_layer.get(layer, set())
            if not top:
                continue
            for s in sets:
                for e in s:
                    st += 1
                    if e in top:
                        sh += 1
        out["static_top10_predictor"] = sh / st if st else 0.0
    return out


def static_topk(counts: Counter, k=N_ACTIVE):
    per_layer: dict[int, list[tuple[int, int]]] = {}
    for (l, e), n in counts.items():
        per_layer.setdefault(l, []).append((n, e))
    return {l: {e for _n, e in sorted(v, reverse=True)[:k]} for l, v in per_layer.items()}


# --------------------------------------------------------------------------- #
# rank agreement between two distributions
# --------------------------------------------------------------------------- #

def spearman_per_layer(counts_a: Counter, counts_b: Counter):
    """Mean Spearman rho over layers for the 512 expert frequencies."""
    rows = []
    for layer in range(N_LAYERS):
        a = [counts_a.get((layer, e), 0) for e in range(N_EXPERTS)]
        b = [counts_b.get((layer, e), 0) for e in range(N_EXPERTS)]
        if sum(a) == 0 or sum(b) == 0:
            continue
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        ra, rb = ranks(a), ranks(b)
        n = len(ra)
        ma, mb = sum(ra) / n, sum(rb) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        da = math.sqrt(sum((x - ma) ** 2 for x in ra))
        db = math.sqrt(sum((y - mb) ** 2 for y in rb))
        rows.append(num / (da * db) if da and db else 0.0)
    return sum(rows) / len(rows) if rows else 0.0


def topk_overlap_per_layer(counts_a: Counter, counts_b: Counter, k=N_ACTIVE):
    ta = static_topk(counts_a, k)
    tb = static_topk(counts_b, k)
    vals = []
    for layer in range(N_LAYERS):
        if not ta[layer] and not tb[layer]:
            continue
        vals.append(len(ta[layer] & tb[layer]) / k)
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

BUDGETS_GIB = [4, 8, 12, 16, 20, 22.4, 25.0, 28, 33.02]


def parse_decode_arg(spec):
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--decode expects name=path, got {item!r}")
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory of <corpus>.tsv[.gz] captures")
    ap.add_argument("--decode", default="", help="name=path[,name=path] decode captures")
    ap.add_argument("--expert-bytes", required=True, help="expert-bytes-<tier>.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default="")
    args = ap.parse_args()

    with open(args.expert_bytes) as fh:
        eb = json.load(fh)
    layer_bytes = eb["expert_bytes_by_layer"]
    assert len(layer_bytes) == N_LAYERS, len(layer_bytes)
    total_expert_bytes = sum(layer_bytes)
    byte_of = lambda le: layer_bytes[le[0]] / N_EXPERTS

    decode_paths = parse_decode_arg(args.decode)
    raw_dir = args.raw

    def raw_path(n):
        p = os.path.join(raw_dir, n + ".tsv")
        return p + ".gz" if not os.path.exists(p) and os.path.exists(p + ".gz") else p

    if args.corpora:
        names = [n for n in args.corpora.split(",") if n]
    else:
        names = []
        for fn in sorted(os.listdir(raw_dir)):
            if fn.endswith(".tsv.gz"):
                names.append(fn[:-7])
            elif fn.endswith(".tsv"):
                names.append(fn[:-4])
    names = [n for n in names if os.path.exists(raw_path(n))]

    data = {n: load_layer_sets(raw_path(n)) for n in names}
    counts = {n: counts_of(v) for n, v in data.items()}
    all_counts = Counter()
    for c in counts.values():
        all_counts.update(c)

    result = {
        "tier": eb.get("tier", "unknown"),
        "n_layers": N_LAYERS,
        "n_experts_per_layer": N_EXPERTS,
        "n_active": N_ACTIVE,
        "total_expert_bytes": total_expert_bytes,
        "total_expert_gib": total_expert_bytes / GIB,
        "layer_expert_bytes": layer_bytes,
        "corpora": {},
        "vram": {},
        "leave_one_out": {},
        "leave_one_corpus_out": {},
        "budget_sweep": {},
        "temporal": {},
        "decode": {},
    }

    for n in names:
        result["corpora"][n] = {
            "tokens": len(next(iter(data[n].values()))),
            "events": sum(counts[n].values()),
            "layers_observed": len(data[n]),
        }

    # --- VRAM budget ----------------------------------------------------- #
    non_expert_128k = 6.86
    non_expert_4k = 6.42
    usably = 31.92
    result["vram"] = {
        "usable_gib": usably,
        "non_expert_128k_gib": non_expert_128k,
        "non_expert_4k_gib": non_expert_4k,
        "arithmetic_expert_budget_128k_gib": usably - non_expert_128k,
        "arithmetic_expert_budget_4k_gib": usably - non_expert_4k,
        "measured_safe_gib": 22.40,
        "note": ("non-expert footprint = sweep VRAM minus GPU-resident expert bytes; "
                 "n=16 and n=24 agree within 0.01 GiB, so it is the dense+KV+state+buffers cost"),
        "source": "bench/results/2026-09-27-expert-placement/sweep-matrix.json",
    }

    # --- budget sweep on the pooled corpus ------------------------------- #
    pooled_layers = {}
    for n in names:
        for l in range(N_LAYERS):
            pooled_layers.setdefault(l, []).extend(data[n][l])

    sweep = {}
    for bg in BUDGETS_GIB:
        budget = bg * GIB
        entry = {}
        chosen, used = hot_by_count(all_counts, byte_of, budget)
        entry["hot_count"] = {"coverage": coverage(chosen, all_counts), "bytes": used,
                              "cells": len(chosen)}
        chosen, used = hot_by_density(all_counts, byte_of, budget)
        entry["hot_density"] = {"coverage": coverage(chosen, all_counts), "bytes": used,
                                "cells": len(chosen)}
        chosen, used, n_cpu = static_contiguous(byte_of, budget)
        entry["static_contiguous"] = {"coverage": coverage(chosen, all_counts), "bytes": used,
                                      "n_cpu_moe": n_cpu, "resident_layers": N_LAYERS - n_cpu}
        chosen, used, lyr = static_whole_layer_greedy(byte_of, budget)
        entry["static_layer_greedy"] = {"coverage": coverage(chosen, all_counts), "bytes": used,
                                        "resident_layers": len(lyr)}
        chosen, used = arrival_order(pooled_layers, byte_of, budget)
        entry["arrival_order"] = {"coverage": coverage(chosen, all_counts), "bytes": used,
                                  "cells": len(chosen)}
        entry["random_expected"] = {"coverage": random_expected_coverage(all_counts, byte_of, budget)}
        sweep[f"{bg:.2f}"] = entry
    result["budget_sweep"] = sweep

    # --- leave-one-out transfer (per corpus) ----------------------------- #
    loo = {}
    for train in names:
        row = {}
        for test in names:
            chosen, used = hot_by_count(counts[train], byte_of, 22.40 * GIB)
            row[test] = {"coverage": coverage(chosen, counts[test]), "bytes": used,
                         "cells": len(chosen)}
        loo[train] = row
    result["leave_one_out"] = {"budget_gib": 22.40, "rows": loo}

    # --- leave-one-corpus-out (build on the rest) ------------------------ #
    loco = {}
    for test in names:
        rest = [n for n in names if n != test]
        train_counts = Counter()
        for n in rest:
            train_counts.update(counts[n])
        entry = {}
        for bg in BUDGETS_GIB:
            chosen, used = hot_by_count(train_counts, byte_of, bg * GIB)
            entry[f"{bg:.2f}"] = {"coverage": coverage(chosen, counts[test]), "bytes": used,
                                  "cells": len(chosen)}
        entry["static_contiguous_22.40"] = static_contiguous(byte_of, 22.40 * GIB)[2]
        chosen, used, n_cpu = static_contiguous(byte_of, 22.40 * GIB)
        entry["static_contiguous_coverage_22.40"] = coverage(chosen, counts[test])
        loco[test] = entry
    result["leave_one_corpus_out"] = loco

    # --- concentration ---------------------------------------------------- #
    result["concentration"] = {n: concentration(counts[n]) for n in names}
    result["concentration_pooled"] = concentration(all_counts)

    # --- temporal --------------------------------------------------------- #
    pooled_static_topk = static_topk(all_counts)
    tem = {}
    for n in names:
        tem[n] = temporal(data[n], static_topk_by_layer=pooled_static_topk)
    tem["pooled_vs_static_topk"] = temporal(pooled_layers, static_topk_by_layer=pooled_static_topk)
    result["temporal"] = tem

    # --- decode transfer -------------------------------------------------- #
    for n, path in decode_paths.items():
        if not os.path.exists(path) and not os.path.exists(path + ".gz"):
            continue
        dec_layers = load_layer_sets(path)
        dec_counts = counts_of(dec_layers)
        pre_counts = counts[n] if n in counts else None
        if pre_counts is None:
            continue
        entry = {
            "decode_tokens": len(next(iter(dec_layers.values()))),
            "decode_events": sum(dec_counts.values()),
            "concentration": concentration(dec_counts),
            "temporal": temporal(dec_layers, static_topk_by_layer=static_topk(dec_counts)),
            "spearman_prefill_vs_decode": spearman_per_layer(pre_counts, dec_counts),
            "top10_overlap_prefill_vs_decode": topk_overlap_per_layer(pre_counts, dec_counts),
            "transfer": {},
        }
        for bg in BUDGETS_GIB:
            budget = bg * GIB
            chosen, _ = hot_by_count(pre_counts, byte_of, budget)
            pre_to_dec = coverage(chosen, dec_counts)
            chosen2, _ = hot_by_count(dec_counts, byte_of, budget)
            dec_to_pre = coverage(chosen2, pre_counts)
            self_dec, _ = hot_by_count(dec_counts, byte_of, budget)
            entry["transfer"][f"{bg:.2f}"] = {
                "prefill_profile_on_decode": pre_to_dec,
                "decode_profile_on_prefill": dec_to_pre,
                "decode_profile_on_decode": coverage(self_dec, dec_counts),
            }
        chosen, used, n_cpu = static_contiguous(byte_of, 22.40 * GIB)
        entry["static_contiguous_coverage_22.40"] = coverage(chosen, dec_counts)
        entry["static_contiguous_n_cpu_22.40"] = n_cpu
        # layer-47 check: decode covers it; prefill only saw the last token
        l47 = [len(dec_counts) and dec_counts.get((47, e), 0) for e in range(N_EXPERTS)]
        non47 = [dec_counts.get((l, e), 0) for l in range(47) for e in range(N_EXPERTS)]
        entry["layer47_decode_events"] = dec_counts and sum(l47)
        entry["layer47_top10_share"] = (sorted(l47, reverse=True)[:10] and
                                        sum(sorted(l47, reverse=True)[:10]) / sum(l47))
        entry["mean_layer_top10_share"] = (
            sum(concentration(dec_counts)["per_layer"][l]["top10_share"] for l in range(47)) / 47)
        result["decode"][n] = entry

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}")

    # --- human summary ---------------------------------------------------- #
    print(f"\ntier={result['tier']} total expert bytes = {result['total_expert_gib']:.2f} GiB")
    for n in names:
        c = result["corpora"][n]
        print(f"  {n:6s} tokens={c['tokens']:5d} events={c['events']}")
    for n, e in result["decode"].items():
        print(f"  decode[{n}] tokens={e['decode_tokens']} events={e['decode_events']}")
    print("\nbudget sweep on pooled corpus (coverage = fraction of routed events served):")
    print(f"{'GiB':>6} {'hot_count':>10} {'hot_dens':>9} {'static_ctg':>11} {'layer_grdy':>11} "
          f"{'arrival':>8} {'random':>7} {'ncpu':>5}")
    for bg in BUDGETS_GIB:
        e = sweep[f"{bg:.2f}"]
        print(f"{bg:6.2f} {e['hot_count']['coverage']:10.4f} {e['hot_density']['coverage']:9.4f} "
              f"{e['static_contiguous']['coverage']:11.4f} {e['static_layer_greedy']['coverage']:11.4f} "
              f"{e['arrival_order']['coverage']:8.4f} {e['random_expected']['coverage']:7.4f} "
              f"{e['static_contiguous']['n_cpu_moe']:5d}")
    cp = result["concentration_pooled"]
    print(f"\npooled: touched {cp['touched_cells']}/{cp['grid_cells']} cells "
          f"({cp['touched_fraction']:.3f}), entropy {cp['entropy_bits']:.3f} bits, "
          f"effective experts {cp['effective_experts']:.0f}")
    print("top-N coverage:", cp["top_n_coverage"])
    print("\nleave-one-out (hot_count @ 22.40 GiB), rows=train, cols=test:")
    print("        " + " ".join(f"{t:>8}" for t in names))
    for train in names:
        print(f"{train:6s} " + " ".join(f"{loo[train][t]['coverage']:8.4f}" for t in names))
    print("\nleave-one-corpus-out (train=all others), coverage on held-out corpus:")
    for test in names:
        e = result["leave_one_corpus_out"][test]
        print(f"  {test:6s} " + " ".join(f"{bg:>6.2f}:{e[f'{bg:.2f}']['coverage']:.4f}"
                                          for bg in [12, 16, 22.4, 25.0]) +
              f"  static22.4={e['static_contiguous_coverage_22.40']:.4f}")
    print("\ntemporal (mean overlap of adjacent-token top-10, same layer):")
    for n in names:
        print(f"  {n:6s} persist={tem[n]['adjacent_persistence']:.4f} "
              f"static_top10_predictor={tem[n].get('static_top10_predictor', float('nan')):.4f}")
    print(f"  pooled persist={tem['pooled_vs_static_topk']['adjacent_persistence']:.4f} "
          f"static_top10_predictor={tem['pooled_vs_static_topk'].get('static_top10_predictor', float('nan')):.4f}")
    for n, e in result["decode"].items():
        print(f"\ndecode[{n}] (teacher-forced, {e['decode_tokens']} tokens):")
        print(f"  spearman(prefill,decode)={e['spearman_prefill_vs_decode']:.3f} "
              f"top10_overlap={e['top10_overlap_prefill_vs_decode']:.3f}")
        print(f"  decode persist={e['temporal']['adjacent_persistence']:.4f} "
              f"static_top10_predictor={e['temporal'].get('static_top10_predictor', float('nan')):.4f}")
        for bg in [16, 22.4]:
            t = e["transfer"][f"{bg:.2f}"]
            print(f"  budget {bg:5.2f} GiB: prefill->decode={t['prefill_profile_on_decode']:.4f} "
                  f"decode->prefill={t['decode_profile_on_prefill']:.4f} "
                  f"decode->decode={t['decode_profile_on_decode']:.4f}")
        print(f"  static22.40 on decode={e['static_contiguous_coverage_22.40']:.4f} "
              f"(n_cpu={e['static_contiguous_n_cpu_22.40']})")
        print(f"  layer47 decode top10_share={e['layer47_top10_share']:.4f} "
              f"vs mean(0..46)={e['mean_layer_top10_share']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
