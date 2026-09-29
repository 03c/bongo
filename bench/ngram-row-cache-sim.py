#!/usr/bin/env python3
"""Simulate PLE/n-gram row-cache hit rate on real token sequences.

Reimplements the exact row index that llama.cpp `qwen4exp` computes for
`per_layer_token_embd.weight` (``src/models/qwen4exp.cpp``,
``llm_graph_input_ple::set_input``): for every token, 16 rows are selected from
two n-gram families (a bigram family of 8 heads and a trigram family of 8 heads)
by hashing the token and its two predecessors with the model's
``ple.layer_multipliers`` and reducing modulo each head's vocabulary size.

Given tokenized prompts, the tool reports, per prompt and for a cache shared
across prompts:

  * rows touched, the distinct 4 KiB pages they fall on, and the "intra-prompt
    recurrence" (share of row touches whose row was already touched earlier in
    the same prompt),
  * LRU hit rate for a range of caches, from 25k rows (~2.25 MB) to 2M rows
    (~180 MB).

Input token files are the JSON written next to this doc, one per prompt::

    {"prompt": "prose", "n_tokens": 5861, "ids": [...]}

Example::

    bench/ngram-row-cache-sim.py \\
        --inventory bench/results/<run>/raw/gguf-inventory-iq2xs.json \\
        --tokens bench/results/<run>/raw/tokens-*.json \\
        --out bench/results/<run>/raw/ngram-row-cache.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

U64 = (1 << 64) - 1
HEADS_PER_GRAM = 8


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def load_ple_params(inventory_path: str) -> Dict:
    with open(inventory_path) as fh:
        inv = json.load(fh)
    meta = inv["metadata"]

    def arr(key: str) -> List[int]:
        return meta[f"qwen4exp.ple.{key}"]

    tensor = next(t for t in inv["tensors"] if t["name"] == "per_layer_token_embd.weight")
    shard = next(s for s in inv["shards"] if s["file"] == tensor["shard"])
    return {
        "multipliers": arr("layer_multipliers"),
        "head_offsets": arr("head_offsets"),
        "head_vocab_sizes": arr("head_vocab_sizes"),
        "eos_token_id": meta["qwen4exp.ple.eos_token_id"],
        "n_heads": meta["qwen4exp.ple.ngram_size"] - 1,
        "heads_per_gram": meta["qwen4exp.ple.heads_per_gram"] if "heads_per_gram" in meta
                          else meta["qwen4exp.ple.heads_per_ngram"],
        "rows": tensor["dims"][1],
        "row_bytes": tensor["bytes"] // tensor["dims"][1],
        "tensor_file_offset": shard["data_section_offset"] + tensor["offset"],
    }


def token_rows(ids: Sequence[int], params: Dict) -> List[List[int]]:
    """The 16 row indices each token reads, in head order (0..15)."""
    mult = params["multipliers"]
    eos = params["eos_token_id"]
    per_gram = params["heads_per_gram"]
    offsets = params["head_offsets"]
    vocab = params["head_vocab_sizes"]
    n_gram = len(mult)

    out: List[List[int]] = []
    for i, tok in enumerate(ids):
        ctx = [tok]
        cut = False
        for s in range(1, n_gram):
            t = ids[i - s] if i - s >= 0 else None
            if cut or t is None or t == eos:
                cut = True
                ctx.append(eos)
            else:
                ctx.append(t)
        rows = [0] * (per_gram * (n_gram - 1))
        for n in range(2, n_gram + 1):
            mixed = ctx[0] * mult[0] & U64
            for j in range(1, n):
                mixed ^= (ctx[j] * mult[j]) & U64
            base = (n - 2) * per_gram
            for g in range(per_gram):
                h = base + g
                rows[h] = mixed % vocab[h] + offsets[h]
        out.append(rows)
    return out


class LRU:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: "OrderedDict[int, None]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def touch(self, key: int) -> bool:
        if self.capacity <= 0:
            self.misses += 1
            return False
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        self.cache[key] = None
        self.misses += 1
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return False


def page_stats(rows_per_token: Sequence[Sequence[int]], tensor_file_offset: int,
               row_bytes: int, page: int = 4096) -> Dict:
    """Distinct 4 KiB pages touched per token (rows are 90 B apart, so pages are shared)."""
    counts: List[int] = []
    total_pages = 0
    for rows in rows_per_token:
        pages = {(tensor_file_offset + r * row_bytes) // page for r in rows}
        counts.append(len(pages))
        total_pages += len(pages)
    counts.sort()
    return {
        "distinct_pages_per_token_mean": statistics.fmean(counts) if counts else 0.0,
        "distinct_pages_per_token_p50": percentile(counts, 50),
        "distinct_pages_per_token_p90": percentile(counts, 90),
        "distinct_pages_per_token_max": counts[-1] if counts else 0,
        "pages_total": total_pages,
        "page_traffic_per_token_bytes": (total_pages / len(counts)) * page if counts else 0.0,
    }


def intra_prompt(rows_per_token: Sequence[Sequence[int]]) -> Dict:
    """Recurrence inside one prompt, no capacity limit (upper bound on any cache)."""
    seen: set[int] = set()
    touches = 0
    repeats = 0
    for rows in rows_per_token:
        for r in rows:
            touches += 1
            if r in seen:
                repeats += 1
            else:
                seen.add(r)
    return {
        "row_touches": touches,
        "unique_rows": len(seen),
        "recurrence": repeats / touches if touches else 0.0,
        "compulsory": len(seen) / touches if touches else 0.0,
    }


def simulate(sequences: Sequence[Dict], capacities: Sequence[int], row_bytes: int,
             tensor_file_offset: int) -> Dict:
    """Per-prompt and shared-cache LRU hit rates."""
    per_prompt = {}
    for seq in sequences:
        per_prompt[seq["prompt"]] = {
            "n_tokens": seq["n_tokens"],
            **page_stats(seq["rows"], tensor_file_offset, row_bytes),
            **intra_prompt(seq["rows"]),
        }

    shared = {}
    for cap in capacities:
        lru = LRU(cap)
        per = {}
        totals_hits = totals_misses = 0
        for seq in sequences:
            lru.hits = lru.misses = 0
            for rows in seq["rows"]:
                for r in rows:
                    lru.touch(r)
            per[seq["prompt"]] = lru.hits / (lru.hits + lru.misses)
            totals_hits += lru.hits
            totals_misses += lru.misses
        shared[str(cap)] = {
            "capacity_rows": cap,
            "capacity_bytes": cap * row_bytes,
            "hit_rate": totals_hits / (totals_hits + totals_misses),
            "per_prompt": per,
        }
    return {"per_prompt": per_prompt, "shared_lru": shared}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--tokens", nargs="+", required=True, help="token JSON files (globs allowed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files: List[str] = []
    for pat in args.tokens:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit("no token files matched")

    params = load_ple_params(args.inventory)
    sequences = []
    for path in files:
        with open(path) as fh:
            rec = json.load(fh)
        sequences.append({
            "prompt": rec["prompt"],
            "source": os.path.basename(path),
            "n_tokens": rec["n_tokens"],
            "rows": token_rows(rec["ids"], params),
        })

    capacities = [0, 25_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
    result = {
        "meta": {
            "inventory": os.path.abspath(args.inventory),
            "row_bytes": params["row_bytes"],
            "rows": params["rows"],
            "tensor_file_offset": params["tensor_file_offset"],
            "prompts": [s["source"] for s in sequences],
            "n_heads": len(params["head_offsets"]),
            "heads_per_gram": params["heads_per_gram"],
            "multipliers": params["multipliers"],
            "eos_token_id": params["eos_token_id"],
        },
        "simulation": simulate(sequences, capacities, params["row_bytes"],
                               params["tensor_file_offset"]),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"wrote {args.out}")
    for name, p in result["simulation"]["per_prompt"].items():
        print(f"  {name:6s} tokens={p['n_tokens']:6d} unique_rows={p['unique_rows']:7d} "
              f"recurrence={p['recurrence'] * 100:5.1f}% "
              f"pages/token={p['distinct_pages_per_token_mean']:5.2f}")
    print("  shared LRU hit rate:")
    for cap, s in result["simulation"]["shared_lru"].items():
        print(f"    {int(cap):>9,} rows ({s['capacity_bytes'] / 1e6:7.1f} MB): "
              f"{s['hit_rate'] * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
