#!/usr/bin/env python3
"""Generate a cheapest-layer-first byte-budget `-ot` expert placement (BAS-76, Step 1).

The Stage-1 rule is `--n-cpu-moe N`: keep the experts of the first N layers on
the CPU and the remaining layers resident in VRAM.  That rule is a poor byte
allocator: it reserves the near-empty layer 47 in prefill and it keeps the
byte-expensive layers resident, so fewer layers fit than the byte budget allows
(R4, `docs/research/expert-activation-skew.md` section 2).

This tool replaces the layer-count rule with a byte budget:

  * layers are sorted cheapest-expert-bytes-first (ties broken by more activation
    events first, then by layer index, so the near-empty layer 47 is dropped);
  * a layer is kept resident while the cumulative *expert* bytes fit the budget;
  * the complement (the byte-expensive layers) is offloaded to the CPU with a
    single llama.cpp `--override-tensor` / `-ot` pattern.

It emits a JSON placement spec plus the exact `-ot` argument.  The coverage
column is the fraction of measured router events served from the resident set,
computed from the R4 per-layer activation counts (`analysis.json`).  Whole-layer
residency means per-layer counts are exact, so no raw capture replay is needed.

Stdlib only.

  python3 bench/gen-ot-placement.py \
      --expert-bytes bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json \
      --analysis bench/results/2026-09-28-expert-activation/analysis.json \
      --budget-gib 22.40 \
      --out bench/results/2026-09-28-byte-budget-placement/placement-iq2-xs-22.40.json
"""

from __future__ import annotations

import argparse
import json
import sys

N_LAYERS = 48
N_EXPERTS = 512
GIB = 1024 ** 3
BYTES_PER_MIB = 1024 ** 2


def load_layer_bytes(path: str) -> list[int]:
    with open(path) as fh:
        data = json.load(fh)
    layer_bytes = data["expert_bytes_by_layer"]
    if len(layer_bytes) != N_LAYERS:
        raise ValueError(f"{path}: expected {N_LAYERS} layers, got {len(layer_bytes)}")
    return [int(b) for b in layer_bytes]


def load_layer_events(path: str | None) -> list[int] | None:
    """Per-layer routed-event counts from the R4 analysis, or None."""
    if not path:
        return None
    with open(path) as fh:
        data = json.load(fh)
    per = data.get("concentration_pooled", {}).get("per_layer")
    if not per:
        return None
    return [int(per[str(layer)]["events"]) for layer in range(N_LAYERS)]


def choose_resident(layer_bytes: list[int], layer_events: list[int] | None,
                    budget_bytes: int) -> tuple[list[int], int]:
    """Return (resident_layers_sorted, used_bytes) under a cheapest-first budget."""
    events = layer_events or [0] * N_LAYERS
    # Cheapest expert bytes first; on a byte tie, prefer the layer with more
    # routed events (drops the near-empty layer 47); finally lowest index.
    order = sorted(range(N_LAYERS), key=lambda l: (layer_bytes[l], -events[l], l))
    resident: list[int] = []
    used = 0
    for layer in order:
        if used + layer_bytes[layer] <= budget_bytes:
            resident.append(layer)
            used += layer_bytes[layer]
    return sorted(resident), used


def cpu_pattern(cpu_layers: list[int]) -> str:
    """A llama.cpp `-ot` value that sends exactly the listed layers' experts to CPU."""
    alt = "|".join(str(layer) for layer in sorted(cpu_layers))
    return rf"blk\.({alt})\.ffn_(gate|up|down)_exps\.weight=CPU"


def coverage(events: list[int] | None, resident: list[int]) -> float | None:
    if not events:
        return None
    total = sum(events)
    if total == 0:
        return None
    return sum(events[layer] for layer in resident) / total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert-bytes", required=True)
    ap.add_argument("--analysis", default=None,
                    help="R4 analysis.json with concentration_pooled.per_layer (for coverage)")
    ap.add_argument("--budget-gib", type=float, default=22.40,
                    help="resident expert byte budget in GiB (default: measured-safe 22.40)")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-arg", action="store_true",
                    help="print only the -ot argument and exit")
    args = ap.parse_args(argv)

    layer_bytes = load_layer_bytes(args.expert_bytes)
    layer_events = load_layer_events(args.analysis)
    total_bytes = sum(layer_bytes)
    budget_bytes = int(round(args.budget_gib * GIB))

    resident, used = choose_resident(layer_bytes, layer_events, budget_bytes)
    cpu_layers = [l for l in range(N_LAYERS) if l not in set(resident)]
    pattern = cpu_pattern(cpu_layers)

    if args.print_arg:
        print(pattern)
        return 0

    cov = coverage(layer_events, resident)
    # Static --n-cpu-moe 16 baseline at the same budget, for the delta.
    static16 = coverage(layer_events, list(range(16, N_LAYERS)))
    static16_bytes = sum(layer_bytes[16:])

    spec = {
        "schema": "bongo.ot-placement.v1",
        "tier": args.tier or (json.load(open(args.expert_bytes)).get("tier")),
        "policy": "cheapest-layer-first byte budget (complement offloaded to CPU)",
        "budget_gib": args.budget_gib,
        "budget_bytes": budget_bytes,
        "total_expert_bytes": total_bytes,
        "total_expert_gib": total_bytes / GIB,
        "resident_layers": resident,
        "n_resident_layers": len(resident),
        "cpu_layers": cpu_layers,
        "n_cpu_layers": len(cpu_layers),
        "resident_expert_bytes": used,
        "resident_expert_gib": used / GIB,
        "cpu_expert_bytes": total_bytes - used,
        "cpu_expert_gib": (total_bytes - used) / GIB,
        "override_tensor": pattern,
        "coverage": {
            "resident_fraction_of_events": cov,
            "static_n_cpu_moe_16": static16,
            "delta_pp": (None if cov is None or static16 is None else (cov - static16) * 100.0),
            "static_n_cpu_moe_16_expert_bytes": static16_bytes,
            "baseline": "R4 concentration_pooled.per_layer.events (whole-layer residency, exact)",
        },
        "engine_revision": "llama.cpp b11223 (4da6337767f973e2b4d0797e5b323d77d8565e4a)",
    }

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(spec, fh, indent=2, sort_keys=True)
        print(f"wrote {args.out}")

    print(f"tier={spec['tier']} budget={args.budget_gib:.2f} GiB "
          f"resident={len(resident)} layers ({used/GIB:.3f} GiB) "
          f"cpu={len(cpu_layers)} layers ({(total_bytes-used)/GIB:.3f} GiB)")
    if cov is not None:
        print(f"coverage resident={cov:.4f} static(n=16)={static16:.4f} "
              f"delta={spec['coverage']['delta_pp']:+.2f} pp")
    print(f"-ot {pattern}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
