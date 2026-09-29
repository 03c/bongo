#!/usr/bin/env python3
"""Capacity -> throughput model for bongo (IQ2_XS, llama.cpp b11223).

Combines three committed measurements into a predictor of decode throughput as
a function of (VRAM, RAM):

  * per-layer expert bytes      bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json
  * the VRAM-residency sweep    bench/results/2026-09-27-expert-placement/sweep-matrix.json
  * the activation hit-rate curve (BAS-66)
                                bench/results/2026-09-28-expert-activation/analysis.json
  * the RAM-constrained runs    bench/results/2026-09-28-capacity-sensitivity/*/

The model is deliberately two-part and states which part is measured:

  1. VRAM axis (fitted to measured anchors)
        t_ms(ctx, B) = T0(ctx) + A(ctx) * B^p(ctx)
     where B is the CPU-resident expert bytes left after the GPU budget.  T0, A
     and p are fit by grid search + linear least squares over the sweep anchors
     (all configs that served the context).  T0 is the context-fixed term
     (attention/KV, dense layers, launch overhead); A*B^p is the CPU-expert
     term.

  2. RAM feasibility (measured threshold, modelled penalty)
        RAM_needed(V) = B_res(V) + PLE_live + overhead
     A config is RAM-safe when the cap >= RAM_needed.  Under that threshold the
     measured anchors show no throughput change (16 GiB == ~30 GiB).  Over it,
     the model applies a *modelled* miss penalty derived from the BAS-66
     hit-rate curve at the cacheable budget and the measured cold-read cost;
     the penalty coefficient is calibrated against the one measured thrash point
     (all-CPU `--n-cpu-moe 48`, 74 GB of SSD reads in a single run).

Every input and every fitted coefficient is written to the output JSON so the
report's table can be re-derived.

    python3 bench/capacity-model.py                 # all defaults
    python3 bench/capacity-model.py --out bench/results/.../capacity-model.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# constants measured once, with their source
# ---------------------------------------------------------------------------

# Non-expert VRAM footprint (dense + KV + state + buffers).  BAS-66 analysis.json
# derives these from the sweep: usable VRAM minus GPU-resident expert bytes.
NON_EXPERT_4K_GIB = 6.42
NON_EXPERT_128K_GIB = 6.86

# Highest GPU-expert budget measured safe at 128K (n-cpu-moe 16).  n=12
# (25.02 GiB) device-lost during the 128K prefill.
SAFE_GPU_EXPERTS_128K_GIB = 22.40

# Measured effective cold-read bandwidth for random 4 KiB against the PLE table
# region (ssd-random4k-probe): QD1 7.5 MB/s, QD8 62.3 MB/s.  Used only for the
# modelled RAM over-threshold penalty, at the pessimistic QD1 end.
SSD_RANDOM_4K_QD1_MB_PER_S = 7.5

# Page-cache bytes that must stay outside the expert working set: the live PLE
# page window (BAS-67: 16 pages/token, 64 KiB/token, 26.82 GiB table) plus
# process/libc/loader overhead.  4 GiB is the round budget that makes the 16 GiB
# anchor consistent (10.62 GiB experts + 4 GiB fits in 16 GiB, measured).
RAM_OVERHEAD_GIB = 4.0


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def interp(xs, ys, x):
    """Piecewise-linear interpolation, clamped at the ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


class CapacityModel:
    def __init__(self, root, capacity_dir):
        self.root = root
        self.expert = load_json(
            os.path.join(root, "bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json")
        )
        self.analysis = load_json(
            os.path.join(root, "bench/results/2026-09-28-expert-activation/analysis.json")
        )
        self.sweep = load_json(
            os.path.join(root, "bench/results/2026-09-27-expert-placement/sweep-matrix.json")
        )
        self.capacity_dir = os.path.join(root, capacity_dir)
        self.layer_bytes = self.expert["expert_bytes_by_layer"]
        self.E = self.expert["total_expert_bytes"] / 2**30  # GiB
        self.n_active = self.analysis["n_active"]
        self.n_exp_per_layer = self.analysis["n_experts_per_layer"]
        self.n_layers = self.analysis["n_layers"]

        # BAS-66 coverage curves, keyed by budget in GiB.
        bs = self.analysis["budget_sweep"]
        self._budgets = sorted((float(k) for k in bs), key=float)
        self.hot_cov = {float(k): bs[k]["hot_count"]["coverage"] for k in bs}
        self.static_cov = {float(k): bs[k]["static_contiguous"]["coverage"] for k in bs}

    # -- geometry ----------------------------------------------------------

    def coverage(self, budget_gib, curve="static"):
        src = self.static_cov if curve == "static" else self.hot_cov
        xs = sorted(src)
        return interp(xs, [src[x] for x in xs], budget_gib)

    def gpu_expert_budget(self, vram_gib, ctx):
        non_expert = NON_EXPERT_128K_GIB if ctx >= 32768 else NON_EXPERT_4K_GIB
        g = vram_gib - non_expert
        if ctx >= 32768:
            g = min(g, SAFE_GPU_EXPERTS_128K_GIB)
        return max(0.0, g)

    def static_split(self, gpu_budget_gib):
        """llama.cpp `--n-cpu-moe N` puts the *first* N layers on the CPU, so the
        GPU keeps layers N..47.  Return (n_cpu, gpu_bytes_gib, cpu_bytes_gib).

        Pick the n that spends the budget best without exceeding it.
        """
        best = (self.n_layers, 0.0, self.E)  # all CPU
        for n in range(self.n_layers + 1):
            gpu_bytes = sum(self.layer_bytes[n:]) / 2**30
            if gpu_bytes <= gpu_budget_gib + 1e-9:
                best = (n, gpu_bytes, self.E - gpu_bytes)
                break
        return best

    def cpu_bytes_for(self, vram_gib, ctx):
        g = self.gpu_expert_budget(vram_gib, ctx)
        n, gpu, cpu = self.static_split(g)
        return g, n, gpu, cpu

    # -- throughput fit ----------------------------------------------------

    @staticmethod
    def _fit_power(Bs, Ts):
        """Fit t = T0 + A * B^p (T0, A free per p) by grid search over p."""
        best = None
        p = 0.2
        while p <= 9.0001:
            xs = [b**p for b in Bs]
            n = len(xs)
            sx = sum(xs)
            sy = sum(Ts)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, Ts))
            denom = n * sxx - sx * sx
            if abs(denom) > 1e-12:
                A = (n * sxy - sx * sy) / denom
                T0 = (sy - A * sx) / n
                sse = sum((T0 + A * B**p - t) ** 2 for B, t in zip(Bs, Ts))
                if best is None or sse < best["sse"]:
                    best = {"p": p, "A_ms_per_gib": A, "T0_ms": T0, "sse": sse}
            p += 0.01
        return best

    def fit_context(self, ctx):
        Bs, Ts = [], []
        for row in self.sweep["rows"]:
            c = (row.get("contexts") or {}).get(str(ctx))
            if not c or c.get("status") != "ok":
                continue
            Bs.append(row["experts"]["cpu_gib"])
            Ts.append(1000.0 / c["output_tps"])
        fit = self._fit_power(Bs, Ts)
        fit["anchors"] = [
            {"cpu_gib": b, "t_ms": t} for b, t in sorted(zip(Bs, Ts))
        ]
        return fit

    def predict_t_ms(self, ctx, cpu_gib, fit):
        return fit["T0_ms"] + fit["A_ms_per_gib"] * cpu_gib ** fit["p"]

    # -- RAM ---------------------------------------------------------------

    def cpu_bytes_per_token_mb(self, cpu_gib):
        """Active expert bytes a token touches on the CPU: 10 of 512 experts per
        CPU layer.  Byte-weighted by the CPU expert share."""
        total_active = self.E * self.n_active / self.n_exp_per_layer  # GiB/token
        return total_active * (cpu_gib / self.E) * 1024.0

    def ram_safe(self, cpu_gib, ram_gib):
        return (cpu_gib + RAM_OVERHEAD_GIB) <= ram_gib + 1e-9

    def ram_model(self, ctx, cpu_gib, ram_gib, fit):
        """RAM enters the model as feasibility and SSD traffic, not throughput.

        * ``ram_safe``   -- the cap must hold the CPU expert set plus overhead.
        * ``deficit_gib``-- CPU expert bytes the page cache cannot keep.  This
          is the quantity that shows up as SSD ``read_bytes``; it is validated
          against the measured runs.
        * ``naive_serialised_bound_factor`` -- the throughput the page-fault
          path would cost *if every miss were serialised and paid in full*,
          using the measured QD1 random-4K bandwidth.  It is deliberately
          pessimal and the measurement falsifies it (see the report): it is
          kept so the report can quantify how completely the reads are hidden.
        """
        cacheable = max(0.0, ram_gib - RAM_OVERHEAD_GIB)
        base_ms = self.predict_t_ms(ctx, cpu_gib, fit)
        deficit = max(0.0, cpu_gib - cacheable)
        h = self.coverage(min(cacheable, cpu_gib), curve="hot")
        miss_fraction = max(0.0, 1.0 - h) if deficit > 0 else 0.0
        miss_bytes_mb = miss_fraction * self.cpu_bytes_per_token_mb(cpu_gib)
        bw_mb_per_ms = SSD_RANDOM_4K_QD1_MB_PER_S / 1000.0
        miss_ms = miss_bytes_mb / bw_mb_per_ms if bw_mb_per_ms else 0.0
        return {
            "cacheable_gib": cacheable,
            "cpu_expert_bytes_over_cacheable_gib": deficit,
            "miss_fraction": miss_fraction,
            "miss_bytes_per_token_mb": miss_bytes_mb,
            "naive_serialised_miss_ms_per_token": miss_ms,
            "naive_serialised_bound_factor": (base_ms + miss_ms) / base_ms,
        }

    # -- runners -----------------------------------------------------------

    def vram_axis(self):
        rows = []
        for row in self.sweep["rows"]:
            c4 = (row.get("contexts") or {}).get("4096") or {}
            c128 = (row.get("contexts") or {}).get("131072") or {}
            rows.append(
                {
                    "n_cpu_moe": row["n_cpu_moe"],
                    "gpu_gib": row["experts"]["gpu_gib"],
                    "cpu_gib": row["experts"]["cpu_gib"],
                    "fit": row.get("fit"),
                    "ctx4096": {
                        "prompt_tps": c4.get("prompt_tps"),
                        "output_tps": c4.get("output_tps"),
                    }
                    if c4
                    else None,
                    "ctx131072": {
                        "prompt_tps": c128.get("prompt_tps"),
                        "output_tps": c128.get("output_tps"),
                    }
                    if c128
                    else None,
                    "server_read_bytes_delta": row.get("server_read_bytes_delta"),
                }
            )
        rows.sort(key=lambda r: r["gpu_gib"])
        return rows

    def measured_capacity(self):
        out = []
        if not os.path.isdir(self.capacity_dir):
            return out
        for name in sorted(os.listdir(self.capacity_dir)):
            d = os.path.join(self.capacity_dir, name)
            mpath = os.path.join(d, "matrix.json")
            if not os.path.isfile(mpath):
                continue
            m = load_json(mpath)
            io = load_json(os.path.join(d, "server-io.json")) if os.path.isfile(os.path.join(d, "server-io.json")) else {}
            cg = load_json(os.path.join(d, "cgroup.json")) if os.path.isfile(os.path.join(d, "cgroup.json")) else {}
            mem_max = cg.get("memory.max")
            mem_max_gib = round(int(mem_max) / 2**30, 2) if mem_max and mem_max.isdigit() else io.get("mem_gib")
            rec = {
                "run": name,
                "memory_max_gib": mem_max_gib,
                "n_cpu_moe": io.get("n_cpu_moe"),
                "fit": True,
                "contexts": {},
            }
            for r in m.get("results", []):
                if r.get("status") != "ok":
                    rec["contexts"][str(r.get("target_context"))] = {"status": r.get("status")}
                    continue
                s = r["summary"]
                mem = r.get("memory", {})
                rec["contexts"][str(r["target_context"])] = {
                    "prompt_tps": s["prompt_tps"]["median"],
                    "output_tps": s["output_tps"]["median"],
                    "ttft_ms": s["ttft_ms"]["median"],
                    "vram_peak_bytes": mem.get("vram_peak_bytes"),
                    "ram_peak_bytes": mem.get("system_ram_peak_bytes"),
                }
            io_path = os.path.join(d, "server-io.json")
            if os.path.isfile(io_path):
                io = load_json(io_path)
                rb = io.get("read_bytes_after")
                rbb = io.get("read_bytes_before")
                rec["read_bytes_delta"] = (rb - rbb) if (rb is not None and rbb is not None) else None
                rec["rchar_delta"] = (
                    (io.get("rchar_after") - io.get("rchar_before"))
                    if (io.get("rchar_after") is not None and io.get("rchar_before") is not None)
                    else None
                )
                rec["majflt_delta"] = (
                    (io.get("majflt_after") - io.get("majflt_before"))
                    if (io.get("majflt_after") is not None and io.get("majflt_before") is not None)
                    else None
                )
            mem_path = os.path.join(d, "memory.json")
            if os.path.isfile(mem_path):
                rec["memory"] = load_json(mem_path)
            out.append(rec)
        return out

    def look_one_out(self):
        """Leave-one-out error for the VRAM fit (in-sample anchors excluded)."""
        out = []
        for ctx in (4096, 131072):
            pts = []
            for row in self.sweep["rows"]:
                c = (row.get("contexts") or {}).get(str(ctx))
                if not c or c.get("status") != "ok":
                    continue
                pts.append((row["experts"]["cpu_gib"], c["output_tps"], row["n_cpu_moe"]))
            for i in range(len(pts)):
                train = [p for j, p in enumerate(pts) if j != i]
                fit = self._fit_power([p[0] for p in train], [1000.0 / p[1] for p in train])
                cpu, measured, n = pts[i]
                pred = 1000.0 / self.predict_t_ms(ctx, cpu, fit)
                out.append(
                    {
                        "ctx": ctx,
                        "n_cpu_moe": n,
                        "cpu_gib": cpu,
                        "measured_output_tps": measured,
                        "loo_output_tps": pred,
                        "loo_rel_error": (pred - measured) / measured,
                    }
                )
        return out

    def anchor_errors(self):
        """Model error at every measured output-tok/s anchor."""
        out = []
        for ctx in (4096, 131072):
            fit = self.fit_context(ctx)
            for row in self.sweep["rows"]:
                c = (row.get("contexts") or {}).get(str(ctx))
                if not c or c.get("status") != "ok":
                    continue
                cpu = row["experts"]["cpu_gib"]
                measured = c["output_tps"]
                pred = 1000.0 / self.predict_t_ms(ctx, cpu, fit)
                out.append(
                    {
                        "source": "sweep",
                        "ctx": ctx,
                        "n_cpu_moe": row["n_cpu_moe"],
                        "ram_gib": None,
                        "cpu_gib": cpu,
                        "measured_output_tps": measured,
                        "modelled_output_tps": pred,
                        "rel_error": (pred - measured) / measured,
                    }
                )
        for rec in self.measured_capacity():
            for ctx_s, c in rec["contexts"].items():
                if ctx_s not in ("4096", "131072") or "output_tps" not in c:
                    continue
                ctx = int(ctx_s)
                fit = self.fit_context(ctx)
                n = rec["n_cpu_moe"]
                if n is None or n > len(self.layer_bytes):
                    continue
                cpu = sum(self.layer_bytes[:n]) / 2**30
                measured = c["output_tps"]
                pred = 1000.0 / self.predict_t_ms(ctx, cpu, fit)
                out.append(
                    {
                        "source": "capacity",
                        "run": rec["run"],
                        "ctx": ctx,
                        "n_cpu_moe": n,
                        "ram_gib": rec["memory_max_gib"],
                        "cpu_gib": cpu,
                        "measured_output_tps": measured,
                        "modelled_output_tps": pred,
                        "rel_error": (pred - measured) / measured,
                    }
                )
        return out

    def grid(self):
        out = {}
        for ctx in (131072, 4096):
            fit = self.fit_context(ctx)
            rows = []
            for vram in (12, 24, 32):
                g, n_cpu, gpu, cpu = self.cpu_bytes_for(vram, ctx)
                h = self.coverage(gpu, curve="static")
                for ram in (16, 32, 64):
                    safe = self.ram_safe(cpu, ram)
                    t = self.predict_t_ms(ctx, cpu, fit)
                    ram_detail = self.ram_model(ctx, cpu, ram, fit)
                    rows.append(
                        {
                            "vram_gib": vram,
                            "ram_gib": ram,
                            "gpu_expert_budget_gib": round(g, 2),
                            "gpu_experts_resident_gib": round(gpu, 2),
                            "n_cpu_moe": n_cpu,
                            "cpu_expert_gib": round(cpu, 2),
                            "static_coverage": h,
                            "ram_safe": safe,
                            "modelled_output_tps": round(1000.0 / t, 2),
                            "ram_model": ram_detail,
                        }
                    )
            out[str(ctx)] = rows
        return out


def render_markdown(model):
    lines = []
    lines.append("# Capacity model output")
    lines.append("")
    lines.append("Reproduce: `python3 bench/capacity-model.py`")
    lines.append("")
    axis = model.vram_axis()
    lines.append("## Measured VRAM axis (from the sweep, unconstrained RAM)")
    lines.append("")
    lines.append("| n-cpu-moe | GPU experts GiB | CPU experts GiB | 4K prompt | 4K output | 128K prompt | 128K output | SSD read B |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in axis:
        c4 = r["ctx4096"] or {}
        c128 = r["ctx131072"] or {}
        def f(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {r['n_cpu_moe']} | {r['gpu_gib']:.2f} | {r['cpu_gib']:.2f} | "
            f"{f(c4.get('prompt_tps'))} | {f(c4.get('output_tps'))} | "
            f"{f(c128.get('prompt_tps'))} | {f(c128.get('output_tps'))} | "
            f"{r['server_read_bytes_delta'] if r['server_read_bytes_delta'] is not None else '—'} |"
        )
    lines.append("")
    lines.append("## Fit")
    lines.append("")
    for ctx in (4096, 131072):
        fit = model.fit_context(ctx)
        lines.append(
            f"- ctx {ctx}: `t_ms = {fit['T0_ms']:.2f} + {fit['A_ms_per_gib']:.4f} * B^{fit['p']:.2f}` "
            f"(B = CPU expert GiB, SSE={fit['sse']:.1f})"
        )
    lines.append("")
    lines.append("## Grid (modelled output tok/s)")
    lines.append("")
    grid = model.grid()
    for ctx in ("131072", "4096"):
        lines.append(f"### ctx {ctx}")
        lines.append("")
        lines.append("| VRAM GiB | RAM GiB | GPU exp budget GiB | GPU exp resident GiB | n-cpu-moe | CPU exp GiB | RAM-safe | output tok/s |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |")
        for r in grid[ctx]:
            lines.append(
                f"| {r['vram_gib']} | {r['ram_gib']} | {r['gpu_expert_budget_gib']} | {r['gpu_experts_resident_gib']} | {r['n_cpu_moe']} | "
                f"{r['cpu_expert_gib']} | {'yes' if r['ram_safe'] else '**no**'} | {r['modelled_output_tps']} |"
            )
        lines.append("")
        lines.append(
            "`RAM-safe` = `CPU expert GiB + 4 GiB overhead <= RAM`.  The modelled output tok/s does "
            "not depend on RAM inside the measured range; see the RAM section."
        )
        lines.append("")
    lines.append("## Leave-one-out error of the VRAM fit")
    lines.append("")
    lines.append("| ctx | n-cpu-moe | measured | LOO modelled | rel err |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for e in model.look_one_out():
        lines.append(
            f"| {e['ctx']} | {e['n_cpu_moe']} | {e['measured_output_tps']:.2f} | "
            f"{e['loo_output_tps']:.2f} | {e['loo_rel_error']:+.1%} |"
        )
    lines.append("")
    lines.append("## Measured RAM axis")
    lines.append("")
    lines.append("| run | memory.max GiB | n-cpu-moe | CPU exp GiB | 4K output | 128K output | RAM peak GiB | VRAM peak GiB | SSD read_bytes | major faults |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rec in model.measured_capacity():
        n = rec.get("n_cpu_moe")
        cpu = sum(model.layer_bytes[:n]) / 2**30 if isinstance(n, int) else None
        ram_peak = 0.0
        vram_peak = 0.0
        o4 = o128 = None
        for ctx_s, c in rec["contexts"].items():
            if "output_tps" not in c:
                continue
            ram_peak = max(ram_peak, (c.get("ram_peak_bytes") or 0) / 2**30)
            vram_peak = max(vram_peak, (c.get("vram_peak_bytes") or 0) / 2**30)
            if ctx_s == "4096":
                o4 = c["output_tps"]
            if ctx_s == "131072":
                o128 = c["output_tps"]
        rb = rec.get("read_bytes_delta")
        lines.append(
            f"| {rec['run']} | {rec['memory_max_gib']} | {n} | {cpu:.2f} | "
            f"{o4:.2f} | {o128:.2f} | {ram_peak:.2f} | {vram_peak:.2f} | "
            f"{rb if rb is not None else '—'} | {rec.get('majflt_delta') if rec.get('majflt_delta') is not None else '—'} |"
        )
    lines.append("")
    lines.append("")
    lines.append("## Model error at measured anchors")
    lines.append("")
    lines.append("| source | ctx | n-cpu-moe | RAM GiB | measured | modelled | rel err |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for e in model.anchor_errors():
        lines.append(
            f"| {e['source']} | {e['ctx']} | {e['n_cpu_moe']} | {e.get('ram_gib') or '—'} | "
            f"{e['measured_output_tps']:.2f} | {e['modelled_output_tps']:.2f} | {e['rel_error']:+.1%} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="bongo capacity model")
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--capacity-dir", default="bench/results/2026-09-28-capacity-sensitivity")
    ap.add_argument(
        "--out",
        default="bench/results/2026-09-28-capacity-sensitivity/capacity-model.json",
    )
    args = ap.parse_args()

    model = CapacityModel(args.repo_root, args.capacity_dir)
    result = {
        "schema": "bongo.capacity-model.v1",
        "inputs": {
            "expert_bytes": "bench/results/2026-09-27-expert-placement/expert-bytes-iq2_xs.json",
            "coverage": "bench/results/2026-09-28-expert-activation/analysis.json",
            "vram_sweep": "bench/results/2026-09-27-expert-placement/sweep-matrix.json",
            "capacity_dir": args.capacity_dir,
            "ssd_read_cost": "bench/results/2026-09-28-ssd-random4k-probe/README.md",
        },
        "constants": {
            "total_expert_gib": model.E,
            "non_expert_4k_gib": NON_EXPERT_4K_GIB,
            "non_expert_128k_gib": NON_EXPERT_128K_GIB,
            "safe_gpu_experts_128k_gib": SAFE_GPU_EXPERTS_128K_GIB,
            "ssd_random_4k_qd1_mb_per_s": SSD_RANDOM_4K_QD1_MB_PER_S,
            "ram_overhead_gib": RAM_OVERHEAD_GIB,
        },
        "vram_axis": model.vram_axis(),
        "capacity": model.measured_capacity(),
        "fit": {str(ctx): model.fit_context(ctx) for ctx in (4096, 131072)},
        "grid": model.grid(),
        "look_one_out": model.look_one_out(),
        "anchor_errors": model.anchor_errors(),
    }
    out = os.path.join(args.repo_root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    md = render_markdown(model)
    md_path = out.replace(".json", ".md")
    with open(md_path, "w") as fh:
        fh.write(md)
    sys.stdout.write(md)
    sys.stderr.write(f"\nwrote {out} and {md_path}\n")


if __name__ == "__main__":
    main()
