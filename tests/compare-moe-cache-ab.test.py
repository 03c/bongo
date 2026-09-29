#!/usr/bin/env python3
"""Regression test for bench/compare-moe-cache-ab.py (BAS-139).

Synthetic matrices + prefix-cache runs exercise the acceptance logic: a +20%
4K decode gain passes; a >2% 128K decode regression fails; a +15% turn TTFT
gain passes without the decode gain. No GPU, no network.

  python3 tests/compare-moe-cache-ab.test.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CMP = os.path.join(ROOT, "bench", "compare-moe-cache-ab.py")

FAIL = 0


def check(cond, msg):
    global FAIL
    print(("ok   - " if cond else "FAIL - ") + msg)
    if not cond:
        FAIL += 1


def matrix(ctx, prompt_tps, output_tps, ttft_ms, needle="pass"):
    return {
        "schema": "bongo-bench/1",
        "results": [{
            "target_context": ctx,
            "status": 200,
            "summary": {
                "prompt_tps": {"median": prompt_tps},
                "output_tps": {"median": output_tps},
                "ttft_ms": {"median": ttft_ms},
            },
            "memory": {"vram_peak_bytes": 30 * 1024**3},
        }],
        "needle": {"status": needle},
    }


def prefix(grow_ttft_ms):
    return {"runs": [
        {"label": "cold_p4096", "ttft_ms": 23000.0, "output_tps": 13.0, "status": 200},
        {"label": "grow_p4096_d512", "ttft_ms": grow_ttft_ms, "output_tps": 14.0, "status": 200},
    ]}


def run(tmp, a, b, ap, bp):
    for name, obj in (("a.json", a), ("b.json", b), ("ap.json", ap), ("bp.json", bp)):
        with open(os.path.join(tmp, name), "w") as fh:
            json.dump(obj, fh)
    out = os.path.join(tmp, "out")
    p = subprocess.run([sys.executable, CMP,
                        "--a", os.path.join(tmp, "a.json"),
                        "--b", os.path.join(tmp, "b.json"),
                        "--a-prefix", os.path.join(tmp, "ap.json"),
                        "--b-prefix", os.path.join(tmp, "bp.json"),
                        "--out", out], capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr
    return json.load(open(os.path.join(out, "moe-cache-ab.json"))), ""


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="moe-cache-ab-test-")

    # base: A = LRU, B = stage 0. 4K decode +20%; no 128K regression.
    b_4k, b_128 = matrix(4096, 170.0, 10.0, 23000.0), matrix(131072, 135.0, 8.0, 900000.0)
    a_4k, a_128 = matrix(4096, 172.0, 12.0, 22500.0), matrix(131072, 136.0, 8.05, 895000.0)
    b = {"results": b_4k["results"] + b_128["results"], "needle": b_4k["needle"]}
    a = {"results": a_4k["results"] + a_128["results"], "needle": a_4k["needle"]}

    rep, err = run(tmp, a, b, prefix(3400.0), prefix(3600.0))
    check(rep is not None, f"comparator exits 0 ({err.strip()[:120]})")
    if rep is None:
        return 1
    check(rep["acceptance"]["decode_4k_pass"], "4K decode +20% passes")
    check(rep["acceptance"]["needle_pass"], "needle pass on both sides")
    check(not rep["acceptance"]["long_context_regressions"], "no long-context regression")
    check(rep["acceptance"]["pass"], "verdict PASS")

    # 128K decode 5% worse -> FAIL on the regression rule
    import copy
    a_bad = copy.deepcopy(a)
    a_bad["results"][1]["summary"]["output_tps"]["median"] = 7.6
    rep2, _ = run(tmp, a_bad, b, prefix(3400.0), prefix(3600.0))
    check(rep2 is not None and not rep2["acceptance"]["pass"], "128K decode -5% fails")
    check(rep2 is not None and any("131072" in r for r in rep2["acceptance"]["long_context_regressions"]),
          "regression names the 128K context")

    # decode gain below target but a >15% turn TTFT gain -> PASS on the turn rule
    rep3, _ = run(tmp, a, b, prefix(2900.0), prefix(3600.0))  # 2900 vs 3600 = -19.4%
    check(rep3 is not None and rep3["acceptance"]["turn_ttft_pass"], "turn TTFT -19% passes")
    check(rep3 is not None and rep3["acceptance"]["pass"], "verdict PASS on the turn rule alone")

    print("PASS" if FAIL == 0 else f"{FAIL} FAILED")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
