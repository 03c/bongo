#!/usr/bin/env python3
"""Regression test for bench/gen-moe-cache-profile.py (BAS-139).

Builds a synthetic router capture with a known hot expert per layer, runs the
generator against a tiny byte budget, and checks the emitted engine profile
(the `L <il> <expert> ...` format) and the JSON metadata. No GPU, no network.

  python3 tests/moe-cache-profile.test.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEN = os.path.join(ROOT, "bench", "gen-moe-cache-profile.py")

N_LAYERS = 48
N_EXPERTS = 512
N_TOKENS = 20
BYTES_PER_LAYER = 1000

FAIL = 0


def check(cond, msg):
    global FAIL
    if cond:
        print(f"ok   - {msg}")
    else:
        print(f"FAIL - {msg}")
        FAIL += 1


def write_trace(path, corpora):
    # each corpus: for every layer, a fixed hot expert on every token plus a
    # rotating cold one, so the hot expert is the clear frequency winner
    for name in corpora:
        with open(os.path.join(path, name + ".tsv"), "w") as fh:
            for t in range(N_TOKENS):
                for l in range(N_LAYERS):
                    hot = (l * 7 + 3) % N_EXPERTS
                    cold = (t * 13 + l) % N_EXPERTS
                    fh.write(f"TOPK\tffn_moe_topk-{l}\t2\t1\t1\t1\t{hot}\t{cold}\n")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="moe-cache-profile-test-")
    corpus_dir = os.path.join(tmp, "raw")
    os.makedirs(corpus_dir)
    write_trace(corpus_dir, ["doc", "code"])

    expert_bytes = os.path.join(tmp, "expert-bytes.json")
    with open(expert_bytes, "w") as fh:
        json.dump({"tier": "test", "expert_bytes_by_layer": [BYTES_PER_LAYER] * N_LAYERS}, fh)

    out = os.path.join(tmp, "profile.txt")
    meta = os.path.join(tmp, "profile.json")

    # budget: 2 experts/layer across all 48 layers
    budget_gib = (2 * N_LAYERS * BYTES_PER_LAYER / N_EXPERTS) / (1024 ** 3)
    proc = subprocess.run(
        [sys.executable, GEN, "--raw", corpus_dir, "--expert-bytes", expert_bytes,
         "--budget-gib", repr(budget_gib), "--corpora", "doc,code",
         "--out", out, "--json-out", meta],
        capture_output=True, text=True)
    check(proc.returncode == 0, f"generator exits 0 ({proc.stderr.strip()[:120]})")
    if proc.returncode != 0:
        return 1

    lines = [l.rstrip("\n") for l in open(out) if l.startswith("L ")]
    check(len(lines) == N_LAYERS, f"one L line per layer ({len(lines)})")

    ok_first = True
    for l, line in enumerate(lines):
        ids = [int(x) for x in line.split()[2:]]
        hot = (l * 7 + 3) % N_EXPERTS
        if not ids or ids[0] != hot:
            ok_first = False
            break
    check(ok_first, "the most frequent expert of each layer is listed first")

    d = json.load(open(meta))
    check(d["cells"] == 2 * N_LAYERS, f"cells == 2/layer ({d['cells']})")
    check(0.5 < d["coverage"] <= 1.0, f"coverage in (0.5,1] ({d['coverage']:.4f})")
    check(d["bytes"] <= d["budget_bytes"], "byte budget respected")
    check(d["policy"].startswith("profile initialisation"), "policy is profile-init + LRU")

    print("PASS" if FAIL == 0 else f"{FAIL} FAILED")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
