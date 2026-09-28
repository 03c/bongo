#!/usr/bin/env python3
"""Self-test for bench/analyze-backend-ab.py (BAS-72).

The A/B decides the shipped default backend, so the rule itself needs a test
that does not need a GPU or two hours of Arc B70 time: synthetic result files
stand in for the real runs, and the analyser must reach the documented verdict.
The Vulkan reference numbers are the real ones from the committed run.

    python3 bench/analyze-backend-ab-selftest.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZE = os.path.join(HERE, "analyze-backend-ab.py")

DEEP = 131072
SMALL = 4096
PREFIX = 31744


def matrix(prompt_tps, output_tps, ttft_ms, vram_gib=29.0):
    def runs(ctx, tps, otps, ttft):
        return [
            {"prompt_tps": tps, "output_tps": otps, "ttft_ms": ttft, "prompt_tokens": ctx}
            for _ in range(3)
        ]

    return {
        "generated_at": "2026-09-28T00:00:00Z",
        "tier": "iq2_xs",
        "harness": {"argv": ["harness.py"]},
        "config": {"context_limit": DEEP, "repeats": 3, "cache_prompt": False,
                   "needle_context": DEEP},
        "needle": {"status": "pass"},
        "results": [
            {"target_context": SMALL, "status": "ok",
             "runs": runs(SMALL, prompt_tps * 1.73, output_tps * 2.2, 17000),
             "memory": {"vram_peak_bytes": vram_gib * (1024 ** 3)}},
            {"target_context": DEEP, "status": "ok",
             "runs": runs(DEEP, prompt_tps, output_tps, ttft_ms),
             "memory": {"vram_peak_bytes": vram_gib * (1024 ** 3)}},
        ],
    }


def prefix_cache(turn_ms, steady_ms=250):
    return {
        "delta_tokens": 512,
        "runs": [
            {"label": f"cold_p{PREFIX}", "ttft_ms": turn_ms * 40, "cached_tokens": 0},
            {"label": f"grow_p{PREFIX}_d512", "ttft_ms": turn_ms, "cached_tokens": PREFIX - 5},
            {"label": f"grow_p{PREFIX}_repeat", "ttft_ms": steady_ms, "cached_tokens": PREFIX + 507},
        ],
    }


def fake_build(root, leg, build_backend):
    """A stand-in llama.cpp build directory carrying one backend's library."""
    bindir = os.path.join(root, "bin", leg)
    os.makedirs(bindir, exist_ok=True)
    binary = os.path.join(bindir, "llama-server")
    with open(binary, "w") as fh:
        fh.write("not a real ELF\n")
    os.chmod(binary, 0o755)
    with open(os.path.join(bindir, f"libggml-{build_backend}.so"), "w") as fh:
        fh.write("")
    return binary


def write_leg(root, leg, mtx, pc, binary):
    be = os.path.join(root, leg)
    os.makedirs(os.path.join(be, "raw"), exist_ok=True)
    os.makedirs(os.path.join(be, "prefix-cache"), exist_ok=True)
    with open(os.path.join(be, "matrix.json"), "w") as fh:
        json.dump(mtx, fh)
    with open(os.path.join(be, "prefix-cache", "prefix-cache.json"), "w") as fh:
        json.dump(pc, fh)
    with open(os.path.join(be, "bongo-config.json"), "w") as fh:
        json.dump({
            "llama_cpp": {"revision": "b11223", "commit": "4da6337767f973e2",
                          "backend": "Vulkan" if leg == "vulkan" else "SYCL",
                          "binary": binary},
            "model": {"tier": "iq2_xs"},
            "gpu": {"name": "Intel Arc Pro B70"},
            "runtime": {"dir": "/tmp/runtime"},
            "server": {"flags": [
                "--model", "m.gguf", "--ctx-size", str(DEEP), "--jinja",
                "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                "--n-gpu-layers", "99", "--n-cpu-moe", "16", "--host", "127.0.0.1",
                "--port", "8080", "--parallel", "1", "--metrics",
                "--device", "Vulkan1" if leg == "vulkan" else "SYCL0"]},
        }, fh)
    with open(os.path.join(be, "raw", f"{leg}-r1.json"), "w") as fh:
        json.dump({"synthetic": True}, fh)


def case(name, want_sycl, vk, sy, sycl_build="sycl"):
    root = tempfile.mkdtemp(prefix="ab-selftest-")
    try:
        write_leg(root, "vulkan", *vk, fake_build(root, "vulkan", "vulkan"))
        if sy is not None:
            write_leg(root, "sycl", *sy, fake_build(root, "sycl", sycl_build))
        proc = subprocess.run(
            [sys.executable, ANALYZE, "--dir", root,
             "--out", os.path.join(root, "summary.md")],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return False, f"analyser failed rc={proc.returncode}: {proc.stderr.strip()}"
        with open(os.path.join(root, "summary.md")) as fh:
            summary = fh.read()
        got_sycl = "**SYCL becomes the default.**" in summary
        if got_sycl != want_sycl:
            tail = summary[summary.index("## Decision"):][:600]
            return False, f"expected sycl={want_sycl}, got {got_sycl}\n{tail}"
        with open(os.path.join(root, "adr-amendment.md")) as fh:
            adr = fh.read()
        if not adr.startswith("## Amendment"):
            return False, "adr-amendment.md was not written"
        return True, name
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    # The real committed Vulkan run: 133.46 tok/s prefill, 8.00 tok/s decode at
    # 131072, 980705 ms cold TTFT, 4165 ms cached turn over a 31744-token prefix.
    vk = (matrix(133.46, 8.00, 980705), prefix_cache(4165))
    results = [
        # SYCL clears every gate: 1.50x cold TTFT, 1.50x cached turn, decode 0.99x.
        case("sycl clears all gates", True, vk,
             (matrix(133.46, 7.90, 653803), prefix_cache(2776))),
        # R6's shape: prefill ~1.5x faster, decode ~1.6x slower.
        case("sycl decode 1.6x slower", False, vk,
             (matrix(133.46, 5.00, 653803), prefix_cache(2776))),
        # Fast prefill, but the agentic cached turn only improves 1.10x.
        case("sycl cached-turn only 1.10x", False, vk,
             (matrix(133.46, 7.90, 653803), prefix_cache(3786))),
        # Marginally under the 1.3x cold-TTFT bar (1.25x).
        case("sycl cold ttft 1.25x", False, vk,
             (matrix(133.46, 7.90, 784564), prefix_cache(3332))),
        # A "SYCL" leg that is really the Vulkan build must never win.
        case("wrong binary in the sycl leg", False, vk,
             (matrix(133.46, 7.90, 653803), prefix_cache(2776)), sycl_build="vulkan"),
        # Missing SYCL data cannot hand the default to SYCL.
        case("no sycl result", False, vk, None),
    ]

    failures = [(ok, msg) for ok, msg in results if not ok]
    for ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {msg.splitlines()[0]}")
        if not ok:
            for line in msg.splitlines()[1:]:
                print(f"      {line}")
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
