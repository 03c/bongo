#!/usr/bin/env python3
"""Measure random-read cost of the bongo PLE / n-gram "second shard" on the reference SSD.

The PLE table (`per_layer_token_embd.weight`, 320,001,536 rows x 90 B = 26.82 GiB)
lives inside shard 1 of every bongo GGUF tier and is read a few rows per token.
This tool measures what that actually costs on the reference box's NVMe:

  * O_DIRECT 4 KiB random reads at queue depth 1 and 16 (device latency, no page cache),
  * buffered (page-cache) random reads, warm and after ``posix_fadvise(DONTNEED)``,
  * a token simulation: 16 rows/token fetched with a prefetch width of 1, 2, 4, 8 or 16,
    which is the number the decode-overlap budget needs.

Everything is read-only. Offsets are derived from the tensor table, not hard-coded, so
the same command works for every tier: point ``--inventory`` at the JSON that
``tools/gguf-inventory.py`` wrote for that tier.

Example::

    tools/gguf-inventory.py --local <tier>-00001-of-00002.gguf \\
        --json bench/results/<run>/raw/gguf-inventory-iq2xs.json --summary
    bench/measure-ssd-ngram.py \\
        --inventory bench/results/<run>/raw/gguf-inventory-iq2xs.json \\
        --out bench/results/<run>/raw/ssd-ngram-latency.json

Requires Linux and Python 3.9+ (uses ``os.preadv`` / ``os.posix_fadvise``).
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import statistics
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

PAGE = 4096  # the unit the PLE reader (and this tool) reads


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def percentile(sorted_us: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted list, in microseconds."""
    if not sorted_us:
        return float("nan")
    k = (len(sorted_us) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_us) - 1)
    return sorted_us[lo] + (sorted_us[hi] - sorted_us[lo]) * (k - lo)


def summarize(latencies_us: Sequence[float], *, wall_s: Optional[float] = None,
              bytes_read: Optional[int] = None) -> Dict[str, float]:
    s = sorted(latencies_us)
    out = {
        "n": len(s),
        "mean_us": statistics.fmean(s) if s else None,
        "p50_us": percentile(s, 50),
        "p90_us": percentile(s, 90),
        "p95_us": percentile(s, 95),
        "p99_us": percentile(s, 99),
        "max_us": s[-1] if s else None,
        "min_us": s[0] if s else None,
    }
    if wall_s:
        out["wall_s"] = wall_s
        out["iops"] = len(s) / wall_s
    if bytes_read is not None and wall_s:
        out["mib_per_s"] = bytes_read / wall_s / (1024 * 1024)
    return out


def touch_offsets(path: str, offsets: Sequence[int], bs: int) -> None:
    """Read every offset once through the page cache (warms it)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        for off in offsets:
            os.pread(fd, bs, off)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# O_DIRECT readers
# --------------------------------------------------------------------------- #

def _direct_worker(path: str, bs: int, work: Sequence[int], out: List[float]) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, bs)  # page-aligned: required for O_DIRECT
    try:
        for i, off in enumerate(work):
            t0 = time.perf_counter_ns()
            os.preadv(fd, [buf], off)
            out[i] = (time.perf_counter_ns() - t0) / 1000.0
    finally:
        buf.close()
        os.close(fd)


def direct_parallel(path: str, offsets: Sequence[int], bs: int,
                    workers: int) -> Tuple[List[float], float]:
    """Issue `offsets` with `workers` concurrent O_DIRECT readers.

    Returns per-read latencies (us) and the wall time of the whole batch.
    """
    chunks: List[List[int]] = [[] for _ in range(workers)]
    for i, off in enumerate(offsets):
        chunks[i % workers].append(off)
    results: List[List[float]] = [[0.0] * len(c) for c in chunks]
    t0 = time.perf_counter()
    threads = [threading.Thread(target=_direct_worker, args=(path, bs, chunks[w], results[w]))
               for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return [x for r in results for x in r], wall


def token_prefetch(path: str, tokens_pages: Sequence[Sequence[int]], bs: int,
                   workers: int) -> Tuple[List[float], float]:
    """Model one token's 16 rows fetched with `workers` readers.

    All threads rendezvous on a barrier per token so the measured latency is the
    time from "first row issued" to "last row done" for that token, i.e. the
    window a real prefetcher has to hide the reads in.
    """
    rows = len(tokens_pages[0])
    if rows % workers != 0:
        raise ValueError(f"workers ({workers}) must divide rows/token ({rows})")
    per = rows // workers
    lat = [0.0] * len(tokens_pages)
    barrier = threading.Barrier(workers)
    stop = threading.Event()

    def worker(w: int) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, bs)
        try:
            for i, pages in enumerate(tokens_pages):
                barrier.wait()
                if stop.is_set():
                    return
                t0 = time.perf_counter_ns()
                for p in pages[w * per:(w + 1) * per]:
                    os.preadv(fd, [buf], p)
                dt = (time.perf_counter_ns() - t0) / 1000.0
                # the token is not ready until the slowest worker returns
                if dt > lat[i]:
                    lat[i] = dt
                barrier.wait()
        finally:
            buf.close()
            os.close(fd)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return lat, wall


# --------------------------------------------------------------------------- #
# region selection
# --------------------------------------------------------------------------- #

def ple_region(inventory_path: str) -> Tuple[str, int, int, int, int]:
    """Return (model_file, abs_offset, length_bytes, rows, row_bytes) for the PLE table.

    The tensor `offset` in the inventory is shard-relative; add the shard's data
    section offset to get the absolute file offset.
    """
    with open(inventory_path) as fh:
        inv = json.load(fh)
    tensor = next(t for t in inv["tensors"] if t["name"] == "per_layer_token_embd.weight")
    shard = next(s for s in inv["shards"] if s["file"] == tensor["shard"])
    # split GGUFs sit next to each other; resolve the shard against the inventory source
    local = (inv.get("source") or {}).get("local")
    model = os.environ.get("BONGO_GGUF_FILE")
    if not model and local:
        model = os.path.join(os.path.dirname(local), tensor["shard"])
    if not model or not os.path.exists(model):
        raise SystemExit(
            f"model shard not found: {model}\n"
            f"set BONGO_GGUF_FILE=<path to {tensor['shard']}>")
    rows = tensor["dims"][1]
    row_bytes = tensor["bytes"] // rows
    return model, shard["data_section_offset"] + tensor["offset"], tensor["bytes"], rows, row_bytes


def sample_pages(start: int, length: int, n: int, bs: int, rng: random.Random) -> List[int]:
    """`n` distinct page-aligned offsets inside [start, start+length)."""
    first = start - (start % bs)
    last = (start + length - bs) - ((start + length - bs) % bs)
    span = (last - first) // bs + 1
    if n > span:
        raise ValueError("requested more pages than the region holds")
    picks = rng.sample(range(span), n)
    return [first + i * bs for i in picks]


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #

def scenario_direct(path: str, starts: Sequence[int], args) -> Dict:
    out: Dict[str, Dict] = {}
    for qd in args.qds:
        lat, wall = direct_parallel(path, starts, args.block_size, qd)
        out[f"direct_qd{qd}"] = summarize(lat, wall_s=wall, bytes_read=len(starts) * args.block_size)
    return out


def scenario_sequential(path: str, region_start: int, region_len: int, args) -> Dict:
    """Sequential O_DIRECT throughput with 1 MiB blocks (the prefill-style ceiling)."""
    bs = 1 << 20
    total = min(args.seq_mib << 20, region_len - bs)
    start = region_start - (region_start % bs)
    lat: List[float] = []
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, bs)
    t0 = time.perf_counter()
    try:
        off = start
        while off < start + total:
            t = time.perf_counter_ns()
            os.preadv(fd, [buf], off)
            lat.append((time.perf_counter_ns() - t) / 1000.0)
            off += bs
    finally:
        buf.close()
        os.close(fd)
    wall = time.perf_counter() - t0
    return {"sequential_direct_1mib": summarize(lat, wall_s=wall, bytes_read=total)}


def scenario_buffered(path: str, region_start: int, region_len: int, args) -> Dict:
    offsets = sample_pages(region_start, region_len, args.reads, args.block_size, args.rng)
    out: Dict[str, Dict] = {}

    # warm: touch once, then re-read through the page cache
    touch_offsets(path, offsets, args.block_size)
    lat_warm: List[float] = []
    fd = os.open(path, os.O_RDONLY)
    try:
        for off in offsets:
            t0 = time.perf_counter_ns()
            os.pread(fd, args.block_size, off)
            lat_warm.append((time.perf_counter_ns() - t0) / 1000.0)
    finally:
        os.close(fd)
    out["buffered_warm"] = summarize(lat_warm, bytes_read=len(offsets) * args.block_size)

    # cold: drop a bounded window with fadvise, then read it with default readahead
    win = min(args.evict_window, region_len)
    win_start = region_start - (region_start % args.block_size)
    cold_offsets = sample_pages(win_start, win, min(args.reads, win // args.block_size), args.block_size, args.rng)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, win_start, win, os.POSIX_FADV_DONTNEED)
        lat_cold: List[float] = []
        for off in cold_offsets:
            t0 = time.perf_counter_ns()
            os.pread(fd, args.block_size, off)
            lat_cold.append((time.perf_counter_ns() - t0) / 1000.0)
    finally:
        os.close(fd)
    out["buffered_cold_dontneed"] = summarize(lat_cold, bytes_read=len(cold_offsets) * args.block_size)
    return out


def scenario_tokens(path: str, region_start: int, region_len: int, args) -> Dict:
    """16 pages per token, fetched with prefetch width 1/2/4/8/16."""
    pages = sample_pages(region_start, region_len, args.tokens * 16, args.block_size, args.rng)
    tokens = [list(pages[i * 16:(i + 1) * 16]) for i in range(args.tokens)]
    out: Dict[str, Dict] = {}
    for w in (1, 2, 4, 8, 16):
        lat, wall = token_prefetch(path, tokens, args.block_size, w)
        out[f"token_window_qd{w}"] = summarize(
            lat, wall_s=wall, bytes_read=args.tokens * 16 * args.block_size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True,
                    help="JSON written by tools/gguf-inventory.py for the tier")
    ap.add_argument("--out", required=True, help="where to write the measurement JSON")
    ap.add_argument("--reads", type=int, default=4000, help="random reads per latency scenario")
    ap.add_argument("--tokens", type=int, default=1000, help="tokens in the prefetch simulation")
    ap.add_argument("--block-size", type=int, default=PAGE)
    ap.add_argument("--qd", type=int, default=16, help="(unused; kept for compatibility)")
    ap.add_argument("--qds", type=int, nargs="+", default=[1, 8, 16, 32, 64],
                    help="queue depths for the direct random test")
    ap.add_argument("--seq-mib", type=int, default=256, help="MiB for the sequential test")
    ap.add_argument("--evict-window", type=int, default=2 << 30, help="bytes fadvise-dropped for the cold test")
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.rng = rng
    model, region_start, region_len, rows, row_bytes = ple_region(args.inventory)
    print(f"PLE region: {model}\n  abs offset {region_start:,}  length {region_len:,} B  "
          f"rows {rows:,}  row {row_bytes} B", file=sys.stderr)

    # every scenario samples its own fresh pages; keep the samples disjoint enough
    starts = sample_pages(region_start, region_len, args.reads, args.block_size, rng)
    result = {
        "meta": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_file": model,
            "inventory": os.path.abspath(args.inventory),
            "region_offset": region_start,
            "region_bytes": region_len,
            "rows": rows,
            "row_bytes": row_bytes,
            "block_size": args.block_size,
            "reads": args.reads,
            "qds": args.qds,
            "tokens": args.tokens,
            "seq_mib": args.seq_mib,
            "seed": args.seed,
            "page_cache": _page_cache_state(),
            "io_pressure_start": io_pressure(),
        },
        "scenarios": {},
    }
    result["scenarios"].update(scenario_direct(model, starts, args))
    result["scenarios"].update(scenario_buffered(model, region_start, region_len, args))
    result["scenarios"].update(scenario_sequential(model, region_start, region_len, args))
    result["scenarios"].update(scenario_tokens(model, region_start, region_len, args))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result["meta"]["io_pressure_end"] = io_pressure()
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)
    for name, s in result["scenarios"].items():
        print(f"  {name:24s} p50={s['p50_us']:8.1f}us p99={s['p99_us']:9.1f}us "
              f"iops={s.get('iops', float('nan')):8.0f}", file=sys.stderr)
    return 0


def _page_cache_state() -> Dict[str, int]:
    state: Dict[str, int] = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(("MemFree:", "Cached:", "Buffers:")):
                    k, v = line.split(":")
                    state[k.strip()] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return state


def io_pressure() -> Dict[str, float]:
    """The kernel's PSI IO pressure, so a contended run is self-documenting."""
    out: Dict[str, float] = {}
    try:
        with open("/proc/pressure/io") as fh:
            for line in fh:
                parts = line.split()
                kind = parts[0]
                for p in parts[1:]:
                    k, v = p.split("=")
                    if k == "avg10":
                        out[f"{kind}_avg10"] = float(v)
    except OSError:
        pass
    return out


if __name__ == "__main__":
    raise SystemExit(main())
