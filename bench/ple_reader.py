#!/usr/bin/env python3
"""PLE / n-gram second-shard reader for bongo (M3.4).

The model's ``per_layer_token_embd.weight`` tensor ("PLE" / n-gram table) is
26.82 GiB in every tier and must stay on SSD.  Every token reads 16 rows of
90 B, one per hash head (``qwen4exp`` ``llm_graph_input_ple::set_input``); the
rows land on 16 distinct 4 KiB pages, so the raw read is 64 KiB/token against
1,440 B of useful data.  R5 (``docs/research/ssd-ngram-shard.md``) measured the
device: 1.65 ms/token serial, 0.26 ms/token with the 16 reads in parallel.

This module is the reader the R5 design calls for:

* a dedicated pool of ``io_depth`` reader threads issuing ``O_DIRECT`` 4 KiB
  page reads as soon as a token's row ids are known (parallel, never serial);
* a bounded LRU **row** cache (>= 1M rows / ~90 MB) in front of the file, so
  the ~45 rows that share a page are not paid for again;
* a bounded in-flight window (``--window`` pages) so a slow disk produces
  bounded latency instead of unbounded memory;
* batch + de-duplicate rows/pages for prefill (one chunk is one pass);
* an mmap baseline (``MmapPle``) that reproduces the llama.cpp
  ``--lazy-mode auto`` page-fault path for a like-for-like comparison.

The engine itself is not linked here: this is the reader implemented against
the real table and measured on the real SSD, so the design and the numbers can
be checked before/while it is wired into llama.cpp.  A C++ port keeps the same
shape; only the pool and the cache move to C++.

Read-only.  Requires Linux + Python 3.9 (``os.preadv`` / ``os.O_DIRECT``).

Example::

    bench/ple_reader.py --inventory bench/results/<run>/raw/gguf-inventory-iq2xs.json \\
        --tokens 'bench/results/<run>/raw/tokens-*.json' \\
        --out bench/results/<run>/raw/ple-reader.json --scenario all
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import mmap
import os
import random
import resource
import statistics
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PAGE = 4096
_SIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngram-row-cache-sim.py")


# --------------------------------------------------------------------------- #
# reuse the exact qwen4exp row hash from the R5 simulation
# --------------------------------------------------------------------------- #

def load_sim_module():
    spec = importlib.util.spec_from_file_location("ngram_row_cache_sim", _SIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize_us(latencies_us: Sequence[float], *, wall_s: Optional[float] = None,
                 bytes_read: Optional[int] = None) -> Dict[str, float]:
    s = sorted(latencies_us)
    out = {
        "n": len(s),
        "mean_us": statistics.fmean(s) if s else None,
        "p50_us": percentile(s, 50),
        "p90_us": percentile(s, 90),
        "p99_us": percentile(s, 99),
        "max_us": s[-1] if s else None,
        "min_us": s[0] if s else None,
    }
    if wall_s:
        out["wall_s"] = wall_s
        out["per_s"] = len(s) / wall_s
    if bytes_read is not None and wall_s and wall_s > 0:
        out["mib_per_s"] = bytes_read / wall_s / (1024 * 1024)
    return out


def _usage() -> Dict[str, int]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {"ru_minflt": ru.ru_minflt, "ru_majflt": ru.ru_majflt, "ru_maxrss_kib": ru.ru_maxrss}


def io_pressure() -> Dict[str, float]:
    """Kernel PSI IO pressure, so a contended run is self-documenting (as in R5)."""
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


def rss_kib() -> int:
    out = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


# --------------------------------------------------------------------------- #
# bounded, aligned O_DIRECT reader pool
# --------------------------------------------------------------------------- #

class IOPool:
    """A fixed pool of reader threads + a bounded window of aligned page slots.

    ``read_pages`` submits at most ``window`` unique page offsets, blocks until
    all of them are in the arena, then returns a ``memoryview`` over the arena
    where slot ``i`` holds ``pages[i]``.  The window is the in-flight cap.
    """

    def __init__(self, path: str, io_depth: int, window: int, direct: bool = True,
                 page: int = PAGE) -> None:
        flags = os.O_RDONLY
        if direct:
            flags |= getattr(os, "O_DIRECT", 0)
        self.fd = os.open(path, flags)
        self.direct = direct and hasattr(os, "O_DIRECT")
        self.page = page
        self.window = window
        self.arena = mmap.mmap(-1, window * page)  # anonymous, page-aligned
        self.mv = memoryview(self.arena)
        # Two conditions on one lock: workers wait for work, the caller waits for the
        # batch to finish.  Only the last completion wakes the caller, so a large
        # window does not become a thundering herd of notifications.
        self.lock = threading.Lock()
        self.work_cv = threading.Condition(self.lock)
        self.done_cv = threading.Condition(self.lock)
        self.queue: List[Tuple[int, int]] = []
        self.qhead = 0
        self.done = 0
        self.total = 0
        self.stop = False
        self.io_errors = 0
        self.threads = [threading.Thread(target=self._worker, name=f"ple-io-{i}", daemon=True)
                        for i in range(max(1, io_depth))]
        for t in self.threads:
            t.start()

    def _worker(self) -> None:
        while True:
            with self.lock:
                while self.qhead >= len(self.queue) and not self.stop:
                    self.work_cv.wait()
                if self.qhead >= len(self.queue):
                    if self.stop:
                        return
                    continue
                slot, off = self.queue[self.qhead]
                self.qhead += 1
            view = self.mv[slot * self.page:(slot + 1) * self.page]
            try:
                os.preadv(self.fd, [view], off)
            except OSError:
                with self.lock:
                    self.io_errors += 1
            with self.lock:
                self.done += 1
                if self.done == self.total:
                    self.done_cv.notify_all()

    def read_pages(self, pages: Sequence[int]) -> memoryview:
        if len(pages) > self.window:
            raise ValueError(f"{len(pages)} pages exceed the window {self.window}")
        if not pages:
            return self.mv
        # deterministic slot order == input order
        with self.done_cv:
            self.queue = list(enumerate(pages))
            self.qhead = 0
            self.done = 0
            self.total = len(pages)
            self.work_cv.notify_all()
            while self.done < self.total:
                self.done_cv.wait()
        return self.mv

    def close(self) -> None:
        with self.work_cv:
            self.stop = True
            self.work_cv.notify_all()
        for t in self.threads:
            t.join(timeout=5)
        self.mv.release()
        self.arena.close()
        os.close(self.fd)


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #

class RowCache:
    """Bounded LRU keyed by global row index, storing the 90 B row."""

    def __init__(self, capacity_rows: int) -> None:
        self.capacity = capacity_rows
        self.rows: "OrderedDict[int, bytes]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> Optional[bytes]:
        val = self.rows.get(key)
        if val is None:
            self.misses += 1
            return None
        self.rows.move_to_end(key)
        self.hits += 1
        return val

    def put(self, key: int, value: bytes) -> None:
        if self.capacity <= 0:
            return
        self.rows[key] = value
        if len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    def __len__(self) -> int:
        return len(self.rows)


class PleReader:
    """Direct-read PLE gather with a bounded row cache and a bounded window."""

    def __init__(self, path: str, region_start: int, rows: int, row_bytes: int, *,
                 cache_rows: int = 1_000_000, io_depth: int = 16, window: int = 64,
                 direct: bool = True, page: int = PAGE) -> None:
        self.path = path
        self.base = region_start
        self.rows = rows
        self.row_bytes = row_bytes
        self.page = page
        self.cache = RowCache(cache_rows)
        self.pool = IOPool(path, io_depth, window, direct=direct, page=page)
        self.stats = {
            "gathers": 0,
            "rows_requested": 0,
            "rows_hit": 0,
            "rows_missed": 0,
            "pages_read": 0,
            "bytes_read": 0,
            "batches": 0,
        }

    # -- geometry ---------------------------------------------------------- #

    def _row_pages(self, row: int) -> Tuple[int, int]:
        off = self.base + row * self.row_bytes
        p0 = off - (off % self.page)
        p1 = (off + self.row_bytes - 1) - ((off + self.row_bytes - 1) % self.page)
        return p0, p1

    def _pages_for_rows(self, rows: Iterable[int]) -> List[int]:
        pages = set()
        for r in rows:
            p0, p1 = self._row_pages(r)
            pages.add(p0)
            if p1 != p0:
                pages.add(p1)
        return sorted(pages)

    def _slots_for(self, pages: Sequence[int]) -> Dict[int, int]:
        return {p: i for i, p in enumerate(pages)}

    def _row_bytes_from(self, row: int, mv: memoryview, slots: Dict[int, int]) -> bytes:
        off = self.base + row * self.row_bytes
        p0, _ = self._row_pages(row)
        delta = off - p0
        if delta + self.row_bytes <= self.page:
            i = slots[p0]
            return bytes(mv[i * self.page + delta: i * self.page + delta + self.row_bytes])
        # straddles a page boundary: join the two pages
        if (p0 + self.page) not in slots:
            raise RuntimeError(f"straddling row {row}: page {p0 + self.page} not in window")
        first = self.page - delta
        i0, i1 = slots[p0], slots[p0 + self.page]
        b0 = bytes(mv[i0 * self.page + delta: i0 * self.page + self.page])
        b1 = bytes(mv[i1 * self.page: i1 * self.page + (self.row_bytes - first)])
        return b0 + b1

    def _read_rows(self, miss_rows: Sequence[int]) -> None:
        """Read every page covering ``miss_rows`` in one window and cache the rows."""
        if not miss_rows:
            return
        pages = self._pages_for_rows(miss_rows)
        # group missed rows by their first page; keep the window intact
        by_page: Dict[int, List[int]] = defaultdict(list)
        crossing: Dict[int, List[int]] = defaultdict(list)
        for r in miss_rows:
            p0, p1 = self._row_pages(r)
            by_page[p0].append(r)
            if p1 != p0:
                crossing[p0].append(p1)
        i = 0
        while i < len(pages):
            chunk: List[int] = []
            while i < len(pages) and len(chunk) < self.pool.window:
                p = pages[i]
                extra = 1 if p in crossing else 0
                if len(chunk) + 1 + extra > self.pool.window:
                    break  # leave p for the next window; never split a straddling row
                chunk.append(p)
                i += 1
                if extra:
                    # the straddling row's second page is the very next in sorted order
                    if i < len(pages) and pages[i] == p + self.page:
                        chunk.append(pages[i])
                        i += 1
            mv = self.pool.read_pages(chunk)
            slots = self._slots_for(chunk)
            self.stats["pages_read"] += len(chunk)
            self.stats["bytes_read"] += len(chunk) * self.page
            self.stats["batches"] += 1
            for p in chunk:
                for r in by_page.get(p, ()):
                    self.cache.put(r, self._row_bytes_from(r, mv, slots))

    # -- API --------------------------------------------------------------- #

    def gather(self, rows: Sequence[int]) -> bytes:
        """Decode path: return the 16 rows (``row_bytes`` each) for one token."""
        rb = self.row_bytes
        out = bytearray(len(rows) * rb)
        miss: List[Tuple[int, int]] = []  # (destination index, row id)
        for i, r in enumerate(rows):
            val = self.cache.get(r)
            if val is None:
                miss.append((i, r))
            else:
                out[i * rb:(i + 1) * rb] = val
        if miss:
            self._read_rows([r for _, r in miss])
            for i, r in miss:
                val = self.cache.rows.get(r)
                if val is not None:
                    out[i * rb:(i + 1) * rb] = val
        self.stats["gathers"] += 1
        self.stats["rows_requested"] += len(rows)
        self.stats["rows_missed"] += len(miss)
        self.stats["rows_hit"] += len(rows) - len(miss)
        return bytes(out)

    def prefetch(self, rows: Sequence[int]) -> None:
        """Prefill path: de-duplicate rows and pages, then read them once."""
        unique = list(dict.fromkeys(rows))
        miss: List[int] = []
        cached = self.cache.rows
        for r in unique:
            if r not in cached:
                miss.append(r)
        if miss:
            self._read_rows(miss)
        self.stats["rows_requested"] += len(rows)
        self.stats["rows_missed"] += len(miss)
        self.stats["rows_hit"] += len(rows) - len(miss)

    def close(self) -> None:
        self.pool.close()


class MmapPle:
    """The llama.cpp ``--lazy-mode auto`` path: mmap + MADV_RANDOM + page faults."""

    def __init__(self, path: str, region_start: int, region_len: int, row_bytes: int,
                 page: int = PAGE) -> None:
        self.page = page
        self.row_bytes = row_bytes
        self.map_base = region_start - (region_start % page)
        self.delta = region_start - self.map_base
        size = region_len + self.delta
        self.fd = os.open(path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, size, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED,
                            offset=self.map_base)
        try:
            self.mm.madvise(mmap.MADV_RANDOM)
        except (AttributeError, OSError):
            pass

    def row(self, r: int) -> bytes:
        off = self.delta + r * self.row_bytes
        return self.mm[off:off + self.row_bytes]

    def touch_pages(self, pages: Sequence[int]) -> int:
        """Fault in exactly these pages; returns a cheap checksum to defeat optimisation."""
        acc = 0
        page = self.page
        mm = self.mm
        for p in pages:
            acc ^= mm[p - self.map_base]
        return acc

    def close(self) -> None:
        self.mm.close()
        os.close(self.fd)


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #

def ple_layout(inventory: str) -> Dict:
    with open(inventory) as fh:
        inv = json.load(fh)
    tensor = next(t for t in inv["tensors"] if t["name"] == "per_layer_token_embd.weight")
    shard = next(s for s in inv["shards"] if s["file"] == tensor["shard"])
    local = (inv.get("source") or {}).get("local")
    model = os.environ.get("BONGO_GGUF_FILE")
    if not model and local:
        model = local
    if not model or not os.path.exists(model):
        raise SystemExit(f"model shard not found: {model}\nset BONGO_GGUF_FILE=<path to {tensor['shard']}>")
    rows = tensor["dims"][1]
    return {
        "model": model,
        "offset": shard["data_section_offset"] + tensor["offset"],
        "length": tensor["bytes"],
        "rows": rows,
        "row_bytes": tensor["bytes"] // rows,
    }


def load_token_sequences(patterns: Sequence[str], params: Dict) -> List[Dict]:
    sim = load_sim_module()
    files: List[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit("no token files matched")
    out = []
    for path in files:
        with open(path) as fh:
            rec = json.load(fh)
        rows = sim.token_rows(rec["ids"], params)
        out.append({
            "prompt": rec["prompt"],
            "source": os.path.basename(path),
            "n_tokens": rec["n_tokens"],
            "rows": rows,
        })
    return out


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #

def evict_pages(fd: int, pages: Sequence[int], page: int = PAGE) -> None:
    """Drop these pages from the page cache so the mmap path really faults.

    Merges runs of consecutive pages into one ``posix_fadvise`` call.
    """
    if not pages:
        return
    ordered = sorted(set(pages))
    start = prev = ordered[0]
    for p in ordered[1:]:
        if p == prev + page:
            prev = p
            continue
        os.posix_fadvise(fd, start, prev + page - start, os.POSIX_FADV_DONTNEED)
        start = prev = p
    os.posix_fadvise(fd, start, prev + page - start, os.POSIX_FADV_DONTNEED)


def scenario_decode(layout: Dict, sequences: Sequence[Dict], *, cache_rows: int,
                    io_depth: int, window: int, max_tokens: int, evict: bool, args) -> Dict:
    """Per-token gather window: reader (cache + pool) vs direct (pool, no cache) vs mmap."""
    tokens: List[List[int]] = []
    for seq in sequences:
        for rows in seq["rows"]:
            tokens.append(rows)
            if max_tokens and len(tokens) >= max_tokens:
                break
        if max_tokens and len(tokens) >= max_tokens:
            break

    out: Dict[str, Dict] = {}

    # reader: warm the cache with the same rows first, then measure the warm window
    reader = PleReader(layout["model"], layout["offset"], layout["rows"], layout["row_bytes"],
                       cache_rows=cache_rows, io_depth=io_depth, window=window)
    t0 = time.perf_counter()
    for rows in tokens:
        reader.gather(rows)
    cold_wall = time.perf_counter() - t0
    lat = []
    for rows in tokens:
        t = time.perf_counter_ns()
        reader.gather(rows)
        lat.append((time.perf_counter_ns() - t) / 1000.0)
    out["reader_warm"] = summarize_us(lat)
    out["reader_warm"]["wall_s_cold_pass"] = cold_wall
    out["reader_warm"]["cache_rows"] = len(reader.cache)
    out["reader_warm"]["hit_rate"] = (reader.cache.hits / (reader.cache.hits + reader.cache.misses)
                                      if (reader.cache.hits + reader.cache.misses) else None)
    out["reader_warm"]["stats"] = dict(reader.stats)
    out["reader_warm"]["rss_kib"] = rss_kib()
    reader.close()

    # direct: same pool, no row cache (every row is a page read)
    direct = PleReader(layout["model"], layout["offset"], layout["rows"], layout["row_bytes"],
                       cache_rows=0, io_depth=io_depth, window=window)
    lat = []
    for rows in tokens:
        t = time.perf_counter_ns()
        direct.gather(rows)
        lat.append((time.perf_counter_ns() - t) / 1000.0)
    out["reader_direct"] = summarize_us(lat)
    out["reader_direct"]["stats"] = dict(direct.stats)
    direct.close()

    # mmap: llama.cpp lazy path.  Cold first (fadvise DONTNEED before the first
    # touch of each page), then warm (page-cache hits on the same mapping).
    mm = MmapPle(layout["model"], layout["offset"], layout["length"], layout["row_bytes"])
    if evict:
        fd = os.open(layout["model"], os.O_RDONLY)
        before = _usage()
        lat = []
        acc = 0
        for rows in tokens:
            pages = []
            for r in rows:
                off = layout["offset"] + r * layout["row_bytes"]
                p0 = off - (off % PAGE)
                pages.append(p0)
                if (off % PAGE) + layout["row_bytes"] > PAGE:
                    pages.append(p0 + PAGE)
            evict_pages(fd, pages)
            t = time.perf_counter_ns()
            for r in rows:
                acc ^= mm.row(r)[0]
            lat.append((time.perf_counter_ns() - t) / 1000.0)
        out["mmap_cold"] = summarize_us(lat)
        after = _usage()
        out["mmap_cold"]["faults"] = {k: after[k] - before[k] for k in ("ru_minflt", "ru_majflt")}
        out["mmap_cold"]["checksum"] = acc
        os.close(fd)

    before = _usage()
    lat = []
    acc = 0
    for rows in tokens:
        t = time.perf_counter_ns()
        for r in rows:
            acc ^= mm.row(r)[0]
        lat.append((time.perf_counter_ns() - t) / 1000.0)
    out["mmap_warm"] = summarize_us(lat)
    after = _usage()
    out["mmap_warm"]["faults"] = {k: after[k] - before[k] for k in ("ru_minflt", "ru_majflt")}
    out["mmap_warm"]["checksum"] = acc
    mm.close()
    return out


def scenario_prefill(layout: Dict, sequences: Sequence[Dict], *, cache_rows: int,
                     io_depth: int, window: int, chunk_tokens: int, evict: bool, args) -> Dict:
    """One prefill chunk, de-duplicated: reader batch vs mmap page faults."""
    rows: List[int] = []
    for seq in sequences:
        for r16 in seq["rows"]:
            rows.extend(r16)
            if len(rows) >= chunk_tokens * 16:
                break
        if len(rows) >= chunk_tokens * 16:
            break
    chunk = chunk_tokens

    reader = PleReader(layout["model"], layout["offset"], layout["rows"], layout["row_bytes"],
                       cache_rows=max(cache_rows, len(rows) + 1024), io_depth=io_depth, window=window)
    before = _usage()
    t0 = time.perf_counter()
    reader.prefetch(rows)
    wall = time.perf_counter() - t0
    after = _usage()
    reader_res = {
        "chunk_tokens": chunk,
        "rows_requested": len(rows),
        "unique_rows": len(set(rows)),
        "pages_read": reader.stats["pages_read"],
        "bytes_read": reader.stats["bytes_read"],
        "batches": reader.stats["batches"],
        "wall_s": wall,
        "mib_per_s": reader.stats["bytes_read"] / wall / (1024 * 1024) if wall else None,
        "pages_per_token": reader.stats["pages_read"] / chunk,
        "tokens_per_s": chunk / wall if wall else None,
        "faults": {k: after[k] - before[k] for k in ("ru_minflt", "ru_majflt")},
        "row_cache_bytes": len(reader.cache) * layout["row_bytes"],
    }
    reader.close()

    # de-duplicated page set the mmap path must fault in for the same chunk
    unique_rows = set(rows)
    pages = set()
    base, rb, page = layout["offset"], layout["row_bytes"], PAGE
    for r in unique_rows:
        off = base + r * rb
        p0 = off - off % page
        pages.add(p0)
        if (off % page) + rb > page:
            pages.add(p0 + page)
    pages = sorted(pages)

    mm = MmapPle(layout["model"], layout["offset"], layout["length"], layout["row_bytes"])
    if evict:
        fd = os.open(layout["model"], os.O_RDONLY)
        evict_pages(fd, pages)
        os.close(fd)
    before = _usage()
    t0 = time.perf_counter()
    mm.touch_pages(pages)
    wall = time.perf_counter() - t0
    after = _usage()
    mm_res = {
        "chunk_tokens": chunk,
        "unique_rows": len(unique_rows),
        "pages_touched": len(pages),
        "bytes_faulted": len(pages) * page,
        "wall_s": wall,
        "mib_per_s": len(pages) * page / wall / (1024 * 1024) if wall else None,
        "tokens_per_s": chunk / wall if wall else None,
        "faults": {k: after[k] - before[k] for k in ("ru_minflt", "ru_majflt")},
        "evicted": evict,
    }
    mm.close()
    gain = (reader_res["tokens_per_s"] / mm_res["tokens_per_s"]
            if reader_res["tokens_per_s"] and mm_res["tokens_per_s"] else None)
    return {"reader": reader_res, "mmap_cold" if evict else "mmap_warm": mm_res,
            "reader_speedup_vs_mmap": gain}


def scenario_self_test(layout: Dict, io_depth: int, window: int) -> Dict:
    """Read a few known rows through every path and compare the bytes."""
    rows = [0, 1, 2, 999, 1_000_000, 12_345_678, 320_001_535]
    reader = PleReader(layout["model"], layout["offset"], layout["rows"], layout["row_bytes"],
                       cache_rows=64, io_depth=io_depth, window=window)
    mm = MmapPle(layout["model"], layout["offset"], layout["length"], layout["row_bytes"])
    fd = os.open(layout["model"], os.O_RDONLY)
    ok = True
    detail = []
    for r in rows:
        want = reader.gather([r])
        if len(want) != layout["row_bytes"]:
            ok = False
        # second gather must be a cache hit and identical
        again = reader.gather([r])
        raw = os.pread(fd, layout["row_bytes"], layout["offset"] + r * layout["row_bytes"])
        mmv = mm.row(r)
        match = want == raw == again == mmv
        ok = ok and match
        detail.append({"row": r, "reader": want.hex()[:24], "pread": raw.hex()[:24], "match": match})
    os.close(fd)
    reader.close()
    mm.close()
    return {"ok": ok, "rows": detail}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--tokens", nargs="*", default=[],
                    help="token JSON files/globs (row-cache-sim format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default="all",
                    help="all | selftest | decode | prefill (comma-separated)")
    ap.add_argument("--cache-rows", type=int, default=1_000_000)
    ap.add_argument("--io-depth", type=int, default=16)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--evict", dest="evict", action="store_true", default=True,
                    help="fadvise(DONTNEED) the mmap pages so the baseline really faults (default)")
    ap.add_argument("--no-evict", dest="evict", action="store_false")
    ap.add_argument("--decode-tokens", type=int, default=4000,
                    help="tokens in the decode window scenario")
    ap.add_argument("--prefill-tokens", type=int, default=2048,
                    help="tokens in one prefill chunk")
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()

    layout = ple_layout(args.inventory)
    sim = load_sim_module()
    params = sim.load_ple_params(args.inventory)
    print(f"PLE: {layout['model']}", file=sys.stderr)
    print(f"  offset {layout['offset']:,}  length {layout['length']:,} B  "
          f"rows {layout['rows']:,}  row {layout['row_bytes']} B", file=sys.stderr)

    scenarios = [s.strip() for s in args.scenario.split(",") if s.strip()]
    if "all" in scenarios:
        scenarios = ["selftest", "decode", "prefill"]

    result = {
        "meta": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inventory": os.path.abspath(args.inventory),
            "model_file": layout["model"],
            "region_offset": layout["offset"],
            "region_bytes": layout["length"],
            "rows": layout["rows"],
            "row_bytes": layout["row_bytes"],
            "cache_rows": args.cache_rows,
            "io_depth": args.io_depth,
            "window": args.window,
            "decode_tokens": args.decode_tokens,
            "prefill_tokens": args.prefill_tokens,
            "seed": args.seed,
            "rss_kib_start": rss_kib(),
            "io_pressure_start": io_pressure(),
        },
        "scenarios": {},
    }

    sequences: List[Dict] = []
    if args.tokens:
        sequences = load_token_sequences(args.tokens, params)
        result["meta"]["token_files"] = [s["source"] for s in sequences]
    elif any(s in ("decode", "prefill") for s in scenarios):
        raise SystemExit("--tokens is required for the decode/prefill scenarios")

    if "selftest" in scenarios:
        result["scenarios"]["selftest"] = scenario_self_test(layout, args.io_depth, args.window)
        print(f"  selftest ok={result['scenarios']['selftest']['ok']}", file=sys.stderr)
    if "decode" in scenarios:
        result["scenarios"]["decode"] = scenario_decode(
            layout, sequences, cache_rows=args.cache_rows, io_depth=args.io_depth,
            window=args.window, max_tokens=args.decode_tokens, evict=args.evict, args=args)
        s = result["scenarios"]["decode"]["reader_warm"]
        print(f"  decode reader p50={s['p50_us']:.1f}us p99={s['p99_us']:.1f}us", file=sys.stderr)
    if "prefill" in scenarios:
        result["scenarios"]["prefill"] = scenario_prefill(
            layout, sequences, cache_rows=args.cache_rows, io_depth=args.io_depth,
            window=args.window, chunk_tokens=args.prefill_tokens, evict=args.evict, args=args)
        p = result["scenarios"]["prefill"]
        mm = p.get("mmap_cold") or p.get("mmap_warm")
        print(f"  prefill reader {p['reader']['tokens_per_s']:.0f} tok/s "
              f"vs mmap {mm['tokens_per_s']:.0f} tok/s "
              f"(x{p['reader_speedup_vs_mmap']:.2f})", file=sys.stderr)

    result["meta"]["rss_kib_end"] = rss_kib()
    result["meta"]["usage_end"] = _usage()
    result["meta"]["io_pressure_end"] = io_pressure()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
