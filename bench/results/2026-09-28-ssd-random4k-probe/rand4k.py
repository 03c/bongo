#!/usr/bin/env python3
"""Read-only 4 KiB random-read probe against the reference box NVMe.

Measures single-queue and 8-queue aggregate throughput of 4 KiB O_DIRECT reads
at random offsets in an existing GGUF file. No writes, no cache mutation of the
file contents (O_DIRECT bypasses the page cache).
"""
import mmap
import os
import random
import statistics
import sys
import threading
import time

PATH = sys.argv[1]
SIZE = os.path.getsize(PATH)
BLK = 4096
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

max_off = SIZE - BLK
offs = [random.randrange(0, max_off, BLK) for _ in range(N)]


def read_one(fd, buf, off):
    return os.preadv(fd, [buf], off)


def run(fd, offs):
    buf = mmap.mmap(-1, BLK)
    lat = []
    for off in offs:
        t0 = time.perf_counter()
        read_one(fd, buf, off)
        lat.append((time.perf_counter() - t0) * 1e6)
    return lat


def main():
    fd = os.open(PATH, os.O_RDONLY | os.O_DIRECT)
    try:
        # warm
        run(fd, offs[:64])
        t0 = time.perf_counter()
        lat = run(fd, offs)
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)

    lat.sort()
    print(f"file={PATH}")
    print(f"size={SIZE} blocks=4KiB n={N} threads=1")
    print(f"  elapsed={dt:.3f}s  iops={N/dt:,.0f}  bw={N*BLK/dt/1e6:.1f} MB/s")
    print(f"  lat us: mean={statistics.mean(lat):.1f} p50={lat[len(lat)//2]:.1f} "
          f"p95={lat[int(len(lat)*0.95)]:.1f} p99={lat[int(len(lat)*0.99)]:.1f} max={lat[-1]:.1f}")

    # threaded
    per = N // THREADS
    chunks = [offs[i * per:(i + 1) * per] for i in range(THREADS)]
    results = []
    lock = threading.Lock()

    def worker(ch):
        fd = os.open(PATH, os.O_RDONLY | os.O_DIRECT)
        try:
            r = run(fd, ch)
        finally:
            os.close(fd)
        with lock:
            results.extend(r)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    total = per * THREADS
    results.sort()
    print(f"\nthreads={THREADS} n={total}")
    print(f"  elapsed={dt:.3f}s  iops={total/dt:,.0f}  bw={total*BLK/dt/1e6:.1f} MB/s")
    print(f"  lat us: mean={statistics.mean(results):.1f} p50={results[len(results)//2]:.1f} "
          f"p95={results[int(len(results)*0.95)]:.1f} p99={results[int(len(results)*0.99)]:.1f}")


if __name__ == "__main__":
    main()
