#!/usr/bin/env python3
"""Sample a constrained llama-server's memory and SSD I/O while a run executes.

Measurement support for the capacity-sensitivity study (BAS-69).  Reads only
``/proc`` and cgroup-v2 ``/sys/fs`` files; it never touches the server.

Records one JSON line per tick so a crashed run still leaves the samples behind:

    {pid, t, io:{rchar,wchar,read_bytes,write_bytes},
     stat:{minflt,majflt}, status:{VmRSS_kb,VmHWM_kb},
     cg:{memory.current, memory.max, memory.events:{...},
         memory.stat:{pgmajfault,file,anon,...}},
     vram:{drm-resident-vram0}}

Usage:
    python3 bench/capacity_sampler.py --pid-file ~/.bongo/run/llama-server.pid \
        --out <results-dir>/samples.jsonl --interval 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time


def read_text(path):
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def read_kv(path):
    """Parse a flat `key value` / `key: value` proc file into a dict."""
    data = read_text(path)
    if data is None:
        return {}
    out = {}
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, _, value = line.partition(":")
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        key = key.strip()
        value = value.strip()
        if value.endswith(" kB"):
            value = value[:-3].strip()
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def read_stat(pid):
    """Parse /proc/<pid>/stat; the comm field may contain spaces, so split on
    the last ')' and index from there."""
    data = read_text(f"/proc/{pid}/stat")
    if not data or ")" not in data:
        return {}
    tail = data[data.rfind(")") + 1:].split()
    names = [
        "state", "ppid", "pgrp", "session", "tty_nr", "tpgid", "flags",
        "minflt", "cminflt", "majflt", "cmajflt",
    ]
    out = {}
    for i, name in enumerate(names):
        if i >= len(tail):
            break
        try:
            out[name] = int(tail[i])
        except ValueError:
            out[name] = tail[i]
    return out


def cgroup_dir(pid):
    data = read_text(f"/proc/{pid}/cgroup")
    if not data:
        return None
    for line in data.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            rel = parts[2].strip().lstrip("/")
            return os.path.join("/sys/fs/cgroup", rel) if rel else "/sys/fs/cgroup"
    return None


def vram_from_fdinfo(pid):
    total = 0
    found = False
    for fd in glob.glob(f"/proc/{pid}/fdinfo/*"):
        data = read_text(fd)
        if not data:
            continue
        for line in data.splitlines():
            key, _, value = line.partition(":")
            if key not in ("drm-resident-vram0", "drm-total-vram0", "drm-memory-vram"):
                continue
            parts = value.strip().split()
            if not parts:
                continue
            try:
                kib = float(parts[0])
            except ValueError:
                continue
            total = max(total, int(kib * 1024))
            found = True
    return total if found else None


def read_pid(pid_file):
    data = read_text(pid_file)
    if not data:
        return None
    try:
        pid = int(data.strip())
    except ValueError:
        return None
    return pid if os.path.isdir(f"/proc/{pid}") else None


def snapshot(pid):
    if pid is None:
        return None
    io = read_kv(f"/proc/{pid}/io")
    stat = read_stat(pid)
    status = read_kv(f"/proc/{pid}/status")
    rec = {
        "pid": pid,
        "t": time.time(),
        "io": io,
        "stat": {
            "minflt": stat.get("minflt"),
            "cminflt": stat.get("cminflt"),
            "majflt": stat.get("majflt"),
            "cmajflt": stat.get("cmajflt"),
        },
        "status": {
            "VmRSS_kb": status.get("VmRSS"),
            "VmHWM_kb": status.get("VmHWM"),
            "VmSize_kb": status.get("VmSize"),
        },
        "vram": {"drm-resident-vram0": vram_from_fdinfo(pid)},
    }
    cg = cgroup_dir(pid)
    if cg:
        events = read_kv(os.path.join(cg, "memory.events"))
        mstat = read_kv(os.path.join(cg, "memory.stat"))
        rec["cgroup"] = {
            "path": cg,
            "memory.current": read_text(os.path.join(cg, "memory.current")),
            "memory.max": (read_text(os.path.join(cg, "memory.max")) or "").strip(),
            "memory.events": {k: v for k, v in events.items() if not isinstance(v, str) or k in (
                "max", "oom", "oom_kill", "high", "low")},
            "memory.stat": {
                k: mstat.get(k)
                for k in (
                    "pgmajfault",
                    "pgfault",
                    "file",
                    "anon",
                    "workingset_refault_file",
                    "workingset_refault_anon",
                )
            },
        }
    return rec


def main():
    ap = argparse.ArgumentParser(description="sample constrained llama-server memory/IO")
    ap.add_argument("--pid-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = run until SIGTERM")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    deadline = time.time() + args.duration if args.duration else None
    seen = False
    with open(args.out, "w") as out:
        while True:
            if deadline is not None and time.time() >= deadline:
                break
            pid = read_pid(args.pid_file)
            rec = snapshot(pid)
            if rec is not None:
                seen = True
                out.write(json.dumps(rec) + "\n")
                out.flush()
            else:
                # record a gap so a missing PID is visible, but only after we
                # have found the server at least once
                if seen:
                    out.write(json.dumps({"t": time.time(), "pid": None}) + "\n")
                    out.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
