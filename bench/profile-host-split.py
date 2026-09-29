#!/usr/bin/env python3
"""Decompose the host/CPU term of a warm-prefix (cached) delta turn.

This is the M4.1 measurement (BAS-144).  The M3.6 profile (BAS-130) split the
512-token delta turn into GPU-busy and "everything else" and showed the
"everything else" is ~80% of the turn.  This probe names the pieces of that
"everything else" from the *host* side, using only `/proc`:

  * per-thread CPU time (`/proc/<pid>/task/<tid>/stat`) split into the CPU
    backend worker threads and the main/host thread;
  * storage bytes read and major/minor page faults (`/proc/<pid>/io`,
    `/proc/<pid>/stat`) -- the model weights of host-resident MoE layers are
    mmap'd from the GGUF, so a page-cache eviction shows up here as a
    multi-gigabyte read during a single delta turn;
  * the server-reported `prompt_ms` and the streaming client wall time, on the
    same request, so the CPU split is tied to the turn it belongs to.

The classification is data-driven: the CPU backend workers are the N busiest
non-main threads during the cold prefill of the same point, where the work is
unambiguously on the CPU.  That set is then summed for the measured turn.

Raw files are committed; the tables live in
`docs/research/host-cpu-decomposition.md`.  Use it through
`bench/run-warm-prefix-profile.sh` (``BONGO_HOST_SPLIT=1``), which owns the
server lifecycle and the single-GPU flock, or standalone against a running
server.

Stdlib only; imports the shared bench_lib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import CORPUS, Tokenizer, get_json, now_iso, read_text, server_root  # noqa: E402
from harness import streaming_measure  # noqa: E402

BASE = os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BONGO_MODEL", "bongo-iq2_xs")

CLK_TCK = os.sysconf("SC_CLK_TCK")


# --------------------------------------------------------------------------
# /proc sampling
# --------------------------------------------------------------------------
def read_task_stats(pid):
    """tid -> {comm, utime, stime, starttime} for every thread of pid."""
    out = {}
    task_dir = f"/proc/{pid}/task"
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return out
    for tid in tids:
        try:
            raw = read_text(os.path.join(task_dir, tid, "stat")) or ""
        except OSError:
            continue
        rp = raw.rfind(")")
        lp = raw.find("(")
        if rp < 0 or lp < 0:
            continue
        comm = raw[lp + 1:rp]
        fields = raw[rp + 2:].split()
        if len(fields) < 22:
            continue
        try:
            out[int(tid)] = {
                "comm": comm,
                "utime": int(fields[11]),
                "stime": int(fields[12]),
                "starttime": int(fields[19]),
                "minflt": int(fields[7]),
                "majflt": int(fields[9]),
            }
        except (ValueError, IndexError):
            continue
    return out


def read_proc_io(pid):
    rec = {}
    raw = read_text(f"/proc/{pid}/io") or ""
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                rec[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return rec


def read_proc_status(pid):
    rec = {}
    raw = read_text(f"/proc/{pid}/status") or ""
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            rec[k.strip()] = v.strip()
    return rec


def snapshot(pid):
    st = read_proc_status(pid)
    return {
        "ts": time.time(),
        "threads": read_task_stats(pid),
        "io": read_proc_io(pid),
        "status": st,
    }


def cpu_ms(thread):
    return (thread["utime"] + thread["stime"]) * 1000.0 / CLK_TCK


def diff_snapshot(before, after):
    """CPU/io deltas plus the per-thread table."""
    bt = before["threads"]
    at = after["threads"]
    per_thread = []
    seen = set(bt) | set(at)
    for tid in seen:
        b = bt.get(tid)
        a = at.get(tid)
        if a is None:
            continue  # thread exited (or was never present) -- skip; note below
        b_cpu = cpu_ms(b) if b else 0.0
        a_cpu = cpu_ms(a)
        per_thread.append(
            {
                "tid": tid,
                "comm": a["comm"],
                "cpu_ms": round(a_cpu - b_cpu, 2),
                "minflt": a["minflt"] - (b["minflt"] if b else 0),
                "majflt": a["majflt"] - (b["majflt"] if b else 0),
                "new": b is None,
            }
        )
    per_thread.sort(key=lambda r: r["cpu_ms"], reverse=True)

    io_b = before["io"]
    io_a = after["io"]
    io_delta = {k: io_a.get(k, 0) - io_b.get(k, 0) for k in set(io_a) | set(io_b)}

    return {
        "wall_ms": round((after["ts"] - before["ts"]) * 1000.0, 2),
        "cpu_total_ms": round(sum(t["cpu_ms"] for t in per_thread), 2),
        "majflt": sum(t["majflt"] for t in per_thread),
        "minflt": sum(t["minflt"] for t in per_thread),
        "io": io_delta,
        "threads": per_thread,
    }


def server_record(pid):
    root = server_root(BASE)
    props_res = get_json(root, "/props", 20)
    rec = {
        "base_url": BASE,
        "server_root": root,
        "props": props_res.json if isinstance(props_res.json, dict) else None,
        "props_status": props_res.status,
        "pid": pid,
        "flags": None,
    }
    if pid:
        raw = read_text(f"/proc/{pid}/cmdline")
        if raw:
            rec["flags"] = [x for x in raw.split("\x00") if x]
    return rec


def parse_n_threads(server):
    flags = server.get("flags") or []
    for i, f in enumerate(flags):
        if f in ("--threads", "-t") and i + 1 < len(flags):
            try:
                return int(flags[i + 1])
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def run_case(tk, pid, label, prompt, cache_prompt, max_tokens=1, timeout=1800,
             ignore_eos=True, extra=None):
    before = snapshot(pid) if pid else {"ts": time.perf_counter(), "threads": {}, "io": {}, "status": {}}
    t0 = time.perf_counter()
    rec = streaming_measure(
        BASE,
        MODEL,
        prompt,
        max_tokens,
        timeout,
        cache_prompt=bool(cache_prompt),
        extra={"ignore_eos": ignore_eos} if (ignore_eos and extra is None) else extra,
    )
    client_wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    after = snapshot(pid) if pid else {"ts": time.perf_counter(), "threads": {}, "io": {}, "status": {}}
    split = diff_snapshot(before, after)

    out = dict(rec)
    out["client_wall_ms"] = client_wall_ms
    out["label"] = label
    out["cache_prompt"] = bool(cache_prompt)
    out["max_tokens"] = max_tokens
    out["ts"] = time.time()
    out["proc"] = split
    out["rss_kb"] = after["status"].get("VmRSS")
    print(
        f"{label:28s} status={out.get('status')} "
        f"prompt_n={out.get('prompt_tokens')} cache_n={out.get('cache_n')} "
        f"prompt_ms={out.get('prompt_ms')} wall_ms={out.get('wall_ms')} "
        f"cpu_ms={split['cpu_total_ms']} read_bytes={split['io'].get('read_bytes')} "
        f"majflt={split['majflt']}",
        flush=True,
    )
    return out


def make_delta(tk, tokens, tag):
    seed = f" | warm-prefix delta {tag} "
    return tk.size_to(tokens, corpus="The reviewer noted the following delta. " + CORPUS, seed=seed)


def classify_workers(cold_case, main_tid, n_threads):
    """Name the CPU backend worker threads from the cold prefill, where the
    work is unambiguously CPU-bound.  The workers are the busiest non-main
    threads; take the known thread count when the server argv names one."""
    threads = [t for t in cold_case["proc"]["threads"] if t["tid"] != main_tid]
    threads.sort(key=lambda t: t["cpu_ms"], reverse=True)
    if n_threads:
        return [t["tid"] for t in threads[:n_threads]]
    wall = cold_case["proc"]["wall_ms"]
    return [t["tid"] for t in threads if t["cpu_ms"] >= 0.20 * wall]


def apply_roles(case, main_tid, worker_tids):
    worker_set = set(worker_tids)
    for t in case["proc"]["threads"]:
        if t["tid"] == main_tid:
            t["role"] = "main"
        elif t["tid"] in worker_set:
            t["role"] = "worker"
        else:
            t["role"] = "other"
    proc = case["proc"]
    proc["cpu_main_ms"] = round(sum(t["cpu_ms"] for t in proc["threads"] if t["role"] == "main"), 2)
    proc["cpu_workers_ms"] = round(sum(t["cpu_ms"] for t in proc["threads"] if t["role"] == "worker"), 2)
    proc["cpu_other_ms"] = round(sum(t["cpu_ms"] for t in proc["threads"] if t["role"] == "other"), 2)
    return case


def main():
    ap = argparse.ArgumentParser(description="host/CPU split of the cached delta turn")
    ap.add_argument("--prefixes", default="16384")
    ap.add_argument("--deltas", default="512")
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("BONGO_CTX", "131072")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=os.environ.get("BONGO_PROFILE_LABEL", "baseline"))
    ap.add_argument("--server-pid", type=int, default=int(os.environ.get("BONGO_SERVER_PID", "0")) or None)
    ap.add_argument("--flags-note", default=os.environ.get("BONGO_PROFILE_FLAGS_NOTE", ""))
    ap.add_argument("--warmup-tokens", type=int, default=256)
    args = ap.parse_args()

    if not args.server_pid:
        print("ERROR: --server-pid is required (per-thread CPU accounting needs it)", file=sys.stderr)
        return 2

    tk = Tokenizer(BASE, timeout=1800)
    if not tk.available:
        tk.count("warmup")

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]
    server = server_record(args.server_pid)
    n_threads = parse_n_threads(server)

    results = {
        "schema": "bongo.warm-prefix-host-split.v1",
        "generated_at": now_iso(),
        "label": args.label,
        "base_url": BASE,
        "model": MODEL,
        "deltas": deltas,
        "prefixes": prefixes,
        "clk_tck": CLK_TCK,
        "server_n_threads": n_threads,
        "server": server,
        "flags_note": args.flags_note,
        "warmup": None,
        "points": [],
    }

    results["warmup"] = run_case(tk, args.server_pid, "warmup", tk.size_to(args.warmup_tokens), False, 1, 600)

    for p in prefixes:
        hard_max = max(1, args.ctx - max(deltas) - 16)
        p_text = tk.size_to(p, hard_max=hard_max)
        real_p = tk.count(p_text)
        if real_p > hard_max:
            real_p = hard_max
        print(f"\n== prefix target {p} (actual {real_p}) ==", flush=True)
        point = {"prefix_target": p, "prefix_tokens": real_p, "cases": [], "worker_tids": []}

        cold = run_case(tk, args.server_pid, f"cold_p{real_p}", p_text, False)
        point["cases"].append(cold)
        worker_tids = classify_workers(cold, args.server_pid, n_threads)
        point["worker_tids"] = worker_tids
        print(f"   worker tids ({len(worker_tids)}): {worker_tids}", flush=True)

        for case in point["cases"]:
            apply_roles(case, args.server_pid, worker_tids)

        point["cases"].append(
            apply_roles(
                run_case(tk, args.server_pid, f"hit_p{real_p}", p_text, True),
                args.server_pid, worker_tids,
            )
        )
        for d in deltas:
            d_text = make_delta(tk, d, f"p{real_p}_d{d}")
            grow_text = p_text + d_text
            real_grow = tk.count(grow_text)
            if real_grow + 1 > args.ctx:
                print(f"skip grow d={d}: {real_grow}+1 > ctx {args.ctx}", flush=True)
                continue
            case = run_case(tk, args.server_pid, f"grow_p{real_p}_d{d}", grow_text, True)
            case["delta_target"] = d
            case["prompt_real_tokens"] = real_grow
            point["cases"].append(apply_roles(case, args.server_pid, worker_tids))

        point["cases"].append(
            apply_roles(
                run_case(tk, args.server_pid, f"hit2_p{real_p}", p_text, True),
                args.server_pid, worker_tids,
            )
        )
        dec = run_case(tk, args.server_pid, f"decode_p{real_p}", p_text, True,
                       max_tokens=args.decode_tokens, ignore_eos=True)
        dec["decode_tokens_requested"] = args.decode_tokens
        point["cases"].append(apply_roles(dec, args.server_pid, worker_tids))

        results["points"].append(point)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
