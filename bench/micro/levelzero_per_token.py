#!/usr/bin/env python3
"""Per-token command-list capture cost probe for the Arc Pro B70 (BAS-71).

BAS-70 (bench/micro/levelzero_probe.py) measured that submitting one closed
``ze_command_list`` costs ~1.4 us flat in the node count, and that *appending*
a node into a not-yet-closed list costs 0.88-1.36 us/node.  The surviving claim
from that work is therefore:

    a captured command list holding one token's ~2,000 kernel nodes, replayed
    once per token, saves the per-node append cost on every token after the
    first -- derived at ~1 us/node x ~2,000 nodes = ~2.8 ms/token.

This probe measures that claim directly instead of deriving it.  For each N it
runs two arms over the same N launch nodes:

  Arm A -- captured:     one closed command list of N nodes built ONCE, then
                         replayed ``reps`` times (execute + synchronise).  This
                         is the CUDA-graph-replay analogue: the rebuild cost is
                         not paid per token.
  Arm B -- per-token:    every rep rebuilds the N nodes into a list, closes it,
                         and submits it the same way.  The rebuild (append) cost
                         IS paid per token.  Two variants:
                           B_reset -- reset and re-append into one list object
                           B_fresh -- create/destroy a fresh list object per rep

The headline number is ``Arm B total - Arm A total`` = the per-token capture
cost that a captured list avoids.  A third series (append-only, never submitted)
gives the clean per-node append cost to compare against BAS-70.

Timed on the submitting host thread with ``time.perf_counter_ns``.  Cold == the
first ``--warmup`` reps of each series (list/object first touch); warm == the
remaining ``--reps``.  The very first rep of each series is also recorded.

Usage (self-configures the Level Zero library paths, like the BAS-70 probe):

    python3 bench/micro/levelzero_per_token.py \
        --out bench/results/<date>-per-token-command-list \
        --reps 1000 --warmup 50 --nodes 43,344,2064,4128
"""
import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spirv_kernels as sk  # noqa: E402
import levelzero_probe as lz  # noqa: E402

## Arm B_fresh needs zeCommandListDestroy, which the BAS-70 probe does not bind.
_ZE_CL_DESTROY = None


def _bind_destroy(ze):
    global _ZE_CL_DESTROY
    fn = ze.lib.zeCommandListDestroy
    fn.restype = ctypes.c_int32
    fn.argtypes = [ctypes.c_void_p]
    _ZE_CL_DESTROY = fn
    return fn


def ensure_runtime_env(runtime_dir):
    """Re-exec with LD_LIBRARY_PATH/ZEL_LIBRARY_PATH set (mirrors the BAS-70 probe)."""
    lz.configure_runtime(runtime_dir)
    if os.environ.get("BONGO_ZE_ENV_READY"):
        return
    env = dict(os.environ)
    env["BONGO_ZE_ENV_READY"] = "1"
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)


def _new_open_list(ze, ctx, dev, ordinal=0):
    ld = lz.ListDesc()
    ld.stype = ze.STYPE_COMMAND_LIST_DESC
    ld.commandQueueGroupOrdinal = ordinal
    lst = ctypes.c_void_p()
    ze.check(ze.zeCommandListCreate(ctx, dev, ctypes.byref(ld), ctypes.byref(lst)),
             "zeCommandListCreate")
    return lst


def _append_n(ze, lst, kernel, group, n):
    append = ze.zeCommandListAppendLaunchKernel
    for _ in range(n):
        ze.check(append(lst, kernel, ctypes.byref(group), None, 0, None),
                 "zeCommandListAppendLaunchKernel")


def _timer():
    return time.perf_counter_ns()


def arm_captured(ze, ctx, dev, kernel, group, queue, n, reps, warmup):
    """Arm A: one closed list of N nodes, built once, replayed reps times."""
    lst = lz.new_list_with_nodes(ze, ctx, dev, kernel, group, n, close=True)
    submit, complete = [], []
    first = None
    for _ in range(reps + warmup):
        t0 = _timer()
        ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst), None),
                 "execute")
        t1 = _timer()
        ze.check(ze.zeCommandQueueSynchronize(queue, ze.UINT64_MAX), "sync")
        t2 = _timer()
        if first is None:
            first = {"submit_ns": t1 - t0, "total_ns": t2 - t0}
        submit.append(t1 - t0)
        complete.append(t2 - t0)
    return _series("captured", n, reps, warmup, submit, complete, None, first)


def arm_rebuild_reset(ze, ctx, dev, kernel, group, queue, n, reps, warmup):
    """Arm B (reset reuse): reset + re-append N + close + submit + sync each rep."""
    lst = _new_open_list(ze, ctx, dev)
    capture, submit, complete = [], [], []
    first = None
    for i in range(reps + warmup):
        t0 = _timer()
        if i:
            ze.check(ze.zeCommandListReset(lst), "reset")
        _append_n(ze, lst, kernel, group, n)
        ze.check(ze.zeCommandListClose(lst), "close")
        t1 = _timer()
        ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst), None),
                 "execute")
        t2 = _timer()
        ze.check(ze.zeCommandQueueSynchronize(queue, ze.UINT64_MAX), "sync")
        t3 = _timer()
        if first is None:
            first = {"capture_ns": t1 - t0, "submit_ns": t2 - t1, "total_ns": t3 - t0}
        capture.append(t1 - t0)
        submit.append(t2 - t1)
        complete.append(t3 - t0)
    out = _series("rebuild_reset", n, reps, warmup, submit, complete, capture, first)
    _ZE_CL_DESTROY(lst)
    return out


def arm_rebuild_fresh(ze, ctx, dev, kernel, group, queue, n, reps, warmup):
    """Arm B (fresh): create a fresh closed list per rep, submit, then destroy it."""
    capture, submit, complete = [], [], []
    first = None
    for _ in range(reps + warmup):
        t0 = _timer()
        lst = _new_open_list(ze, ctx, dev)
        _append_n(ze, lst, kernel, group, n)
        ze.check(ze.zeCommandListClose(lst), "close")
        t1 = _timer()
        ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst), None),
                 "execute")
        t2 = _timer()
        ze.check(ze.zeCommandQueueSynchronize(queue, ze.UINT64_MAX), "sync")
        t3 = _timer()
        _ZE_CL_DESTROY(lst)
        if first is None:
            first = {"capture_ns": t1 - t0, "submit_ns": t2 - t1, "total_ns": t3 - t0}
        capture.append(t1 - t0)
        submit.append(t2 - t1)
        complete.append(t3 - t0)
    return _series("rebuild_fresh", n, reps, warmup, submit, complete, capture, first)


def append_only(ze, ctx, dev, kernel, group, n, reps, warmup):
    """Clean per-node append cost: append N nodes into an open list, never submit."""
    lst = _new_open_list(ze, ctx, dev)
    per_node = []
    for i in range(reps + warmup):
        if i:
            ze.check(ze.zeCommandListReset(lst), "reset")
        t0 = _timer()
        _append_n(ze, lst, kernel, group, n)
        t1 = _timer()
        per_node.append((t1 - t0) / n)
    cold = lz.summarize(per_node[:warmup])
    warm = lz.summarize(per_node[warmup:])
    _ZE_CL_DESTROY(lst)
    return {"nodes": n, "append_us_per_node_cold": cold,
            "append_us_per_node_warm": warm,
            "append_warm_samples_ns": per_node[warmup:],
            "first_rep_us_per_node": per_node[0]}


def _series(name, n, reps, warmup, submit, complete, capture, first):
    cs, cw = lz.split_cold_warm(submit, warmup)
    ts, tw = lz.split_cold_warm(complete, warmup)
    out = {
        "arm": name,
        "nodes": n,
        "reps": reps,
        "warmup": warmup,
        "submit_cold": lz.summarize(cs),
        "submit_warm": lz.summarize(cw),
        "total_cold": lz.summarize(ts),
        "total_warm": lz.summarize(tw),
        "submit_warm_samples_ns": cw,
        "total_warm_samples_ns": tw,
        "first_rep": first,
    }
    if capture is not None:
        ks, kw = lz.split_cold_warm(capture, warmup)
        out["capture_cold"] = lz.summarize(ks)
        out["capture_warm"] = lz.summarize(kw)
        out["capture_warm_samples_ns"] = kw
        if n:
            out["capture_warm_us_per_node"] = kw and lz.summarize([s / n for s in kw])
    return out


def _delta_us_per_token(a, b):
    """b - a, in us/token, from the warm totals."""
    if not a or not b:
        return None
    return b["total_warm"]["median_us"] - a["total_warm"]["median_us"]


def measure_n(ze, ctx, dev, kernel, group, queue, n, reps, warmup, variants):
    entry = {
        "nodes": n,
        "arm_A_captured": arm_captured(ze, ctx, dev, kernel, group, queue, n, reps, warmup)
                                   if variants.get("captured") else None,
        "append_only": append_only(ze, ctx, dev, kernel, group, n, reps, warmup)
                       if variants.get("append_only") else None,
    }
    if variants.get("rebuild_reset"):
        entry["arm_B_reset"] = arm_rebuild_reset(ze, ctx, dev, kernel, group, queue,
                                                 n, reps, warmup)
    if variants.get("rebuild_fresh"):
        entry["arm_B_fresh"] = arm_rebuild_fresh(ze, ctx, dev, kernel, group, queue,
                                                 n, reps, warmup)

    # per-token delta (captured vs each rebuild variant) and the prediction
    deltas = {}
    a = entry.get("arm_A_captured")
    for key, label in (("arm_B_reset", "rebuild_reset"), ("arm_B_fresh", "rebuild_fresh")):
        b = entry.get(key)
        if a and b:
            deltas[label + "_minus_captured"] = _delta_us_per_token(a, b)
    if a:
        deltas["captured_total_us_per_token"] = a["total_warm"]["median_us"]
    if entry.get("append_only"):
        per_node = entry["append_only"]["append_us_per_node_warm"].get("median_us")
        deltas["append_only_median_us_per_node"] = per_node
        if per_node is not None:
            deltas["predicted_delta_us_per_token"] = per_node * n
    entry["deltas"] = deltas
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--runtime-dir", default=os.path.expanduser("~/.bongo/runtime"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--nodes", default="43,344,2064,4128",
                    help="43x1, 43x8, 43x48, 43x48x2 (43 = Strata block nodes)")
    ap.add_argument("--skip-captured", action="store_true")
    ap.add_argument("--skip-rebuild-reset", action="store_true")
    ap.add_argument("--skip-rebuild-fresh", action="store_true")
    ap.add_argument("--skip-append-only", action="store_true")
    args = ap.parse_args()

    ensure_runtime_env(os.path.expanduser(args.runtime_dir))
    runtime_dir = os.path.expanduser(args.runtime_dir)
    loader = os.path.join(runtime_dir, "usr", "lib64", "libze_loader.so.1")
    ze = lz.Ze(loader)
    _bind_destroy(ze)

    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "raw")
    os.makedirs(raw, exist_ok=True)

    lz.log("init: loading %s" % loader)
    drivers, how = ze.init()
    n = ctypes.c_uint32(0)
    ze.zeDeviceGet(drivers[0], ctypes.byref(n), None)
    devs = (ctypes.c_void_p * n.value)()
    ze.zeDeviceGet(drivers[0], ctypes.byref(n), devs)
    dev = devs[0]
    props = lz.DeviceProperties()
    props.stype = ze.STYPE_DEVICE_PROPERTIES
    ze.zeDeviceGetProperties(dev, ctypes.byref(props))
    dprops = lz.DriverProperties()
    dprops.stype = ze.STYPE_DEVICE_PROPERTIES
    ze.zeDriverGetProperties(drivers[0], ctypes.byref(dprops))
    api = ctypes.c_uint32(0)
    ze.zeDriverGetApiVersion(drivers[0], ctypes.byref(api))

    gcount = ctypes.c_uint32(0)
    ze.zeDeviceGetCommandQueueGroupProperties(dev, ctypes.byref(gcount), None)
    groups = (lz.QueueGroupProps * gcount.value)()
    for i in range(gcount.value):
        groups[i].stype = ze.STYPE_COMMAND_QUEUE_GROUP_PROPERTIES
    ze.zeDeviceGetCommandQueueGroupProperties(dev, ctypes.byref(gcount), groups)

    env = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "uname": platform.platform(),
        "kernel": lz.shell("uname -r"),
        "os": lz.shell("cat /etc/os-release | head -2"),
        "driver_xe": lz.shell("lspci -nnk -s 03:00.0 | tr '\\n' ' '"),
        "xe_version": lz.shell("cat /sys/module/xe/version 2>/dev/null || true"),
        "device": {
            "name": props.name.decode(errors="replace"),
            "vendorId": hex(props.vendorId),
            "deviceId": hex(props.deviceId),
            "coreClockRate_mhz": props.coreClockRate,
            "numSlices": props.numSlices,
            "numSubslicesPerSlice": props.numSubslicesPerSlice,
            "numEUsPerSubslice": props.numEUsPerSubslice,
            "api_version": "1.%d" % (api.value & 0xFF),
            "driver_version_raw": dprops.driverVersion,
            "init_path": how,
        },
        "queue_groups": [{"ordinal": i, "flags": hex(g.flags), "numQueues": g.numQueues}
                         for i, g in enumerate(groups)],
        "runtime": {
            "runtime_dir": runtime_dir,
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "zel_library_path": os.environ.get("ZEL_LIBRARY_PATH", ""),
            "libze_loader": lz.shell("ls -l %s/usr/lib64/libze_loader.so.1* 2>/dev/null | tr '\\n' ' '" % runtime_dir),
            "libze_intel_gpu": lz.shell("ls -l %s/usr/lib64/libze_intel_gpu.so.1* 2>/dev/null | tr '\\n' ' '" % runtime_dir),
            "libigc": lz.shell("ls -l %s/usr/lib64/libigc.so.2* 2>/dev/null | tr '\\n' ' '" % runtime_dir),
            "libsycl": lz.shell("ls %s/usr/lib64/libsycl.so* 2>/dev/null || echo absent" % runtime_dir),
            "command": " ".join(sys.argv),
        },
        "method": {
            "arm_A_captured": "one closed command list of N launch nodes built once, replayed (execute+sync) per rep",
            "arm_B_rebuild_reset": "per rep: zeCommandListReset + append N nodes + close + submit + sync (list object reused)",
            "arm_B_rebuild_fresh": "per rep: zeCommandListCreate + append N nodes + close + submit + sync + destroy",
            "append_only": "append N nodes into an open list, timed, never closed or submitted (BAS-70 capture analogue)",
            "kernel": "k_store(__global volatile int*, int): flag[0] = value  (SPIR-V assembled by spirv_kernels.py)",
            "clock": "time.perf_counter_ns on the host thread that submits",
            "cold": "first --warmup reps of each series",
            "warm": "remaining --reps reps of each series",
        },
    }

    ctx = ctypes.c_void_p()
    cd = lz.CtxDesc()
    cd.stype = ze.STYPE_CONTEXT_DESC
    ze.check(ze.zeContextCreate(drivers[0], ctypes.byref(cd), ctypes.byref(ctx)), "zeContextCreate")

    mods = lz.make_module(ze, ctx, dev, [("k_store", sk.module_store())], "per_token")
    kernels = {name: lz.make_kernel(ze, mod, name) for name, mod, _ in mods}
    store_kern = kernels["k_store"]
    queue = lz.new_queue(ze, ctx, dev, 0)
    group = lz.GroupCount(1, 1, 1)

    # Every launch must have valid arguments or it faults the device (BAS-70: device lost).
    flag = lz.alloc_shared(ze, ctx, dev, 64)
    lz.set_args(ze, store_kern, [(8, flag.value), (4, 1)])

    variants = {
        "captured": not args.skip_captured,
        "append_only": not args.skip_append_only,
        "rebuild_reset": not args.skip_rebuild_reset,
        "rebuild_fresh": not args.skip_rebuild_fresh,
    }
    node_counts = [int(x) for x in args.nodes.split(",") if x != ""]
    results = {"environment": env, "variants": variants, "series": {}}
    lz.log("device: %s (api 1.%d, init=%s)"
           % (props.name.decode(errors="replace"), api.value & 0xFF, how))

    for node_n in node_counts:
        lz.log("N=%d: reps=%d warmup=%d" % (node_n, args.reps, args.warmup))
        results["series"][str(node_n)] = measure_n(
            ze, ctx, dev, store_kern, group, queue, node_n,
            args.reps, args.warmup, variants)

    path = os.path.join(raw, "per-token-command-list.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    ze.zeMemFree(ctx, flag)
    print(json.dumps({"wrote": path, "series": list(results["series"])}, indent=1))


if __name__ == "__main__":
    main()
