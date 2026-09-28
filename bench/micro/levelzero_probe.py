#!/usr/bin/env python3
"""Level-Zero graph/submission cost and USM doorbell probe for the Arc Pro B70.

Answers the two open unknowns in docs/research/strata-architecture.md section 7:

  U4  what one command-list submission (the Level-Zero analogue of a captured
      CUDA graph replay) costs on this stack, against launching the same N
      kernels as N separate submissions;
  U5  whether a device-side store into USM *shared* memory is observed by the
      host without an intervening driver entry, and whether a device-side spin
      (the doorbell) is therefore usable as a handoff.

Why this is hand-rolled: the reference box has no SYCL/DPC++ compiler and no
oneAPI SYCL runtime, so SYCL command graphs are unavailable.  Level Zero is the
layer underneath both, and libze_loader.so.1 plus the NEO driver are present, so
the probe drives ze_api.h directly through ctypes.  The two micro kernels are
assembled as SPIR-V words by bench/micro/spirv_kernels.py (no compiler needed).

Usage (single command, self-configures its library paths):

    python3 bench/micro/levelzero_probe.py --out bench/results/<date>-levelzero-submission

Requires only the user-local runtime that ./bongo.sh --runtime user installs
($BONGO_HOME/runtime, overridable with --runtime-dir).
"""
import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spirv_kernels as sk  # noqa: E402


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def configure_runtime(runtime_dir):
    """Point the loader at the user-local NEO/IGC/oneAPI prefix.

    NEO dlopens IGC and IGC dlopens LLVM, so both the prefix lib dirs and
    llvm15/lib must be on the link path; ZEL_LIBRARY_PATH is what the Level Zero
    loader uses to find libze_intel_gpu.so.1.  These are the same variables
    bongo.sh's setup_runtime_env() exports.

    glibc reads LD_LIBRARY_PATH once at process start, so the script re-execs
    itself when the path is not already in place (see ensure_runtime_env).
    """
    libdirs = [
        os.path.join(runtime_dir, "usr", "lib64"),
        os.path.join(runtime_dir, "usr", "lib64", "llvm15", "lib"),
        os.path.join(runtime_dir, "opt", "intel", "oneapi", "redist", "lib"),
    ]
    libdirs = [d for d in libdirs if os.path.isdir(d)]
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs + existing)
    os.environ.setdefault("ZEL_LIBRARY_PATH", os.path.join(runtime_dir, "usr", "lib64"))
    return libdirs


def ensure_runtime_env(runtime_dir):
    """Re-exec with LD_LIBRARY_PATH/ZEL_LIBRARY_PATH set, then continue."""
    libdirs = configure_runtime(runtime_dir)
    if os.environ.get("BONGO_ZE_ENV_READY"):
        return libdirs
    env = dict(os.environ)
    env["BONGO_ZE_ENV_READY"] = "1"
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)


def log(msg):
    print("[probe] %s" % msg, file=sys.stderr, flush=True)


def shell(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return "error: %s" % exc


# --------------------------------------------------------------------------
# ze_api.h bindings (only what the probe needs)
# --------------------------------------------------------------------------
class CtxDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]


class QueueDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("ordinal", ctypes.c_uint32),
                ("index", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("mode", ctypes.c_uint32),
                ("priority", ctypes.c_uint32)]


class ListDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                ("commandQueueGroupOrdinal", ctypes.c_uint32), ("flags", ctypes.c_uint32)]


class ModuleDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("format", ctypes.c_uint32),
                ("inputSize", ctypes.c_size_t), ("pInputModule", ctypes.c_void_p),
                ("pBuildFlags", ctypes.c_char_p), ("pConstants", ctypes.c_void_p)]


class KernelDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pKernelName", ctypes.c_char_p)]


class GroupCount(ctypes.Structure):
    _fields_ = [("groupCountX", ctypes.c_uint32), ("groupCountY", ctypes.c_uint32),
                ("groupCountZ", ctypes.c_uint32)]


class DeviceMemDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("ordinal", ctypes.c_uint32)]


class HostMemDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]


class FenceDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]


class UUID(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint8 * 16)]


class DriverProperties(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                ("driverVersion", ctypes.c_uint32), ("uuid", UUID)]


class DeviceProperties(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("type", ctypes.c_uint32),
                ("vendorId", ctypes.c_uint32), ("deviceId", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("subdeviceId", ctypes.c_uint32), ("coreClockRate", ctypes.c_uint32),
                ("maxMemAllocSize", ctypes.c_uint64), ("maxHardwareContexts", ctypes.c_uint32),
                ("maxCommandQueuePriority", ctypes.c_uint32), ("numThreadsPerEU", ctypes.c_uint32),
                ("physicalEUSimdWidth", ctypes.c_uint32), ("numEUsPerSubslice", ctypes.c_uint32),
                ("numSubslicesPerSlice", ctypes.c_uint32), ("numSlices", ctypes.c_uint32),
                ("timerResolution", ctypes.c_uint64), ("timestampValidBits", ctypes.c_uint32),
                ("kernelTimestampValidBits", ctypes.c_uint32), ("uuid", UUID),
                ("name", ctypes.c_char * 256)]


class QueueGroupProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("maxMemoryFillPatternSize", ctypes.c_size_t), ("numQueues", ctypes.c_uint32)]


class InitDriverTypeDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]


def bind(lib, name, args):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int32
    fn.argtypes = args
    return fn


class Ze:
    STYPE_DEVICE_PROPERTIES = 0x3
    STYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 0x6
    STYPE_CONTEXT_DESC = 0xd
    STYPE_COMMAND_QUEUE_DESC = 0xe
    STYPE_COMMAND_LIST_DESC = 0xf
    STYPE_FENCE_DESC = 0x12
    STYPE_DEVICE_MEM_ALLOC_DESC = 0x15
    STYPE_HOST_MEM_ALLOC_DESC = 0x16
    STYPE_MODULE_DESC = 0x1b
    STYPE_KERNEL_DESC = 0x1d
    STYPE_INIT_DRIVER_TYPE_DESC = 0x00020021
    MODULE_FORMAT_IL_SPIRV = 0
    INIT_DRIVER_TYPE_FLAG_GPU = 1
    UINT64_MAX = 0xFFFFFFFFFFFFFFFF

    def __init__(self, loader_path):
        lib = ctypes.CDLL(loader_path)
        self.lib = lib
        b = lambda n, a: bind(lib, n, a)  # noqa: E731
        cp = ctypes.c_void_p
        self.zeDriverGetApiVersion = b("zeDriverGetApiVersion", [cp, ctypes.POINTER(ctypes.c_uint32)])
        self.zeDriverGetProperties = b("zeDriverGetProperties", [cp, ctypes.POINTER(DriverProperties)])
        self.zeInitDrivers = b("zeInitDrivers", [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(cp),
                                                 ctypes.POINTER(InitDriverTypeDesc)])
        self.zeInit = b("zeInit", [ctypes.c_uint32])
        self.zeDriverGet = b("zeDriverGet", [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(cp)])
        self.zeDeviceGet = b("zeDeviceGet", [cp, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(cp)])
        self.zeDeviceGetProperties = b("zeDeviceGetProperties", [cp, ctypes.POINTER(DeviceProperties)])
        self.zeDeviceGetCommandQueueGroupProperties = b(
            "zeDeviceGetCommandQueueGroupProperties",
            [cp, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(QueueGroupProps)])
        self.zeContextCreate = b("zeContextCreate", [cp, ctypes.POINTER(CtxDesc), ctypes.POINTER(cp)])
        self.zeCommandQueueCreate = b("zeCommandQueueCreate",
                                      [cp, cp, ctypes.POINTER(QueueDesc), ctypes.POINTER(cp)])
        self.zeCommandListCreate = b("zeCommandListCreate",
                                     [cp, cp, ctypes.POINTER(ListDesc), ctypes.POINTER(cp)])
        self.zeCommandListCreateImmediate = b("zeCommandListCreateImmediate",
                                              [cp, cp, ctypes.POINTER(QueueDesc), ctypes.POINTER(cp)])
        self.zeCommandListClose = b("zeCommandListClose", [cp])
        self.zeCommandListReset = b("zeCommandListReset", [cp])
        self.zeCommandListAppendLaunchKernel = b(
            "zeCommandListAppendLaunchKernel",
            [cp, cp, ctypes.POINTER(GroupCount), cp, ctypes.c_uint32, ctypes.POINTER(cp)])
        self.zeCommandListHostSynchronize = b("zeCommandListHostSynchronize", [cp, ctypes.c_uint64])
        self.zeCommandQueueExecuteCommandLists = b(
            "zeCommandQueueExecuteCommandLists", [cp, ctypes.c_uint32, ctypes.POINTER(cp), cp])
        self.zeCommandQueueSynchronize = b("zeCommandQueueSynchronize", [cp, ctypes.c_uint64])
        self.zeModuleCreate = b("zeModuleCreate", [cp, cp, ctypes.POINTER(ModuleDesc),
                                                   ctypes.POINTER(cp), ctypes.POINTER(cp)])
        self.zeModuleBuildLogGetString = b("zeModuleBuildLogGetString",
                                           [cp, ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p])
        self.zeKernelCreate = b("zeKernelCreate", [cp, ctypes.POINTER(KernelDesc), ctypes.POINTER(cp)])
        self.zeKernelSetGroupSize = b("zeKernelSetGroupSize", [cp, ctypes.c_uint32, ctypes.c_uint32,
                                                               ctypes.c_uint32])
        self.zeKernelSetArgumentValue = b("zeKernelSetArgumentValue",
                                          [cp, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p])
        self.zeMemAllocShared = b("zeMemAllocShared", [cp, ctypes.POINTER(DeviceMemDesc),
                                                       ctypes.POINTER(HostMemDesc), ctypes.c_size_t,
                                                       ctypes.c_size_t, cp, ctypes.POINTER(cp)])
        self.zeMemFree = b("zeMemFree", [cp, cp])
        self.zeFenceCreate = b("zeFenceCreate", [cp, ctypes.POINTER(FenceDesc), ctypes.POINTER(cp)])
        self.zeFenceHostSynchronize = b("zeFenceHostSynchronize", [cp, ctypes.c_uint64])
        self.zeFenceQueryStatus = b("zeFenceQueryStatus", [cp])

    # -- convenience ------------------------------------------------------
    def init(self):
        d = InitDriverTypeDesc()
        d.stype = self.STYPE_INIT_DRIVER_TYPE_DESC
        d.flags = self.INIT_DRIVER_TYPE_FLAG_GPU
        n = ctypes.c_uint32(0)
        rc = self.zeInitDrivers(ctypes.byref(n), None, ctypes.byref(d))
        if rc == 0 and n.value:
            arr = (ctypes.c_void_p * n.value)()
            self.zeInitDrivers(ctypes.byref(n), arr, ctypes.byref(d))
            return list(arr), "zeInitDrivers"
        rc = self.zeInit(0)
        if rc != 0:
            raise RuntimeError("Level Zero init failed rc=0x%08x" % (rc & 0xFFFFFFFF))
        n = ctypes.c_uint32(0)
        self.zeDriverGet(ctypes.byref(n), None)
        arr = (ctypes.c_void_p * n.value)()
        self.zeDriverGet(ctypes.byref(n), arr)
        return list(arr), "zeInit"

    def check(self, rc, what):
        if rc != 0:
            raise RuntimeError("%s rc=0x%08x" % (what, rc & 0xFFFFFFFF))
        return rc

    def build_log(self, handle):
        if not handle:
            return ""
        sz = ctypes.c_size_t(0)
        self.zeModuleBuildLogGetString(handle, ctypes.byref(sz), None)
        buf = ctypes.create_string_buffer(sz.value or 1)
        self.zeModuleBuildLogGetString(handle, ctypes.byref(sz), buf)
        return buf.value.decode(errors="replace")


def make_module(ze, ctx, dev, blobs, name):
    """Create one module containing every SPIR-V blob given (concatenated entry points)."""
    # Each blob is a complete module; put them in one module by keeping the first
    # header and appending the bodies is not valid SPIR-V.  Instead build one
    # module per kernel and return the handles.
    modules = []
    for kernel_name, blob in blobs:
        buf = ctypes.create_string_buffer(blob)
        desc = ModuleDesc()
        desc.stype = ze.STYPE_MODULE_DESC
        desc.format = ze.MODULE_FORMAT_IL_SPIRV
        desc.inputSize = len(blob)
        desc.pInputModule = ctypes.cast(buf, ctypes.c_void_p)
        mod = ctypes.c_void_p()
        log = ctypes.c_void_p()
        rc = ze.zeModuleCreate(ctx, dev, ctypes.byref(desc), ctypes.byref(mod), ctypes.byref(log))
        if rc != 0:
            raise RuntimeError("zeModuleCreate(%s) rc=0x%08x log=%s"
                               % (kernel_name, rc & 0xFFFFFFFF, ze.build_log(log.value)))
        modules.append((kernel_name, mod, buf))
    return modules


def make_kernel(ze, mod, name, group=(1, 1, 1)):
    kd = KernelDesc()
    kd.stype = ze.STYPE_KERNEL_DESC
    kd.pKernelName = name.encode() if isinstance(name, str) else name
    kern = ctypes.c_void_p()
    ze.check(ze.zeKernelCreate(mod, ctypes.byref(kd), ctypes.byref(kern)), "zeKernelCreate " + str(name))
    ze.check(ze.zeKernelSetGroupSize(kern, *group), "zeKernelSetGroupSize " + str(name))
    return kern


def set_args(ze, kern, args):
    for i, (size, value) in enumerate(args):
        if isinstance(value, int):
            v = ctypes.c_void_p(value)
            src = ctypes.byref(v)
        else:
            src = ctypes.cast(value, ctypes.c_void_p)
        ze.check(ze.zeKernelSetArgumentValue(kern, i, size, src), "zeKernelSetArgumentValue %d" % i)


HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED = 1 << 1
DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED = 1 << 1


def alloc_shared(ze, ctx, dev, size, host_flags=0, device_flags=0):
    dd = DeviceMemDesc()
    dd.stype = ze.STYPE_DEVICE_MEM_ALLOC_DESC
    dd.flags = device_flags
    hd = HostMemDesc()
    hd.stype = ze.STYPE_HOST_MEM_ALLOC_DESC
    hd.flags = host_flags
    p = ctypes.c_void_p()
    ze.check(ze.zeMemAllocShared(ctx, ctypes.byref(dd), ctypes.byref(hd), size, 64, dev,
                                 ctypes.byref(p)), "zeMemAllocShared")
    return p


def new_queue(ze, ctx, dev, ordinal=0):
    qd = QueueDesc()
    qd.stype = ze.STYPE_COMMAND_QUEUE_DESC
    qd.ordinal = ordinal
    q = ctypes.c_void_p()
    ze.check(ze.zeCommandQueueCreate(ctx, dev, ctypes.byref(qd), ctypes.byref(q)), "zeCommandQueueCreate")
    return q


def new_list_with_nodes(ze, ctx, dev, kernel, group, nodes, ordinal=0, close=True):
    ld = ListDesc()
    ld.stype = ze.STYPE_COMMAND_LIST_DESC
    ld.commandQueueGroupOrdinal = ordinal
    lst = ctypes.c_void_p()
    ze.check(ze.zeCommandListCreate(ctx, dev, ctypes.byref(ld), ctypes.byref(lst)),
             "zeCommandListCreate")
    for _ in range(nodes):
        ze.check(ze.zeCommandListAppendLaunchKernel(lst, kernel, ctypes.byref(group), None, 0, None),
                 "zeCommandListAppendLaunchKernel")
    if close:
        ze.check(ze.zeCommandListClose(lst), "zeCommandListClose")
    return lst


def new_fence(ze, queue):
    # zeFenceCreate takes the *command queue*, not the context.
    fd = FenceDesc()
    fd.stype = ze.STYPE_FENCE_DESC
    f = ctypes.c_void_p()
    ze.check(ze.zeFenceCreate(queue, ctypes.byref(fd), ctypes.byref(f)), "zeFenceCreate")
    return f


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def summarize(samples_ns):
    if not samples_ns:
        return {}
    us = [s / 1000.0 for s in samples_ns]
    return {
        "n": len(us),
        "min_us": min(us),
        "median_us": statistics.median(us),
        "mean_us": statistics.fmean(us),
        "p95_us": sorted(us)[int(0.95 * (len(us) - 1))],
        "max_us": max(us),
    }


def split_cold_warm(samples, warmup):
    return samples[:warmup], samples[warmup:]


# --------------------------------------------------------------------------
# U4 — submission cost
# --------------------------------------------------------------------------
def u4_submission_cost(ze, ctx, dev, store_kern, queue, reps, warmup, node_counts):
    group = GroupCount(1, 1, 1)
    results = {"reps": reps, "warmup": warmup, "series": {}}

    for n in node_counts:
        entry = {}

        # ---- (A) one submission carrying N nodes ("graph replay" analogue) ----
        lst = new_list_with_nodes(ze, ctx, dev, store_kern, group, n)
        submit_ns, total_ns = [], []
        for _ in range(reps + warmup):
            t0 = time.perf_counter_ns()
            ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst), None),
                     "execute")
            t1 = time.perf_counter_ns()
            ze.check(ze.zeCommandQueueSynchronize(queue, ze.UINT64_MAX), "sync")
            t2 = time.perf_counter_ns()
            submit_ns.append(t1 - t0)
            total_ns.append(t2 - t0)
        cs, cw = split_cold_warm(submit_ns, warmup)
        ts, tw = split_cold_warm(total_ns, warmup)
        entry["batched"] = {
            "submissions_per_rep": 1,
            "nodes_per_submission": n,
            "submit_cold": summarize(cs), "submit_warm": summarize(cw),
            "total_cold": summarize(ts), "total_warm": summarize(tw),
            "submit_warm_samples_ns": cw, "total_warm_samples_ns": tw,
        }

        # ---- (B) N separate submissions of one node each ----
        if n:
            lists = [new_list_with_nodes(ze, ctx, dev, store_kern, group, 1) for _ in range(n)]
            submit_ns, total_ns = [], []
            for _ in range(reps + warmup):
                for lst1 in lists:
                    t0 = time.perf_counter_ns()      # per submission, not per rep
                    ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst1), None),
                             "execute")
                    t1 = time.perf_counter_ns()
                    ze.check(ze.zeCommandQueueSynchronize(queue, ze.UINT64_MAX), "sync")
                    submit_ns.append(t1 - t0)
                    total_ns.append(time.perf_counter_ns() - t0)
            # (B) records one sample per submission; (A) one sample per rep = per submission.
            cs, cw = split_cold_warm(submit_ns, warmup * n)
            ts, tw = split_cold_warm(total_ns, warmup * n)
            entry["individual"] = {
                "submissions_per_rep": n,
                "nodes_per_submission": 1,
                "submit_cold": summarize(cs), "submit_warm": summarize(cw),
                "total_cold": summarize(ts), "total_warm": summarize(tw),
                "submit_warm_samples_ns": cw, "total_warm_samples_ns": tw,
            }
        else:
            entry["individual"] = {"skipped": "no nodes; 0 individual submissions is not measurable"}
        results["series"][str(n)] = entry

    # ---- (0) cost of appending a node into a not-yet-closed list (graph capture) ----
    build_entry = {}
    for n in node_counts:
        if not n:
            continue
        ld = ListDesc()
        ld.stype = ze.STYPE_COMMAND_LIST_DESC
        lst = ctypes.c_void_p()
        samples = []
        for _ in range(reps):
            ze.check(ze.zeCommandListCreate(ctx, dev, ctypes.byref(ld), ctypes.byref(lst)),
                     "create")
            t0 = time.perf_counter_ns()
            for _ in range(n):
                ze.check(ze.zeCommandListAppendLaunchKernel(
                    lst, store_kern, ctypes.byref(group), None, 0, None), "append")
            t1 = time.perf_counter_ns()
            ze.check(ze.zeCommandListClose(lst), "close")
            samples.append((t1 - t0) / n)
            ze.check(ze.zeCommandListReset(lst), "reset")
        build_entry[str(n)] = {"nodes": n, "append_us_per_node": summarize(samples),
                               "append_samples_ns": samples}
    results["graph_capture_append"] = build_entry

    # ---- (C) immediate command list: every append is submitted ----
    imm_entry = {}
    try:
        qd = QueueDesc()
        qd.stype = ze.STYPE_COMMAND_QUEUE_DESC
        qd.ordinal = 0
        imm = ctypes.c_void_p()
        ze.check(ze.zeCommandListCreateImmediate(ctx, dev, ctypes.byref(qd), ctypes.byref(imm)),
                 "zeCommandListCreateImmediate")
        for n in (1, 43):
            submit_ns = []
            ze.zeCommandListHostSynchronize(imm, ze.UINT64_MAX)
            for _ in range(reps + warmup):
                t0 = time.perf_counter_ns()
                for _ in range(n):
                    ze.check(ze.zeCommandListAppendLaunchKernel(
                        imm, store_kern, ctypes.byref(group), None, 0, None), "imm append")
                t1 = time.perf_counter_ns()
                ze.zeCommandListHostSynchronize(imm, ze.UINT64_MAX)
                submit_ns.append((t1 - t0) / n)
            cs, cw = split_cold_warm(submit_ns, warmup)
            imm_entry[str(n)] = {"nodes_submitted": n, "append_cold_us": summarize(cs),
                                 "append_warm_us": summarize(cw),
                                 "append_warm_samples_ns": cw}
    except RuntimeError as exc:
        imm_entry["error"] = str(exc)
    results["immediate_list"] = imm_entry
    return results


def crossover(node_counts, u4):
    """Smallest N where one batched submission beats N individual submissions.

    Returns the first N whose warm median (a) is at or below (b), for the pure
    host-side submission cost and for submit+complete separately.
    """
    def first_le(key):
        for n in node_counts:
            if not n:
                continue
            e = u4["series"][str(n)]
            if e["batched"][key]["median_us"] <= e["individual"][key]["median_us"]:
                return n
        return None
    return {"submit_only": first_le("submit_warm"), "submit_plus_complete": first_le("total_warm")}


# --------------------------------------------------------------------------
# U5 — USM flag visibility / device wait
# --------------------------------------------------------------------------
def spin_until(addr, target, timeout_ns, poll=None):
    """Spin on a host-readable int in shared USM.  Returns (observed, ns, polls)."""
    t0 = time.perf_counter_ns()
    polls = 0
    while True:
        polls += 1
        if ctypes.c_int.from_address(addr).value != target:
            return True, time.perf_counter_ns() - t0, polls
        if poll is not None:
            poll()
        if time.perf_counter_ns() - t0 > timeout_ns:
            return False, time.perf_counter_ns() - t0, polls


def u5_flag_visibility(ze, ctx, dev, kern, queue, reps, timeout_ms):
    """kern writes value 1 into a shared-USM flag."""
    flag = alloc_shared(ze, ctx, dev, 64)
    fence = new_fence(ze, queue)
    group = GroupCount(1, 1, 1)
    set_args(ze, kern, [(8, flag.value), (4, 1)])
    lst = new_list_with_nodes(ze, ctx, dev, kern, group, 1)

    timeout_ns = int(timeout_ms * 1e6)
    out = {"timeout_ms": timeout_ms, "reps": reps}

    # (a) host spins with NO Level Zero call in the loop
    samples, misses, polls_total = [], 0, 0
    for _ in range(reps):
        ctypes.c_int.from_address(flag.value).value = 0
        ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst), None), "execute")
        ok, ns, polls = spin_until(flag.value, 0, timeout_ns)
        if not ok:
            misses += 1
        else:
            samples.append(ns)
        polls_total += polls
        ze.check(ze.zeCommandQueueSynchronize(queue, ze.UINT64_MAX), "sync")
    log("U5a done: observed=%d missed=%d" % (len(samples), misses))
    out["pure_host_spin"] = {"observed": len(samples), "missed": misses,
                             "latency_us": summarize(samples), "mean_polls": polls_total / max(reps, 1),
                             "samples_ns": samples}

    # (b) host spins and issues a driver call each iteration (the CUDA cudaEventQuery analogue)
    samples, misses, polls_total = [], 0, 0
    for _ in range(reps):
        ctypes.c_int.from_address(flag.value).value = 0
        ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst), None), "execute")
        ok, ns, polls = spin_until(flag.value, 0, timeout_ns,
                                   poll=lambda: ze.zeFenceQueryStatus(fence))
        if not ok:
            misses += 1
        else:
            samples.append(ns)
        polls_total += polls
        ze.check(ze.zeCommandQueueSynchronize(queue, ctypes.c_uint64(ze.UINT64_MAX)), "sync")
    log("U5b done: observed=%d missed=%d" % (len(samples), misses))
    out["spin_with_fence_query"] = {"observed": len(samples), "missed": misses,
                                    "latency_us": summarize(samples),
                                    "mean_polls": polls_total / max(reps, 1),
                                    "samples_ns": samples}

    # (c) sanity: time from submit to completion of the whole submission
    samples = []
    for _ in range(reps):
        ctypes.c_int.from_address(flag.value).value = 0
        t0 = time.perf_counter_ns()
        ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst), None), "execute")
        ze.check(ze.zeCommandQueueSynchronize(queue, ctypes.c_uint64(ze.UINT64_MAX)), "sync")
        samples.append(time.perf_counter_ns() - t0)
    log("U5c done")
    out["submit_to_complete"] = {"latency_us": summarize(samples), "samples_ns": samples}
    return out


def u5_doorbell(ze, ctx, dev, kern, queue, reps, publish_timeout_ms, release_delay_ms,
                budget):
    """Strata's doorbell: the device publishes a flag, then spins waiting for the host.

    The kernel is bounded (spirv_kernels.module_doorbell_bounded) so a release the
    device never observes ends the spin instead of wedging the GPU; it records
    what it last read into `result`.

    Scenarios:
      control_pre_set   ack=1 before the launch      -> must read 1 (proves the handshake can work)
      no_release        host never writes ack        -> must read 0, and bounds the spin
      plain_store       host stores ack=1 mid-flight, no Level Zero call in between
      store_and_query   host stores ack=1 then calls zeFenceQueryStatus once
      uncached_alloc    like plain_store but the allocation is BIAS_UNCACHED both sides
      after_new_submit  host stores ack=1 then submits a second empty command list
    """
    results = {"reps": reps, "release_delay_ms": release_delay_ms,
               "spin_budget_iterations": budget, "scenarios": {}}

    def run_scenario(name, host_flags, device_flags, mode):
        flag = alloc_shared(ze, ctx, dev, 64, host_flags, device_flags)
        ack = alloc_shared(ze, ctx, dev, 64, host_flags, device_flags)
        result = alloc_shared(ze, ctx, dev, 64, host_flags, device_flags)
        fence = new_fence(ze, queue)
        empty = new_list_with_nodes(ze, ctx, dev, kern, GroupCount(1, 1, 1), 0)
        group = GroupCount(1, 1, 1)
        set_args(ze, kern, [(8, flag.value), (4, 1), (8, ack.value), (8, result.value)])
        lst = new_list_with_nodes(ze, ctx, dev, kern, group, 1)
        publish_latency, saw_release, ack_latency, publish_missed = [], 0, [], 0
        walls, result_words = [], []
        for _ in range(reps):
            ctypes.c_int.from_address(flag.value).value = 0
            ctypes.c_int.from_address(ack.value).value = 1 if mode == "pre_set" else 0
            ctypes.c_int.from_address(result.value).value = -1
            t0 = time.perf_counter_ns()
            ze.check(ze.zeCommandQueueExecuteCommandLists(queue, 1, ctypes.byref(lst),
                                                          fence), "execute")
            if mode != "pre_set":
                ok, ns, _ = spin_until(flag.value, 0, int(publish_timeout_ms * 1e6))
                if ok:
                    publish_latency.append(ns)
                else:
                    publish_missed += 1
                if mode != "no_release":
                    deadline = t0 + int(release_delay_ms * 1e6)
                    while time.perf_counter_ns() < deadline:
                        pass
                    t_release = time.perf_counter_ns()
                    ctypes.c_int.from_address(ack.value).value = 1
                    if mode == "query":
                        ze.zeFenceQueryStatus(fence)
                    elif mode == "new_submit":
                        ze.check(ze.zeCommandQueueExecuteCommandLists(
                            queue, 1, ctypes.byref(empty), None), "execute empty")
            ze.check(ze.zeCommandQueueSynchronize(queue, ze.UINT64_MAX), "sync")
            walls.append(time.perf_counter_ns() - t0)
            got = ctypes.c_int.from_address(result.value).value
            result_words.append(got)
            if got == 1:
                saw_release += 1
                ack_latency.append(time.perf_counter_ns() - t_release if mode != "pre_set" else 0)
        results["scenarios"][name] = {
            "device_saw_host_release": saw_release,
            "device_never_saw_release": reps - saw_release,
            "publish_observed_by_host": len(publish_latency),
            "publish_missed_by_host": publish_missed,
            "publish_latency_us": summarize(publish_latency),
            "publish_samples_ns": publish_latency,
            "kernel_wall_us": summarize(walls),
            "device_result_words": result_words,
        }

    run_scenario("control_pre_set", 0, 0, "pre_set")
    run_scenario("no_release", 0, 0, "none")
    run_scenario("plain_store", 0, 0, "store")
    run_scenario("store_and_query", 0, 0, "query")
    run_scenario("after_new_submit", 0, 0, "new_submit")
    run_scenario("uncached_alloc", HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED,
                 DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED, "store")
    return results


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--runtime-dir", default=os.path.expanduser("~/.bongo/runtime"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--nodes", default="0,1,4,16,43,200")
    ap.add_argument("--u5-reps", type=int, default=200)
    ap.add_argument("--u5-timeout-ms", type=int, default=3000)
    ap.add_argument("--hold-ms", type=float, default=2.0,
                    help="device spin before the host releases the ack, in ms")
    ap.add_argument("--spin-budget", type=int, default=1_000_000,
                    help="bounded device spin iterations (1e6 ~ 34 ms at ~34 ns/iteration)")
    ap.add_argument("--skip-u4", action="store_true")
    ap.add_argument("--skip-u5", action="store_true")
    args = ap.parse_args()

    libdirs = ensure_runtime_env(os.path.expanduser(args.runtime_dir))
    loader = os.path.join(os.path.expanduser(args.runtime_dir), "usr", "lib64", "libze_loader.so.1")
    ze = Ze(loader)

    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "raw")
    os.makedirs(raw, exist_ok=True)

    log("init: loading %s" % loader)
    drivers, how = ze.init()
    n = ctypes.c_uint32(0)
    ze.zeDeviceGet(drivers[0], ctypes.byref(n), None)
    devs = (ctypes.c_void_p * n.value)()
    ze.zeDeviceGet(drivers[0], ctypes.byref(n), devs)
    dev = devs[0]
    props = DeviceProperties()
    props.stype = ze.STYPE_DEVICE_PROPERTIES
    ze.zeDeviceGetProperties(dev, ctypes.byref(props))
    dprops = DriverProperties()
    dprops.stype = ze.STYPE_DEVICE_PROPERTIES
    ze.zeDriverGetProperties(drivers[0], ctypes.byref(dprops))
    api = ctypes.c_uint32(0)
    ze.zeDriverGetApiVersion(drivers[0], ctypes.byref(api))

    gcount = ctypes.c_uint32(0)
    ze.zeDeviceGetCommandQueueGroupProperties(dev, ctypes.byref(gcount), None)
    groups = (QueueGroupProps * gcount.value)()
    for i in range(gcount.value):
        groups[i].stype = ze.STYPE_COMMAND_QUEUE_GROUP_PROPERTIES
    ze.zeDeviceGetCommandQueueGroupProperties(dev, ctypes.byref(gcount), groups)

    env = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "uname": platform.platform(),
        "kernel": shell("uname -r"),
        "os": shell("cat /etc/os-release | head -2"),
        "driver_xe": shell("lspci -nnk -s 03:00.0 | tr '\\n' ' '"),
        "xe_version": shell("cat /sys/module/xe/version 2>/dev/null || true"),
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
            "runtime_dir": os.path.expanduser(args.runtime_dir),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "zel_library_path": os.environ.get("ZEL_LIBRARY_PATH", ""),
            "libze_loader": shell("ls -l %s/libze_loader.so.1* 2>/dev/null | tr '\\n' ' '" % os.path.join(os.path.expanduser(args.runtime_dir), "usr/lib64")),
            "libze_intel_gpu": shell("ls -l %s/libze_intel_gpu.so.1* 2>/dev/null | tr '\\n' ' '" % os.path.join(os.path.expanduser(args.runtime_dir), "usr/lib64")),
            "libigc": shell("ls -l %s/libigc.so.2* 2>/dev/null | tr '\\n' ' '" % os.path.join(os.path.expanduser(args.runtime_dir), "usr/lib64")),
            "libsycl": shell("ls %s/libsycl.so* 2>/dev/null || echo absent" % os.path.join(os.path.expanduser(args.runtime_dir), "usr/lib64")),
            "command": " ".join(sys.argv),
        },
        "method": {
            "graph_analogue": "one zeCommandQueueExecuteCommandLists call over a closed command list of N launch nodes",
            "submission_analogue": "one zeCommandQueueExecuteCommandLists call per node",
            "kernel": "k_store(__global volatile int*, int): flag[0] = value  (SPIR-V assembled by spirv_kernels.py)",
            "clock": "time.perf_counter_ns on the host thread that submits",
        },
    }

    ctx = ctypes.c_void_p()
    cd = CtxDesc()
    cd.stype = ze.STYPE_CONTEXT_DESC
    ze.check(ze.zeContextCreate(drivers[0], ctypes.byref(cd), ctypes.byref(ctx)), "zeContextCreate")

    mods = make_module(ze, ctx, dev, [
        ("k_store", sk.module_store()),
        ("k_wait", sk.module_wait()),
        ("k_doorbell", sk.module_doorbell_bounded(budget=args.spin_budget)),
        ("k_empty", sk.module_empty()),
    ], "probe")
    kernels = {name: make_kernel(ze, mod, name) for name, mod, _ in mods}
    store_kern = kernels["k_store"]

    log("device: %s (api 1.%d, init=%s)" % (props.name.decode(errors="replace"), api.value & 0xFF, how))
    queue = new_queue(ze, ctx, dev, 0)
    results = {"environment": env}

    # The U4 kernel needs valid arguments or it faults the device; point it at a
    # small shared flag for the whole submission series.
    u4_flag = alloc_shared(ze, ctx, dev, 64)
    set_args(ze, store_kern, [(8, u4_flag.value), (4, 1)])

    log("U4: reps=%d warmup=%d nodes=%s" % (args.reps, args.warmup, args.nodes))
    if not args.skip_u4:
        node_counts = [int(x) for x in args.nodes.split(",") if x != ""]
        results["u4_submission"] = u4_submission_cost(ze, ctx, dev, store_kern, queue,
                                                      args.reps, args.warmup, node_counts)
        results["u4_crossover_nodes"] = crossover(node_counts, results["u4_submission"])

    log("U5: reps=%d timeout_ms=%d hold_ms=%s" % (args.u5_reps, args.u5_timeout_ms, args.hold_ms))
    if not args.skip_u5:
        results["u5_flag"] = u5_flag_visibility(ze, ctx, dev, store_kern, queue,
                                                args.u5_reps, args.u5_timeout_ms)
        results["u5_doorbell"] = u5_doorbell(ze, ctx, dev, kernels["k_doorbell"], queue,
                                             args.u5_reps, args.u5_timeout_ms, args.hold_ms,
                                             args.spin_budget)

    path = os.path.join(raw, "levelzero-probe.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps({"wrote": path,
                      "u4_crossover_nodes": results.get("u4_crossover_nodes"),
                      "u5_pure_host_spin_observed":
                          results.get("u5_flag", {}).get("pure_host_spin", {}).get("observed"),
                      "u5_pure_host_spin_missed":
                          results.get("u5_flag", {}).get("pure_host_spin", {}).get("missed"),
                      "u5_doorbell_observed": results.get("u5_doorbell", {}).get("observed")},
                     indent=1))


if __name__ == "__main__":
    main()
