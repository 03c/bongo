#!/usr/bin/env python3
"""Shared library for the bongo benchmark harness.

Stdlib only.  Nothing in here imports from the network or mutates the repo.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "bongo-bench/1"
HARNESS_VERSION = "1.0.0"
NEEDLE = "VAULT-COORD-7391-QXZ"

CORPUS = (
    "The mountain road curved past the reservoir, where the morning fog had not yet lifted. "
    "Engineers logged each pressure reading in a ledger, comparing it against the previous week. "
    "A heron stood motionless at the water's edge, watching the shallow channel for movement. "
    "The turbine hall hummed at a steady frequency, a sound felt more than heard. "
    "Later, the maintenance crew replaced two worn bearings and returned the unit to service. "
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def server_root(base_url):
    """llama-server exposes /props and /tokenize at the root, not under /v1."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        return b[:-3] or b
    return b


def human_bytes(n):
    if n is None:
        return "n/a"
    step = 1024.0
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(v) < step or unit == "TiB":
            return f"{v:.2f} {unit}"
        v /= step


def num(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def read_text(path, limit=None):
    try:
        with open(path, "r", errors="replace") as fh:
            data = fh.read()
        return data if limit is None else data[:limit]
    except Exception:  # noqa: BLE001
        return None


def run_cmd(cmd, timeout=20):
    """Run a host command, never raising. Returns a small record."""
    try:
        p = subprocess.run(
            cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout
        )
        return {
            "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class HttpResult:
    __slots__ = ("status", "body", "elapsed_ms", "json")

    def __init__(self, status, body, elapsed_ms):
        self.status = status
        self.body = body
        self.elapsed_ms = elapsed_ms
        try:
            self.json = json.loads(body.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            self.json = None


def post_json(base_url, path, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResult(resp.status, resp.read(), (time.perf_counter() - t0) * 1000.0)
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), (time.perf_counter() - t0) * 1000.0)
    except Exception as exc:  # noqa: BLE001
        return HttpResult(None, f"{type(exc).__name__}: {exc}".encode(), (time.perf_counter() - t0) * 1000.0)


def get_json(base_url, path, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + path, headers={"Accept": "application/json"}, method="GET"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResult(resp.status, resp.read(), (time.perf_counter() - t0) * 1000.0)
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), (time.perf_counter() - t0) * 1000.0)
    except Exception as exc:  # noqa: BLE001
        return HttpResult(None, f"{type(exc).__name__}: {exc}".encode(), (time.perf_counter() - t0) * 1000.0)


def post_raw(base_url, path, raw_body, content_type, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=raw_body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResult(resp.status, resp.read(), (time.perf_counter() - t0) * 1000.0)
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), (time.perf_counter() - t0) * 1000.0)
    except Exception as exc:  # noqa: BLE001
        return HttpResult(None, f"{type(exc).__name__}: {exc}".encode(), (time.perf_counter() - t0) * 1000.0)


def stream_completion(base_url, payload, timeout):
    """POST a streaming completion.

    Returns ``(ttft_ms, ttfb_ms, total_ms, text, meta)``.  ``ttft_ms`` is the
    time to the first non-empty content token.  Never raises.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    parts = []
    meta = {"error": None, "chunks": 0, "usage": None}
    ttft_ms = None
    ttfb_ms = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if ttfb_ms is None and line:
                    ttfb_ms = (time.perf_counter() - t0) * 1000.0
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                meta["chunks"] += 1
                try:
                    obj = json.loads(chunk)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(obj.get("usage"), dict):
                    meta["usage"] = obj["usage"]
                for choice in obj.get("choices") or []:
                    piece = choice.get("text")
                    if piece is None and isinstance(choice.get("delta"), dict):
                        piece = choice["delta"].get("content")
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        parts.append(piece)
    except urllib.error.HTTPError as exc:
        meta["error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:2000]}"
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
    return ttft_ms, ttfb_ms, (time.perf_counter() - t0) * 1000.0, "".join(parts), meta


# ---------------------------------------------------------------------------
# machine spec
# ---------------------------------------------------------------------------


def collect_machine_spec():
    spec = {
        "hostname": platform.node(),
        "uname": run_cmd("uname -a")["stdout"],
        "os_release": read_text("/etc/os-release"),
        "cpu": run_cmd("lscpu")["stdout"],
        "memory": run_cmd("free -b")["stdout"],
        "meminfo": read_text("/proc/meminfo"),
        "pci_display": "\n".join(
            line
            for line in run_cmd("lspci -nn")["stdout"].splitlines()
            if any(k in line.lower() for k in ("vga", "3d controller", "display controller"))
        ),
        "dri_nodes": sorted(glob.glob("/dev/dri/*")),
        "drm_cards": {},
        "drivers": {},
        "modules": {},
    }
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(card):
            continue
        name = os.path.basename(card)
        spec["drm_cards"][name] = read_text(os.path.join(card, "device", "uevent"))
        drv = os.path.realpath(os.path.join(card, "device", "driver"))
        spec["drivers"][name] = os.path.basename(drv) if drv else None
    for mod in ("xe", "i915", "amdgpu"):
        ver = read_text(f"/sys/module/{mod}/version")
        if ver is not None or os.path.isdir(f"/sys/module/{mod}"):
            spec["modules"][mod] = {
                "version": (ver or "").strip() or None,
                "refcnt": (read_text(f"/sys/module/{mod}/refcnt") or "").strip(),
            }
    return spec


class MemorySampler:
    """Samples GPU VRAM and process system RAM in a background thread.

    Backends are tried in order and the one that worked is recorded.  Nothing
    assumes an Intel tool is installed: ``/proc/<pid>/fdinfo`` and
    ``/proc/<pid>/status`` are always available on Linux.
    """

    def __init__(self, server_pids=None, interval=0.25):
        self.explicit_pids = list(server_pids or [])
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.vram_method = None
        self.ram_method = None

    def server_pids(self):
        if self.explicit_pids:
            return [p for p in self.explicit_pids if os.path.isdir(f"/proc/{p}")]
        pids = []
        for entry in glob.glob("/proc/[0-9]*"):
            cmd = read_text(os.path.join(entry, "cmdline"))
            if not cmd:
                continue
            flat = cmd.replace("\x00", " ")
            if "llama-server" in flat or "llama_server" in flat:
                try:
                    pids.append(int(os.path.basename(entry)))
                except ValueError:
                    pass
        return pids

    def _vram_fdinfo(self):
        total = 0.0
        found = False
        for pid in self.server_pids():
            per_key = {}
            for fd in glob.glob(f"/proc/{pid}/fdinfo/*"):
                data = read_text(fd)
                if not data:
                    continue
                for line in data.splitlines():
                    key, _, value = line.partition(":")
                    if key not in ("drm-total-vram0", "drm-resident-vram0", "drm-memory-vram"):
                        continue
                    parts = value.strip().split()
                    if not parts:
                        continue
                    try:
                        kib = float(parts[0])
                    except ValueError:
                        continue
                    per_key[key] = max(per_key.get(key, 0.0), kib * 1024.0)
                    found = True
            val = per_key.get("drm-resident-vram0") or per_key.get("drm-total-vram0") or per_key.get("drm-memory-vram")
            if val:
                total += val
        return total if found else None

    def _vram_sysfs(self):
        total = 0.0
        found = False
        # Only the Intel card matters here.  On a mixed Intel+AMD box the AMD
        # iGPU also exposes mem_info_vram_used; counting it would be wrong.
        for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
            if "-" in os.path.basename(card):
                continue
            driver = os.path.basename(os.path.realpath(os.path.join(card, "device", "driver")))
            vendor = (read_text(os.path.join(card, "device", "vendor")) or "").strip()
            if driver not in ("xe", "i915") and vendor.lower() != "0x8086":
                continue
            val = read_text(os.path.join(card, "device", "mem_info_vram_used"))
            if val:
                try:
                    total += float(val.strip())
                    found = True
                except ValueError:
                    pass
        return total if found else None

    def _vram_xpu_smi(self):
        res = run_cmd("xpu-smi stats -d 0 -j", timeout=6)
        if res["returncode"] != 0 or not res["stdout"]:
            return None
        try:
            obj = json.loads(res["stdout"])
        except Exception:  # noqa: BLE001
            return None
        found = None

        def walk(node):
            nonlocal found
            if isinstance(node, dict):
                for k, v in node.items():
                    if "memory" in k.lower() and ("used" in k.lower() or "physical" in k.lower()):
                        try:
                            n = float(v)
                            if n > 0:
                                found = max(found or 0, n)
                        except (TypeError, ValueError):
                            pass
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(obj)
        return found

    def sample_vram(self):
        for name, fn in (
            ("fdinfo", self._vram_fdinfo),
            ("sysfs", self._vram_sysfs),
            ("xpu-smi", self._vram_xpu_smi),
        ):
            try:
                val = fn()
            except Exception:  # noqa: BLE001
                val = None
            if val is not None:
                self.vram_method = name
                return val
        self.vram_method = self.vram_method or "unavailable"
        return None

    def sample_ram(self):
        pids = self.server_pids()
        if pids:
            rss = 0.0
            hwm = 0.0
            for pid in pids:
                status = read_text(f"/proc/{pid}/status") or ""
                for line in status.splitlines():
                    if line.startswith("VmRSS:"):
                        rss += float(line.split()[1]) * 1024
                    elif line.startswith("VmHWM:"):
                        hwm += float(line.split()[1]) * 1024
            self.ram_method = "proc_status_sum"
            return {"process_rss": rss, "process_peak_rss": hwm}
        self.ram_method = "host_only"
        return {"process_rss": None, "process_peak_rss": None}

    def host_mem(self):
        info = {}
        for line in (read_text("/proc/meminfo") or "").splitlines():
            key, _, value = line.partition(":")
            parts = value.strip().split()
            if parts:
                try:
                    info[key] = int(parts[0])
                except ValueError:
                    pass
        return info

    def _loop(self):
        while not self._stop.is_set():
            try:
                host = self.host_mem()
                rec = {
                    "t": time.time(),
                    "vram": self.sample_vram(),
                    "ram": self.sample_ram(),
                    "host_mem_available": host.get("MemAvailable"),
                    "host_mem_total": host.get("MemTotal"),
                }
            except Exception as exc:  # noqa: BLE001
                rec = {"t": time.time(), "error": f"{type(exc).__name__}: {exc}"}
            self.samples.append(rec)
            self._stop.wait(self.interval)

    def start(self):
        if self.interval <= 0:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def window_peak(self, start_ts, end_ts):
        window = [s for s in self.samples if start_ts <= s.get("t", 0) <= end_ts]
        vram = [s["vram"] for s in window if s.get("vram")]
        rss = [(s.get("ram") or {}).get("process_rss") for s in window]
        rss = [v for v in rss if v]
        hwm = [(s.get("ram") or {}).get("process_peak_rss") for s in window]
        hwm = [v for v in hwm if v]
        return {
            "vram_peak_bytes": max(vram) if vram else None,
            "vram_method": self.vram_method,
            "system_ram_peak_bytes": max(rss) if rss else None,
            "system_ram_process_hwm_bytes": max(hwm) if hwm else None,
            "system_ram_method": self.ram_method,
            "samples": len(window),
        }


class Tokenizer:
    """Best-effort token counter using the server's ``/tokenize`` endpoint.

    ``/tokenize`` lives at the server root, so the ``/v1`` suffix is stripped.
    """

    def __init__(self, base_url, timeout=60):
        self.base_url = server_root(base_url)
        self.timeout = timeout
        self.available = None

    def count(self, text):
        res = post_json(self.base_url, "/tokenize", {"content": text}, self.timeout)
        if res.status == 200 and isinstance(res.json, dict) and isinstance(res.json.get("tokens"), list):
            self.available = True
            return len(res.json["tokens"])
        self.available = False
        return max(1, len(text) // 4)

    def size_to(self, target_tokens, corpus=CORPUS, seed=" ", hard_max=None):
        """Build a deterministic blob of about ``target_tokens`` tokens.

        If ``hard_max`` is given the result is guaranteed to count at most
        ``hard_max`` tokens (within the tokenizer's own count), so a prompt can
        never overflow the server's ``n_ctx`` once ``max_tokens`` is added.
        """
        text = (corpus + seed).strip()
        for _ in range(12):
            n = self.count(text)
            if n <= 0:
                break
            if hard_max is not None and n > hard_max:
                text = self._rebuild(corpus, seed, max(1, int(len(text) * (hard_max / n) * 0.98)))
                continue
            ratio = target_tokens / n
            if abs(ratio - 1.0) < 0.005:
                break
            text = self._rebuild(corpus, seed, max(1, int(len(text) * ratio)))
        return text

    @staticmethod
    def _rebuild(corpus, seed, length):
        reps = length // max(1, len(corpus)) + 1
        return ((corpus + seed) * reps)[:length]
