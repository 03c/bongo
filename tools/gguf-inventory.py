#!/usr/bin/env python3
"""GGUF tensor inventory for the bongo model tiers.

Reads only the GGUF header + tensor table of a (possibly remote, split) GGUF file,
computes every tensor's exact byte size from its dimensions and ggml type, and
groups the result into the buckets the buffer-placement plan needs (experts,
shared experts, n-gram table, everything else).

Two sources are supported:

  * a Hugging Face repo file, fetched with HTTP range requests only
    (``--repo ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`` + ``--file X.gguf``,
    or the shorthand ``hf:repo/file.gguf``);
  * a local GGUF file (``--local /path/to/file.gguf``).

Split GGUFs (``-00001-of-000NN.gguf``) are resolved automatically: every shard is
parsed and the tensors are merged into one inventory with a per-tensor shard tag.

The amount of data actually downloaded is reported (``bytes_fetched``) so the
range-read claim is auditable.

Examples
--------

    # Network inventory of one tier (both shards), JSON + human summary
    tools/gguf-inventory.py --repo ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF \\
        --file Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \\
        --json out/iq2xs.json --summary

    # Local file (no network)
    tools/gguf-inventory.py --local /models/tier.gguf --summary

    # Cross-check the published SHA256SUMS / release-manifest.json
    tools/gguf-inventory.py hf:ukisai/.../X-00001-of-00002.gguf --cross-check-manifest manifest.json

Requires Python 3.9+; standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HF_BASE = "https://huggingface.co"
HF_RESOLVE = HF_BASE + "/{repo}/resolve/main/{file}"
HF_API = HF_BASE + "/api/models/{repo}"

# --------------------------------------------------------------------------- #
# ggml type table: type_id -> (name, block_size, type_size_in_bytes)
#
# block_size = number of elements covered by one stored block.
# type_size  = stored bytes for one block.
# Sizes are from llama.cpp's ggml_type_size()/ggml_blck_size() tables.  Type 42
# (block 256, 72 bytes) is not in the older upstream tables; its parameters here
# were derived from the GGUF layout itself (see --verify-layout) and the name
# "Q2_0" was confirmed against tensor-allocation/*.rco-allocation.txt in the
# model repo, which labels every blk.N.ffn_down_exps.weight of that type "Q2_0".
# --------------------------------------------------------------------------- #
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    31: ("Q4_0_4_4", 32, 18),
    32: ("Q4_0_4_8", 32, 18),
    33: ("Q4_0_8_8", 32, 18),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    36: ("IQ4_NL_4_4", 32, 18),
    37: ("IQ4_NL_4_8", 32, 18),
    38: ("IQ4_NL_8_8", 32, 18),
    39: ("MXFP4", 32, 17),
    42: ("Q2_0", 256, 72),
}

# gguf metadata value types
_GGUF_UINT8, _GGUF_INT8, _GGUF_UINT16, _GGUF_INT16 = 0, 1, 2, 3
_GGUF_UINT32, _GGUF_INT32, _GGUF_FLOAT32, _GGUF_BOOL = 4, 5, 6, 7
_GGUF_STRING, _GGUF_ARRAY, _GGUF_UINT64, _GGUF_INT64, _GGUF_FLOAT64 = 8, 9, 10, 11, 12

_FIXED = {
    _GGUF_UINT8: ("<B", 1),
    _GGUF_INT8: ("<b", 1),
    _GGUF_UINT16: ("<H", 2),
    _GGUF_INT16: ("<h", 2),
    _GGUF_UINT32: ("<I", 4),
    _GGUF_INT32: ("<i", 4),
    _GGUF_FLOAT32: ("<f", 4),
    _GGUF_BOOL: ("<B", 1),
    _GGUF_UINT64: ("<Q", 8),
    _GGUF_INT64: ("<q", 8),
    _GGUF_FLOAT64: ("<d", 8),
}


class InventoryError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Byte sources
# --------------------------------------------------------------------------- #
class RangeReader:
    """Random-access reader over HTTP range requests, with fetch accounting."""

    def __init__(self, url: str, size: int, token: Optional[str] = None, chunk: int = 1 << 20):
        self.url = url
        self.size = size
        self.token = token
        self.chunk = chunk
        self._cache: Dict[int, bytes] = {}
        self._pos = 0
        self.bytes_fetched = 0
        self.requests = 0

    @property
    def pos(self) -> int:
        return self._pos

    def _fetch(self, start: int) -> bytes:
        end = min(start + self.chunk, self.size) - 1
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": "bongo-gguf-inventory/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise InventoryError(f"range request {start}-{end} failed: {exc}") from exc
        self.requests += 1
        self.bytes_fetched += len(data)
        return data

    def read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            base = (self._pos // self.chunk) * self.chunk
            if base not in self._cache:
                self._cache[base] = self._fetch(base)
            data = self._cache[base]
            off = self._pos - base
            take = min(n - len(out), len(data) - off)
            if take <= 0:
                raise InventoryError(f"short read at offset {self._pos}")
            out += data[off : off + take]
            self._pos += take
        return bytes(out)


class LocalReader:
    """Random-access reader over a local file."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        self.size = os.fstat(self._fh.fileno()).st_size
        self.bytes_fetched = 0
        self.requests = 0
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    def read(self, n: int) -> bytes:
        data = self._fh.read(n)
        if len(data) != n:
            raise InventoryError(f"short read in {self.path}")
        self.bytes_fetched += len(data)
        self.requests += 1
        self._pos += n
        return data


# --------------------------------------------------------------------------- #
# GGUF parsing
# --------------------------------------------------------------------------- #
def _u(reader, fmt: str, size: int):
    return struct.unpack(fmt, reader.read(size))[0]


def _read_string(reader, limit: int = 1 << 30) -> str:
    n = _u(reader, "<Q", 8)
    if n > limit:
        raise InventoryError(f"implausible string length {n}")
    return reader.read(n).decode("utf-8", "replace")


def _read_value(reader, vtype: int):
    if vtype == _GGUF_STRING:
        return _read_string(reader)
    if vtype == _GGUF_ARRAY:
        elem_type = _u(reader, "<I", 4)
        count = _u(reader, "<Q", 8)
        return [_read_value(reader, elem_type) for _ in range(count)]
    if vtype in _FIXED:
        fmt, size = _FIXED[vtype]
        return _u(reader, fmt, size)
    raise InventoryError(f"unknown GGUF metadata value type {vtype}")


@dataclass
class Tensor:
    name: str
    dims: List[int]
    type_id: int
    offset: int  # relative to the start of this shard's tensor data section
    file: str
    n_bytes: Optional[int] = None
    type_name: Optional[str] = None
    type_inferred: bool = False

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class Shard:
    name: str
    size: int
    version: int = 0
    alignment: int = 32
    metadata: Dict[str, object] = field(default_factory=dict)
    tensors: List[Tensor] = field(default_factory=list)
    header_end: int = 0
    data_start: int = 0
    bytes_fetched: int = 0
    requests: int = 0
    split_no: Optional[int] = None
    split_count: Optional[int] = None
    split_tensors_count: Optional[int] = None


def parse_header(reader, name: str, size: int) -> Shard:
    magic = reader.read(4)
    if magic != b"GGUF":
        raise InventoryError(f"{name}: not a GGUF file (magic={magic!r})")
    shard = Shard(name=name, size=size)
    shard.version = _u(reader, "<I", 4)
    n_tensors = _u(reader, "<Q", 8)
    n_kv = _u(reader, "<Q", 8)

    for _ in range(n_kv):
        key = _read_string(reader)
        vtype = _u(reader, "<I", 4)
        shard.metadata[key] = _read_value(reader, vtype)

    for _ in range(n_tensors):
        tname = _read_string(reader)
        n_dims = _u(reader, "<I", 4)
        dims = [_u(reader, "<Q", 8) for _ in range(n_dims)]
        type_id = _u(reader, "<I", 4)
        offset = _u(reader, "<Q", 8)
        shard.tensors.append(Tensor(tname, dims, type_id, offset, name))

    align = int(shard.metadata.get("general.alignment", 32) or 32)
    shard.alignment = align
    shard.header_end = reader.pos
    shard.data_start = (shard.header_end + align - 1) // align * align
    shard.bytes_fetched = reader.bytes_fetched
    shard.requests = reader.requests
    shard.split_no = shard.metadata.get("split.no")
    shard.split_count = shard.metadata.get("split.count")
    shard.split_tensors_count = shard.metadata.get("split.tensors.count")
    return shard


def size_from_table(n_elements: int, type_id: int) -> Optional[int]:
    entry = GGML_TYPES.get(type_id)
    if entry is None:
        return None
    _, block, tsize = entry
    if n_elements % block:
        raise InventoryError(f"{n_elements} elements not divisible by block {block}")
    return (n_elements // block) * tsize


def resolve_sizes(shards: Sequence[Shard], inflate: bool = True) -> List[str]:
    """Fill in Tensor.n_bytes / type_name; infer unknown type parameters.

    Returns a list of human-readable notes about inferred/corrected entries.
    """
    notes: List[str] = []
    for shard in shards:
        for t in shard.tensors:
            entry = GGML_TYPES.get(t.type_id)
            if entry is not None:
                t.type_name = entry[0]
                t.n_bytes = size_from_table(t.n_elements, t.type_id)
            else:
                t.type_name = f"UNKNOWN_{t.type_id}"

    # Infer unknown types from the shard layout: with tensors ordered by offset,
    # the span up to the next tensor is (padding + real size), padding < alignment.
    for shard in shards:
        unknown = [t for t in shard.tensors if t.n_bytes is None]
        if not unknown:
            continue
        if not inflate:
            for t in unknown:
                t.n_bytes = 0
            notes.append(f"{shard.name}: type ids {sorted({t.type_id for t in unknown})} unknown, sizes unset")
            continue
        ordered = sorted(shard.tensors, key=lambda t: t.offset)
        spans = {}
        for cur, nxt in zip(ordered, ordered[1:]):
            spans[id(cur)] = nxt.offset - cur.offset
        # last tensor: file end bounds it
        last = ordered[-1]
        spans[id(last)] = shard.size - shard.data_start - last.offset
        for t in unknown:
            span = spans.get(id(t))
            if span is None or span <= 0:
                raise InventoryError(f"{shard.name}:{t.name} unknown type, no usable span")
            solved = None
            for block in (256, 32, 16, 8, 4, 2, 1):
                if t.n_elements % block:
                    continue
                nblocks = t.n_elements // block
                approx = span / nblocks
                tsize = round(approx)
                if tsize <= 0:
                    continue
                stored = tsize * nblocks
                if 0 <= span - stored < shard.alignment:
                    solved = (block, tsize)
                    break
            if solved is None:
                raise InventoryError(f"{shard.name}:{t.name} cannot infer type {t.type_id} from span {span}")
            block, tsize = solved
            GGML_TYPES[t.type_id] = (f"UNKNOWN_{t.type_id}", block, tsize)
            t.type_name = f"UNKNOWN_{t.type_id}"
            t.n_bytes = size_from_table(t.n_elements, t.type_id)
            t.type_inferred = True
            notes.append(
                f"{shard.name}: inferred type {t.type_id} = block {block}/size {tsize} from {t.name}"
            )
    return notes


def verify_layout(shards: Sequence[Shard]) -> List[dict]:
    """Check that computed tensor sizes tile each shard's data section exactly."""
    problems: List[dict] = []
    for shard in shards:
        ordered = sorted(shard.tensors, key=lambda t: t.offset)
        cursor = shard.data_start
        for t in ordered:
            abs_off = shard.data_start + t.offset
            if abs_off < cursor:
                problems.append(
                    {
                        "shard": shard.name,
                        "tensor": t.name,
                        "issue": "overlap",
                        "expected_min_offset": cursor - shard.data_start,
                        "offset": t.offset,
                    }
                )
            pad = abs_off - cursor
            if pad < 0 or pad >= shard.alignment:
                problems.append(
                    {
                        "shard": shard.name,
                        "tensor": t.name,
                        "issue": "misalignment" if pad >= shard.alignment else "overlap",
                        "padding": pad,
                    }
                )
            cursor = abs_off + (t.n_bytes or 0)
        tail = shard.size - cursor
        if not (0 <= tail < shard.alignment):
            problems.append(
                {
                    "shard": shard.name,
                    "tensor": "<end>",
                    "issue": "unaccounted tail",
                    "tail_bytes": tail,
                }
            )
    return problems


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #
_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)


def hf_file_list(repo: str, token: Optional[str] = None) -> Dict[str, int]:
    """Return {filename: size_bytes} for a HF repo (one small JSON request)."""
    url = HF_API.format(repo=repo) + "?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "bongo-gguf-inventory/1"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    return {s["rfilename"]: s.get("size", 0) for s in payload.get("siblings", [])}


def split_siblings(filename: str) -> List[str]:
    m = _SPLIT_RE.match(filename)
    if not m:
        return [filename]
    count = int(m.group("count"))
    return [f"{m.group('stem')}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


@dataclass
class Source:
    repo: Optional[str]
    files: List[str]
    local: Optional[str] = None
    sizes: Dict[str, int] = field(default_factory=dict)


def resolve_source(args) -> Source:
    token = None
    if args.hf_token_env:
        token = os.environ.get(args.hf_token_env)
        if not token:
            raise InventoryError(f"environment variable {args.hf_token_env} is empty")
    if args.local:
        if not os.path.exists(args.local):
            raise InventoryError(f"no such file: {args.local}")
        return Source(repo=None, files=[os.path.basename(args.local)], local=args.local)
    if args.repo and args.file:
        listing = hf_file_list(args.repo, token)
        files = [f for f in split_siblings(args.file) if f in listing]
        if not files:
            raise InventoryError(f"{args.file} not found in {args.repo}")
        return Source(repo=args.repo, files=files, sizes={f: listing[f] for f in files})
    raise InventoryError("give --local FILE, or --repo REPO --file FILE")


def read_source(args, source: Source, token: Optional[str]) -> List[Shard]:
    shards: List[Shard] = []
    if source.local:
        reader = LocalReader(source.local)
        shards.append(parse_header(reader, source.files[0], reader.size))
        return shards
    for name in source.files:
        url = HF_RESOLVE.format(repo=source.repo, file=name)
        size = source.sizes[name]
        reader = RangeReader(url, size, token=token, chunk=args.chunk)
        shards.append(parse_header(reader, name, size))
    return shards


# --------------------------------------------------------------------------- #
# Classification (what the placement plan cares about)
# --------------------------------------------------------------------------- #
EXPERT_SUFFIXES = (".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight")
SHEXP_SUFFIXES = (".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight")
NGRAM_NAMES = ("per_layer_token_embd.weight",)


def classify(name: str) -> str:
    if any(name.endswith(s) for s in EXPERT_SUFFIXES):
        return "experts"
    if any(name.endswith(s) for s in SHEXP_SUFFIXES):
        return "shared_experts"
    if name in NGRAM_NAMES:
        return "ngram_table"
    if name in ("token_embd.weight", "output.weight"):
        return "embed_head"
    return "other"


def tensor_class(name: str) -> str:
    """Per-tensor-class key: layer index stripped so 48 layers collapse to one row."""
    if name.startswith("blk."):
        parts = name.split(".", 2)
        if len(parts) == 3:
            return parts[2]
    return name


def build_inventory(shards: Sequence[Shard], source: Source) -> dict:
    tensors = [t for s in shards for t in s.tensors]
    buckets: Dict[str, dict] = {}
    for t in tensors:
        bucket = classify(t.name)
        b = buckets.setdefault(bucket, {"count": 0, "bytes": 0, "types": {}})
        b["count"] += 1
        b["bytes"] += t.n_bytes or 0
        b["types"][t.type_name] = b["types"].get(t.type_name, 0) + 1

    classes: Dict[str, dict] = {}
    for t in tensors:
        key = tensor_class(t.name)
        c = classes.setdefault(key, {"count": 0, "bytes": 0, "types": {}, "bucket": classify(t.name)})
        c["count"] += 1
        c["bytes"] += t.n_bytes or 0
        c["types"][t.type_name] = c["types"].get(t.type_name, 0) + 1

    per_shard = []
    for s in shards:
        data_bytes = sum(t.n_bytes or 0 for t in s.tensors)
        per_shard.append(
            {
                "file": s.name,
                "bytes": s.size,
                "header_bytes": s.header_end,
                "data_section_offset": s.data_start,
                "data_section_bytes": s.size - s.data_start,
                "tensor_data_bytes": data_bytes,
                "padding_bytes": (s.size - s.data_start) - data_bytes,
                "tensor_count": len(s.tensors),
                "split_no": s.split_no,
                "split_count": s.split_count,
                "split_tensors_count": s.split_tensors_count,
                "bytes_fetched": s.bytes_fetched,
                "range_requests": s.requests,
                "tensor_byte_bytes_by_bucket": _bucket_totals(s.tensors),
            }
        )

    return {
        "source": {
            "repo": source.repo,
            "files": source.files,
            "local": source.local,
            "requested": source.files[0],
        },
        "gguf_version": shards[0].version if shards else None,
        "alignment": shards[0].alignment if shards else None,
        "metadata": _safe_metadata(shards[0].metadata) if shards else {},
        "bytes_fetched": sum(s.bytes_fetched for s in shards),
        "range_requests": sum(s.requests for s in shards),
        "shards": per_shard,
        "tensor_count": len(tensors),
        "total_tensor_bytes": sum(t.n_bytes or 0 for t in tensors),
        "buckets": buckets,
        "classes": dict(sorted(classes.items())),
        "tensors": [
            {
                "name": t.name,
                "shard": t.file,
                "type": t.type_name,
                "type_id": t.type_id,
                "dims": t.dims,
                "elements": t.n_elements,
                "bytes": t.n_bytes,
                "offset": t.offset,
                "bucket": classify(t.name),
            }
            for t in sorted(tensors, key=lambda t: (t.file, t.offset))
        ],
    }


def _bucket_totals(tensors: Iterable[Tensor]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in tensors:
        out[classify(t.name)] = out.get(classify(t.name), 0) + (t.n_bytes or 0)
    return out


def _safe_metadata(md: Dict[str, object]) -> Dict[str, object]:
    """Keep the small scalar/interesting metadata, drop the giant tokenizer arrays."""
    keep_exact = {
        "general.architecture",
        "general.name",
        "general.size_label",
        "general.file_type",
        "general.quantization_version",
        "split.no",
        "split.count",
        "split.tensors.count",
    }
    out: Dict[str, object] = {}
    for k, v in md.items():
        if k in keep_exact or k.startswith("qwen4exp.ple"):
            if isinstance(v, list) and len(v) > 16:
                out[k] = f"<array len={len(v)}>"
            else:
                out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Cross-checks
# --------------------------------------------------------------------------- #
def cross_check_manifest(inv: dict, manifest_path: str) -> List[dict]:
    manifest = json.load(open(manifest_path))
    results: List[dict] = []
    by_name = {f["file"]: f for f in manifest.get("files", [])}
    tier = None
    for name, sizes in by_name.items():
        if inv["source"]["requested"] == name:
            tier = sizes.get("tier")
    for shard in inv["shards"]:
        entry = by_name.get(shard["file"])
        if entry is None:
            results.append({"file": shard["file"], "check": "in manifest", "ok": False})
            continue
        results.append(
            {
                "file": shard["file"],
                "check": "byte size matches release-manifest.json",
                "expected": entry["bytes"],
                "actual": shard["bytes"],
                "ok": entry["bytes"] == shard["bytes"],
            }
        )
    if tier:
        model = next((m for m in manifest.get("models", []) if m["tier"] == tier), None)
        if model:
            shard_total = sum(s["bytes"] for s in inv["shards"])
            expected_shards = len(model["shards"]) if model.get("shards") else 1
            all_shards = len(inv["shards"]) == expected_shards
            delta = shard_total - model["unsplit_bytes"]
            # The unsplit file carries ONE header; each shard carries its own, so the
            # shard sum is slightly larger. Anything past a few MiB means the wrong file
            # set was parsed, so bound this check instead of demanding equality.
            results.append(
                {
                    "check": f"{tier}: sum of shards ~= unsplit_bytes (header duplication)",
                    "expected": model["unsplit_bytes"],
                    "actual": shard_total,
                    "delta_bytes": delta,
                    "ok": all_shards and abs(delta) <= (1 << 21),
                    "note": "unsplit file has one header, the shards have one each",
                }
            )
            results.append(
                {
                    "check": f"{tier}: tensor count == manifest tensor_count",
                    "expected": model["tensor_count"],
                    "actual": inv["tensor_count"],
                    "ok": model["tensor_count"] == inv["tensor_count"],
                }
            )
    return results


def cross_check_alloc(inv: dict, alloc_path: str) -> List[dict]:
    """Compare the parsed per-tensor types with tensor-allocation/*.rco-allocation.txt."""
    expected: Dict[str, str] = {}
    for line in open(alloc_path):
        if "=" not in line or line.startswith("#"):
            continue
        name, typ = line.strip().split("=", 1)
        expected[name] = typ
    results: List[dict] = []
    mismatches = 0
    missing = 0
    for t in inv["tensors"]:
        want = expected.get(t["name"])
        if want is None:
            missing += 1
            continue
        if want != t["type"]:
            mismatches += 1
            if mismatches <= 10:
                results.append(
                    {"tensor": t["name"], "check": "type", "expected": want, "actual": t["type"], "ok": False}
                )
    results.append(
        {
            "check": "per-tensor type == rco-allocation.txt",
            "compared": len(inv["tensors"]) - missing,
            "mismatches": mismatches,
            "not_in_alloc_file": missing,
            "ok": mismatches == 0 and missing == 0,
        }
    )
    return results


def cross_check_capsule(inv: dict, capsule_path: str) -> List[dict]:
    """Compare parsed tensor -> shard/offset/size against an exact-source capsule.

    The capsule stores ``shard_offset`` as the absolute byte offset inside the shard
    file (gguf-py's ``data_offset`` includes the header), so we add this shard's
    data-section start before comparing.
    """
    capsule = json.load(open(capsule_path))
    data_start = {s["file"]: s["data_section_offset"] for s in inv["shards"]}
    seg_tensors = {s["tensor"]: s for s in capsule["segments"] if "tensor" in s}
    mismatches: List[dict] = []
    compared = 0
    for t in inv["tensors"]:
        seg = seg_tensors.get(t["name"])
        if seg is None:
            continue
        compared += 1
        if seg.get("bytes") != t["bytes"]:
            mismatches.append(
                {"tensor": t["name"], "field": "bytes", "expected": seg["bytes"], "actual": t["bytes"]}
            )
        if seg.get("shard") != t["shard"]:
            mismatches.append(
                {"tensor": t["name"], "field": "shard", "expected": seg["shard"], "actual": t["shard"]}
            )
        if seg.get("shard_offset") != (data_start.get(t["shard"], 0) + t["offset"]):
            mismatches.append(
                {
                    "tensor": t["name"],
                    "field": "offset",
                    "expected": seg["shard_offset"],
                    "actual": data_start.get(t["shard"], 0) + t["offset"],
                }
            )
    return [
        {
            "check": "tensor -> shard/offset/bytes == exact-source capsule",
            "compared": compared,
            "mismatches": len(mismatches),
            "ok": not mismatches and compared == inv["tensor_count"],
            "examples": mismatches[:10],
        }
    ]


# --------------------------------------------------------------------------- #
# Self-test: build a synthetic GGUF and re-read it through the local path
# --------------------------------------------------------------------------- #
def _write_gguf(path: str, metadata: List[Tuple[str, int, object]], tensors: List[Tuple[str, List[int], int, int]],
                alignment: int = 32) -> None:
    """Write a minimal GGUF v3 file. tensors = (name, dims, type_id, payload_bytes)."""
    out = bytearray()

    def u32(v: int) -> None:
        out.extend(struct.pack("<I", v))

    def u64(v: int) -> None:
        out.extend(struct.pack("<Q", v))

    def s(text: str) -> None:
        raw = text.encode()
        u64(len(raw))
        out.extend(raw)

    out += b"GGUF"
    u32(3)
    u64(len(tensors))
    u64(len(metadata))
    for key, vtype, value in metadata:
        s(key)
        u32(vtype)
        if vtype == _GGUF_STRING:
            s(str(value))
        elif vtype == _GGUF_UINT32:
            u32(int(value))
        else:
            raise ValueError(f"self-test writer: unsupported metadata type {vtype}")
    offsets = []
    cursor = 0
    for name, dims, type_id, payload in tensors:
        s(name)
        u32(len(dims))
        for d in dims:
            u64(d)
        u32(type_id)
        u64(cursor)
        offsets.append((cursor, payload))
        cursor = (cursor + payload + alignment - 1) // alignment * alignment
    header_end = len(out)
    data_start = (header_end + alignment - 1) // alignment * alignment
    out.extend(b"\0" * (data_start - header_end))
    for _off, payload in offsets:
        out.extend(b"\x5a" * payload)
        out.extend(b"\0" * (-len(out) % alignment))
    with open(path, "wb") as fh:
        fh.write(out)


def self_test(verbose: bool = True) -> int:
    """Exercise the local path, the type table, sizing, buckets and layout check."""
    import tempfile

    cases = [
        ("token_embd.weight", [32, 4], 8, 136),          # Q8_0: 128 el / 32 * 34
        ("per_layer_token_embd.weight", [16, 64], 20, 576),  # IQ4_NL: 1024 / 32 * 18
        ("blk.0.ffn_gate_exps.weight", [256, 2, 4], 22, 656),  # IQ2_S: 2048 / 256 * 82
        ("blk.0.ffn_down_exps.weight", [256, 2, 4], 42, 576),  # Q2_0: 2048 / 256 * 72
        ("blk.0.ffn_down_shexp.weight", [64, 8], 0, 2048),      # F32: 512 * 4
        ("blk.1.ffn_gate_exps.weight", [256, 2, 4], 99, 512),   # unknown -> inferred
    ]
    meta = [
        ("general.architecture", _GGUF_STRING, "qwen4exp"),
        ("general.alignment", _GGUF_UINT32, 32),
        ("split.count", _GGUF_UINT32, 1),
    ]
    failures: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "selftest.gguf")
        _write_gguf(path, meta, cases)
        args = argparse.Namespace(
            repo=None, local=path, file=None, json=None, list_tensors=False,
            cross_check_manifest=None, cross_check_alloc=None, cross_check_capsule=None,
            hf_token_env=None, chunk=1 << 20, summary=False, markdown=False,
            no_infer=False,
        )
        source = resolve_source(args)
        shards = read_source(args, source, None)
        notes = resolve_sizes(shards, inflate=True)
        problems = verify_layout(shards)
        inv = build_inventory(shards, source)

    by_name = {t["name"]: t for t in inv["tensors"]}
    expect_bytes = {name: payload for name, _d, _t, payload in cases}
    expect_type = {"token_embd.weight": "Q8_0", "per_layer_token_embd.weight": "IQ4_NL",
                   "blk.0.ffn_gate_exps.weight": "IQ2_S", "blk.0.ffn_down_exps.weight": "Q2_0",
                   "blk.0.ffn_down_shexp.weight": "F32"}
    for name, want in expect_bytes.items():
        got = by_name.get(name, {}).get("bytes")
        if got != want:
            failures.append(f"{name}: bytes {got} != {want}")
    for name, want in expect_type.items():
        got = by_name.get(name, {}).get("type")
        if got != want:
            failures.append(f"{name}: type {got} != {want}")
    if not by_name.get("blk.1.ffn_gate_exps.weight", {}).get("type_id") == 99:
        failures.append("unknown-type tensor not present")
    want_buckets = {"experts": 656 + 576 + 512, "ngram_table": 576, "embed_head": 136, "shared_experts": 2048}
    for bucket, want in want_buckets.items():
        got = inv["buckets"].get(bucket, {}).get("bytes")
        if got != want:
            failures.append(f"bucket {bucket}: {got} != {want}")
    if inv["tensor_count"] != len(cases):
        failures.append(f"tensor_count {inv['tensor_count']} != {len(cases)}")
    if problems:
        failures.append(f"layout problems: {problems}")
    if not any("inferred type 99" in n for n in notes):
        failures.append("unknown type 99 was not inferred")
    # Range/header read accounting: we must have fetched at least the header, and far
    # less than the file (the tensor payload is never downloaded).
    for shard in inv["shards"]:
        if not (
            shard["header_bytes"] <= shard["bytes_fetched"] <= shard["bytes"]
        ):
            failures.append(
                f"{shard['file']}: fetched {shard['bytes_fetched']} outside "
                f"[{shard['header_bytes']}, {shard['bytes']}]"
            )
    if inv["total_tensor_bytes"] <= inv["bytes_fetched"]:
        failures.append("self-test payload is not larger than what was read")

    if verbose:
        if failures:
            print("SELF-TEST FAILED")
            for f in failures:
                print("  -", f)
        else:
            print("SELF-TEST OK: local GGUF parse, type table, buckets, layout check and unknown-type")
            print(f"              inference all behaved as expected ({len(cases)} synthetic tensors)")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def gib(n: int) -> str:
    return f"{n / (1024 ** 3):.2f}"


def gb(n: int) -> str:
    return f"{n / 1e9:.2f}"


def print_summary(inv: dict, args) -> None:
    print(f"source          : {inv['source']['repo'] or inv['source']['local']}")
    for f in inv["source"]["files"]:
        print(f"  file          : {f}")
    print(f"gguf version   : {inv['gguf_version']}   alignment: {inv['alignment']} B")
    print(f"tensor count   : {inv['tensor_count']}")
    print(f"tensor bytes   : {inv['total_tensor_bytes']} ({gib(inv['total_tensor_bytes'])} GiB)")
    print(f"bytes fetched  : {inv['bytes_fetched']} ({inv['bytes_fetched'] / 1e6:.2f} MB)")
    print(f"read ops       : {inv['range_requests']} (local reads, or HTTP range requests for a repo)")
    print()
    print("shards:")
    hdr = f"  {'file':<62} {'file GiB':>9} {'tensors':>8} {'data GiB':>9} {'fetched MB':>10}"
    print(hdr)
    for s in inv["shards"]:
        print(
            f"  {s['file']:<62} {gib(s['bytes']):>9} {s['tensor_count']:>8} "
            f"{gib(s['tensor_data_bytes']):>9} {s['bytes_fetched'] / 1e6:>10.2f}"
        )
    print()
    print("buckets:")
    print(f"  {'bucket':<18} {'tensors':>8} {'bytes':>14} {'GiB':>8}  types")
    for name, b in sorted(inv["buckets"].items(), key=lambda kv: -kv[1]["bytes"]):
        types = ", ".join(f"{k}x{v}" for k, v in sorted(b["types"].items()))
        print(f"  {name:<18} {b['count']:>8} {b['bytes']:>14} {gib(b['bytes']):>8}  {types}")
    print()
    if args.list_tensors:
        print("tensors:")
        print(f"  {'name':<44} {'shard':<6} {'type':<10} {'elements':>13} {'bytes':>14} {'GiB':>7}")
        for t in inv["tensors"]:
            short = t["shard"].split("-")[-3] if "-" in t["shard"] else t["shard"]
            print(
                f"  {t['name']:<44} {short:<6} {t['type']:<10} {t['elements']:>13} "
                f"{t['bytes']:>14} {gib(t['bytes']):>7}"
            )
    else:
        print("per-class totals (top 30 by bytes):")
        print(f"  {'class':<40} {'tensors':>7} {'GiB':>8}  types")
        rows = sorted(inv["classes"].items(), key=lambda kv: -kv[1]["bytes"])[:30]
        for name, c in rows:
            types = ", ".join(f"{k}x{v}" for k, v in sorted(c["types"].items()))
            print(f"  {name:<40} {c['count']:>7} {gib(c['bytes']):>8}  {types}")


def print_markdown(inv: dict) -> None:
    """Reproducible markdown tables; the doc is assembled from this output."""
    print(f"<!-- generated by tools/gguf-inventory.py from {inv['source']['requested']} -->")
    print(f"<!-- bytes fetched: {inv['bytes_fetched']} in {inv['range_requests']} read ops -->")
    print()
    print("### Shards\n")
    print("| File | Bytes | GiB | Tensors | Tensor bytes | GiB | Header bytes | Fetch MB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in inv["shards"]:
        print(
            f"| `{s['file']}` | {s['bytes']} | {gib(s['bytes'])} | {s['tensor_count']} | "
            f"{s['tensor_data_bytes']} | {gib(s['tensor_data_bytes'])} | {s['header_bytes']} | "
            f"{s['bytes_fetched'] / 1e6:.2f} |"
        )
    print()
    print("### Buckets\n")
    print("| Bucket | Tensors | Bytes | GiB | Types |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, b in sorted(inv["buckets"].items(), key=lambda kv: -kv[1]["bytes"]):
        types = ", ".join(f"{k} x{v}" for k, v in sorted(b["types"].items()))
        print(f"| `{name}` | {b['count']} | {b['bytes']} | {gib(b['bytes'])} | {types} |")
    print()
    print("### Per-tensor class\n")
    print("| Class | Tensors | Bytes | GiB | Types |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, c in sorted(inv["classes"].items(), key=lambda kv: -kv[1]["bytes"]):
        types = ", ".join(f"{k} x{v}" for k, v in sorted(c["types"].items()))
        print(f"| `{name}` | {c['count']} | {c['bytes']} | {gib(c['bytes'])} | {types} |")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--repo", help="Hugging Face repo id, e.g. ukisai/Swift-...-GGUF")
    src.add_argument("--local", help="path to a local GGUF file")
    ap.add_argument("--file", help="file inside --repo (first shard is enough)")
    ap.add_argument("--json", help="write the full inventory as JSON to this path")
    ap.add_argument("--list-tensors", action="store_true", help="print every tensor instead of class totals")
    ap.add_argument("--cross-check-manifest", metavar="PATH", help="release-manifest.json to check against")
    ap.add_argument("--cross-check-alloc", metavar="PATH", help="rco-allocation.txt to check types against")
    ap.add_argument("--cross-check-capsule", metavar="PATH", help="exact-source-recovery capsule to check against")
    ap.add_argument("--hf-token-env", default="HF_TOKEN", help="env var holding a HF token (default: HF_TOKEN)")
    ap.add_argument("--chunk", type=int, default=1 << 20, help="range-request window size in bytes")
    ap.add_argument("--summary", action="store_true", help="print the human summary (default when no --json)")
    ap.add_argument("--markdown", action="store_true", help="print reproducible markdown tables")
    ap.add_argument("--self-test", action="store_true", help="run the offline self-test and exit")
    ap.add_argument("--no-infer", action="store_true", help="fail on unknown ggml types instead of inferring")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not (args.repo or args.local):
        ap.error("give --repo REPO --file FILE, --local FILE, or --self-test")

    # shorthand form: --repo hf:owner/repo/file.gguf (file then implied)
    if args.repo and args.repo.startswith("hf:"):
        parts = args.repo[3:].split("/", 2)
        if len(parts) != 3:
            ap.error("hf: shorthand must be hf:owner/repo/file.gguf")
        if args.file:
            ap.error("give either --repo hf:owner/repo/file.gguf or --repo owner/repo --file file.gguf")
        args.repo, args.file = "/".join(parts[:2]), parts[2]

    token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    source = resolve_source(args)
    shards = read_source(args, source, token)
    notes = resolve_sizes(shards, inflate=not args.no_infer)
    problems = verify_layout(shards)
    inv = build_inventory(shards, source)
    inv["layout_problems"] = problems
    inv["notes"] = notes

    if args.cross_check_manifest:
        inv["cross_checks_manifest"] = cross_check_manifest(inv, args.cross_check_manifest)
    if args.cross_check_alloc:
        inv["cross_checks_alloc"] = cross_check_alloc(inv, args.cross_check_alloc)
    if args.cross_check_capsule:
        inv["cross_checks_capsule"] = cross_check_capsule(inv, args.cross_check_capsule)

    if args.markdown:
        print_markdown(inv)

    if args.summary or (not args.json and not args.markdown):
        print_summary(inv, args)
        for n in notes:
            print(f"note: {n}")
        if problems:
            print(f"\nLAYOUT PROBLEMS ({len(problems)}):")
            for p in problems[:20]:
                print("  ", p)
        else:
            print("\nlayout check   : OK (computed sizes tile every shard exactly, padding < alignment)")

    for key in ("cross_checks_manifest", "cross_checks_alloc", "cross_checks_capsule"):
        if key in inv:
            print(f"\n{key}:")
            for row in inv[key]:
                print("  ", json.dumps(row))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(inv, fh, indent=1)
            fh.write("\n")
        print(f"\nwrote {args.json}")
    return 0 if not problems else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InventoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
