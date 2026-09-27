#!/usr/bin/env python3
"""bongo benchmark harness.

Measures a running OpenAI-compatible llama-server endpoint (the one that
``bongo.sh`` starts) and writes ``matrix.json`` + ``matrix.md`` into a results
directory.

Per tier and per context length it records:

* prompt-processing throughput (prefill tok/s)
* output/decode throughput (tok/s)
* time-to-first-token (TTFT, ms)
* peak VRAM and system RAM
* a 128K "needle" retrieval check (proves the context is real)

``bench/run.sh`` is the single documented command; nothing needs manual edits
between runs.  Stdlib Python 3 only.

Exit codes: 0 = full success, 3 = partial/negative result recorded, 2 = fatal
(endpoint unreachable).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import (  # noqa: E402
    CORPUS,
    HARNESS_VERSION,
    NEEDLE,
    SCHEMA,
    HttpResult,
    MemorySampler,
    Tokenizer,
    collect_machine_spec,
    get_json,
    human_bytes,
    num,
    now_iso,
    post_json,
    server_root,
    post_raw,
    read_text,
    run_cmd,
    stream_completion,
)


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------


def detect_bongo_config(repo_root):
    candidates = [
        os.environ.get("BONGO_CONFIG"),
        str(Path(repo_root) / "bongo.local.json"),
        str(Path(repo_root) / "build" / "bongo-config.json"),
        str(Path(repo_root) / "bongo-config.json"),
        str(Path.home() / ".config" / "bongo" / "config.json"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            txt = read_text(path)
            try:
                return {"path": path, "content": json.loads(txt) if txt else None}
            except Exception:  # noqa: BLE001
                return {"path": path, "content": txt}
    return None


def detect_gguf_dir(repo_root, explicit, bongo_config=None):
    if explicit:
        return explicit
    env = os.environ.get("BONGO_GGUF_DIR")
    if env and os.path.isdir(env):
        return env
    if bongo_config and isinstance(bongo_config.get("content"), dict):
        cfg = bongo_config["content"]
        for key in ("gguf_dir", "model_dir", "gguf_path", "model_path"):
            val = cfg.get(key)
            if val:
                cand = val if os.path.isdir(val) else os.path.dirname(val)
                if cand and os.path.isdir(cand):
                    return cand
    for cand in (
        Path(repo_root) / "models",
        Path.home() / ".cache" / "bongo" / "models",
    ):
        if cand.is_dir():
            return str(cand)
    return None


def hash_file(path, mode="full"):
    import hashlib

    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if mode == "sampled":
            h.update(fh.read(8 * 1024 * 1024))
            if size > 16 * 1024 * 1024:
                fh.seek(-8 * 1024 * 1024, os.SEEK_END)
                h.update(fh.read())
        else:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(block)
    return {"sha256": h.hexdigest(), "hash_mode": mode, "size_bytes": size}


def model_shards(gguf_dir, hash_mode, cache_path):
    if not gguf_dir or not os.path.isdir(gguf_dir):
        return None
    import glob

    cache = {}
    if cache_path and os.path.isfile(str(cache_path)):
        try:
            cache = json.loads(read_text(str(cache_path)) or "{}")
        except Exception:  # noqa: BLE001
            cache = {}
    shards = []
    for path in sorted(glob.glob(os.path.join(gguf_dir, "**", "*.gguf"), recursive=True)):
        rel = os.path.relpath(path, gguf_dir)
        st = os.stat(path)
        key = f"{rel}:{st.st_size}:{int(st.st_mtime)}:{hash_mode}"
        if key in cache:
            rec = cache[key]
        elif hash_mode == "none":
            rec = {"sha256": None, "hash_mode": "none", "size_bytes": st.st_size}
        else:
            rec = hash_file(path, hash_mode)
        shards.append({"path": rel, **rec})
        cache[key] = rec
    if cache_path:
        try:
            Path(str(cache_path)).write_text(json.dumps(cache, indent=2))
        except Exception:  # noqa: BLE001
            pass
    return shards


def server_flags(pid):
    if not pid:
        return None
    cmd = read_text(f"/proc/{pid}/cmdline")
    if not cmd:
        return None
    return cmd.split("\x00")


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def summarize(runs, key):
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    med = statistics.median(vals)
    out = {
        "n": len(vals),
        "median": round(med, 3),
        "mean": round(statistics.fmean(vals), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
    }
    if len(vals) > 1:
        sd = statistics.stdev(vals)
        out["stdev"] = round(sd, 3)
        out["cv"] = round(sd / med, 4) if med else None
    return out


def timings_of(res):
    if isinstance(res.json, dict) and isinstance(res.json.get("timings"), dict):
        return res.json["timings"]
    return {}


def completion_call(base_url, model, prompt, max_tokens, timeout, extra=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
    }
    if extra:
        payload.update(extra)
    res = post_json(base_url, "/completions", payload, timeout)
    t = timings_of(res)
    text = ""
    usage = res.json.get("usage") if isinstance(res.json, dict) else None
    if isinstance(res.json, dict):
        choices = res.json.get("choices") or []
        if choices and isinstance(choices[0], dict):
            text = choices[0].get("text") or ""
    out = {
        "status": res.status,
        "wall_ms": round(res.elapsed_ms, 2),
        "prompt_tokens": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": t.get("prompt_per_second"),
        "output_tokens": t.get("predicted_n"),
        "output_ms": t.get("predicted_ms"),
        "output_tps": t.get("predicted_per_second"),
        "usage": usage,
        "text": text[:200],
    }
    if res.status != 200:
        out["error"] = res.body.decode("utf-8", "replace")[:2000]
    return out


def run_error_cases(base_url, model, context_limit, args):
    """Send malformed / boundary requests and record how the server handled them.

    The harness must survive all of these; the server's status + message is the
    result.  This covers acceptance criterion 4 ("malformed-prompt / error cases
    handled").
    """
    cases = []
    long_prompt = "token " * (context_limit + 2048)

    def record(name, kind, res, note=""):
        body = res.body.decode("utf-8", "replace")[:1200] if isinstance(res.body, bytes) else str(res.body)
        cases.append(
            {
                "name": name,
                "kind": kind,
                "http_status": res.status,
                "handled": res.status is not None,
                "is_error_status": bool(res.status is None or res.status >= 400),
                "response_excerpt": body[:1200],
                "note": note,
            }
        )

    record(
        "empty_prompt",
        "json",
        post_json(base_url, "/completions", {"model": model, "prompt": "", "max_tokens": 4}, args.timeout),
        "empty string prompt",
    )
    record(
        "missing_prompt_field",
        "json",
        post_json(base_url, "/completions", {"model": model, "max_tokens": 4}, args.timeout),
        "no prompt key",
    )
    record(
        "unknown_model",
        "json",
        post_json(base_url, "/completions", {"model": "definitely-not-a-model", "prompt": "hi", "max_tokens": 4}, args.timeout),
        "unknown model id",
    )
    record(
        "negative_max_tokens",
        "json",
        post_json(base_url, "/completions", {"model": model, "prompt": "hi", "max_tokens": -1}, args.timeout),
        "negative max_tokens",
    )
    record(
        "prompt_exceeds_context",
        "json",
        post_json(base_url, "/completions", {"model": model, "prompt": long_prompt, "max_tokens": 4}, args.timeout),
        f"prompt ~{context_limit + 2048} whitespace tokens > n_ctx",
    )
    record(
        "non_json_body",
        "raw",
        post_raw(base_url, "/completions", b"this is not json", "application/json", args.timeout),
        "invalid JSON body",
    )
    return cases


# ---------------------------------------------------------------------------
# top-level benchmark
# ---------------------------------------------------------------------------


def fatal_matrix(base_url, args, machine, status, body):
    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "harness": {"version": HARNESS_VERSION, "argv": sys.argv, "command": args.command},
        "endpoint": {"base_url": base_url},
        "tier": args.tier,
        "fatal_error": {
            "stage": "GET /v1/models",
            "status": status,
            "body": body[:2000],
            "message": "endpoint unreachable or not an OpenAI-compatible server",
        },
        "machine": machine,
        "results": [],
        "needle": {"status": "skipped", "reason": "endpoint unreachable"},
        "error_cases": [],
        "highest_working_context": None,
        "verdict": {"ok": False, "reason": "endpoint unreachable"},
    }


def benchmark(args):
    base_url = args.base_url.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    machine = collect_machine_spec()
    bongo_config = detect_bongo_config(args.repo_root)
    gguf_dir = detect_gguf_dir(args.repo_root, args.gguf_dir, bongo_config)

    # ---- preflight -------------------------------------------------------
    models_res = get_json(base_url, "/models", args.timeout)
    if models_res.status != 200:
        matrix = fatal_matrix(
            base_url, args, machine, models_res.status, models_res.body.decode("utf-8", "replace")
        )
        (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2))
        (out_dir / "matrix.md").write_text(render_markdown(matrix))
        print(f"FATAL: {base_url}/models returned {models_res.status}", file=sys.stderr)
        return 2

    model_id = args.model
    if not model_id:
        try:
            model_id = models_res.json["data"][0]["id"]
        except Exception:  # noqa: BLE001
            model_id = "default"

    props_res = get_json(server_root(base_url), "/props", args.timeout)
    props = props_res.json if isinstance(props_res.json, dict) else None
    context_limit = None
    llama_build = None
    if isinstance(props, dict):
        dgs = props.get("default_generation_settings") or {}
        context_limit = dgs.get("n_ctx") or props.get("n_ctx")
        bi = props.get("build_info") or props.get("build")
        llama_build = bi if isinstance(bi, str) else None
    if not context_limit:
        context_limit = max(args.contexts) + 1024
    if not gguf_dir and isinstance(props, dict) and props.get("model_path"):
        gguf_dir = os.path.dirname(props["model_path"]) or None

    explicit_pids = args.server_pid or []
    discovered = MemorySampler(server_pids=explicit_pids).server_pids()
    if not discovered and explicit_pids:
        discovered = explicit_pids
    flags = server_flags(discovered[0]) if discovered else None

    sampler = MemorySampler(server_pids=discovered, interval=args.memory_interval)
    sampler.start()
    tokenizer = Tokenizer(base_url, timeout=args.timeout)

    warmup = completion_call(base_url, model_id, "Hello.", 8, args.timeout)

    results = []
    failures = []
    highest_working = None

    contexts = args.contexts
    for idx, ctx in enumerate(contexts):
        entry = {"target_context": ctx, "status": "ok", "runs": [], "summary": {}}
        # reserve room for generated tokens when the target is the whole context
        target = ctx if ctx < context_limit else max(1, context_limit - max(args.max_tokens, 64))
        entry["planned_prompt_tokens"] = target
        prompt = tokenizer.size_to(target, CORPUS)
        entry["builder_uses_server_tokenizer"] = bool(tokenizer.available)
        t_start = time.time()
        for rep in range(args.repeats):
            run_start = time.time()
            rec = completion_call(base_url, model_id, prompt, args.max_tokens, args.timeout)
            run_end = time.time()
            rec["repeat"] = rep + 1
            rec["memory"] = sampler.window_peak(run_start, run_end)

            s_payload = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": args.ttft_tokens,
                "temperature": 0.0,
                "stream": True,
                "cache_prompt": False,
            }
            st_start = time.time()
            ttft_ms, ttfb_ms, total_ms, stream_text, stream_meta = stream_completion(
                base_url, s_payload, args.timeout
            )
            st_end = time.time()
            rec["ttft_ms"] = round(ttft_ms, 2) if ttft_ms is not None else None
            rec["ttfb_ms"] = round(ttfb_ms, 2) if ttfb_ms is not None else None
            rec["stream_total_ms"] = round(total_ms, 2)
            rec["stream_error"] = stream_meta.get("error")
            rec["stream_text"] = stream_text[:120]
            rec["stream_memory"] = sampler.window_peak(st_start, st_end)
            entry["runs"].append(rec)
            (raw_dir / f"{args.tier}-ctx{ctx}-r{rep + 1}.json").write_text(json.dumps(rec, indent=2))

            if rec["status"] != 200 or stream_meta.get("error"):
                entry["status"] = "error"
                entry["error"] = rec.get("error") or stream_meta.get("error")
                failures.append(
                    {
                        "context": ctx,
                        "repeat": rep + 1,
                        "status": rec["status"],
                        "error": entry["error"],
                    }
                )
                break
        t_end = time.time()
        entry["memory"] = sampler.window_peak(t_start, t_end)
        entry["summary"] = {
            "prompt_tps": summarize(entry["runs"], "prompt_tps"),
            "output_tps": summarize(entry["runs"], "output_tps"),
            "ttft_ms": summarize(entry["runs"], "ttft_ms"),
            "prompt_tokens": summarize(entry["runs"], "prompt_tokens"),
        }
        if entry["summary"].get("prompt_tokens"):
            entry["actual_prompt_tokens_median"] = entry["summary"]["prompt_tokens"]["median"]
        if entry["status"] == "ok":
            highest_working = ctx
        results.append(entry)
        if entry["status"] != "ok" and args.stop_on_error:
            for later in contexts[idx + 1:]:
                results.append(
                    {
                        "target_context": later,
                        "status": "skipped",
                        "reason": f"stopped after error at context {ctx}",
                        "runs": [],
                        "summary": {},
                    }
                )
            break

    # ---- needle ----------------------------------------------------------
    needle_ctx = args.needle_context
    needle = {"context": needle_ctx, "needle": NEEDLE, "status": "skipped"}
    if needle_ctx:
        if needle_ctx > context_limit:
            needle["reason"] = f"requested needle context {needle_ctx} > server n_ctx {context_limit}"
        else:
            target = max(1, min(needle_ctx, context_limit) - max(args.needle_tokens, 64))
            filler = tokenizer.size_to(target, CORPUS)
            depth = len(filler) // 2
            document = f"{filler[:depth]}\nThe secret access code for the vault is {NEEDLE}.\n{filler[depth:]}"
            question = (
                "\n\nQuestion: What is the secret access code for the vault? "
                "Answer with only the code.\nAnswer:"
            )
            res = completion_call(
                base_url, model_id, document + question, args.needle_tokens, args.timeout
            )
            answer = res.get("text") or ""
            needle = {
                "context": needle_ctx,
                "needle": NEEDLE,
                "prompt_tokens": res.get("prompt_tokens"),
                "status": "pass" if NEEDLE.lower() in answer.lower() else "fail",
                "answer": answer[:400],
                "http_status": res.get("status"),
                "error": res.get("error"),
            }
            (raw_dir / "needle.json").write_text(json.dumps(needle, indent=2))

    error_cases = run_error_cases(base_url, model_id, context_limit, args)
    sampler.stop()

    failed_contexts = [r["target_context"] for r in results if r["status"] in ("error", "skipped")]
    all_ok = not failed_contexts and needle.get("status") == "pass"
    matrix = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "harness": {"version": HARNESS_VERSION, "argv": sys.argv, "command": args.command},
        "endpoint": {"base_url": base_url, "model": model_id, "models_response": models_res.json},
        "tier": args.tier,
        "config": {
            "contexts": contexts,
            "repeats": args.repeats,
            "max_tokens": args.max_tokens,
            "ttft_tokens": args.ttft_tokens,
            "needle_context": needle_ctx,
            "needle_tokens": args.needle_tokens,
            "context_limit": context_limit,
            "hash_mode": args.hash_mode,
        },
        "machine": machine,
        "bongo_config": bongo_config,
        "server": {
            "pids": discovered,
            "flags": flags,
            "llama_cpp_build_info": llama_build,
            "props": props,
        },
        "model": {
            "id": model_id,
            "gguf_dir": gguf_dir,
            "shards": model_shards(gguf_dir, args.hash_mode, out_dir / ".hashes.json"),
        },
        "warmup": warmup,
        "results": results,
        "needle": needle,
        "error_cases": error_cases,
        "memory": aggregate_memory(results, sampler),
        "failures": failures,
        "highest_working_context": highest_working,
        "verdict": {
            "ok": all_ok,
            "failed_contexts": failed_contexts,
            "needle": needle.get("status"),
            "notes": build_notes(results, needle, context_limit, args),
        },
    }
    (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2))
    (out_dir / "matrix.md").write_text(render_markdown(matrix))
    print(f"wrote {out_dir / 'matrix.json'} and {out_dir / 'matrix.md'}")
    return 0 if all_ok else 3


def aggregate_memory(results, sampler):
    vram = [r.get("memory", {}).get("vram_peak_bytes") or 0 for r in results]
    ram = [r.get("memory", {}).get("system_ram_peak_bytes") or 0 for r in results]
    hwm = [r.get("memory", {}).get("system_ram_process_hwm_bytes") or 0 for r in results]
    return {
        "vram_peak_bytes": max(vram) if any(vram) else None,
        "vram_method": sampler.vram_method,
        "system_ram_peak_bytes": max(ram) if any(ram) else None,
        "system_ram_process_hwm_bytes": max(hwm) if any(hwm) else None,
        "system_ram_method": sampler.ram_method,
    }


def build_notes(results, needle, context_limit, args):
    notes = []
    if context_limit:
        notes.append(f"server reported n_ctx={context_limit}")
    if needle.get("status") == "fail":
        notes.append("128K needle retrieval FAILED: context may be accepted but not usable")
    for r in results:
        if r["status"] != "ok":
            notes.append(f"context {r['target_context']} failed: {(r.get('error') or '')[:200]}")
    return notes


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(m):
    lines = []
    lines.append(f"# bongo baseline benchmark — {m.get('tier', '?')}")
    lines.append("")
    lines.append(f"Generated: `{m.get('generated_at', '?')}`  ")
    ep = m.get("endpoint", {})
    lines.append(f"Endpoint: `{ep.get('base_url', '?')}`  ")
    lines.append(f"Model: `{ep.get('model', (m.get('model') or {}).get('id', '?'))}`  ")
    lines.append(f"Harness: `{m.get('harness', {}).get('version', '?')}`  ")
    lines.append(f"Command: `{m.get('harness', {}).get('command', ' '.join(m.get('harness', {}).get('argv', [])))}`")
    lines.append("")
    if "fatal_error" in m:
        lines.append("## FATAL")
        lines.append("")
        lines.append(f"```\n{json.dumps(m['fatal_error'], indent=2)}\n```")
        return "\n".join(lines) + "\n"

    verdict = m.get("verdict", {})
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- all measured contexts OK: **{verdict.get('ok')}**")
    lines.append(f"- highest context that worked: **{m.get('highest_working_context')}**")
    lines.append(f"- 128K needle: **{m.get('needle', {}).get('status')}**")
    for note in verdict.get("notes", []) or []:
        lines.append(f"- note: {note}")
    lines.append("")

    cfg = m.get("config", {})
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- contexts: `{cfg.get('contexts')}`  ")
    lines.append(f"- repeats: `{cfg.get('repeats')}`  ")
    lines.append(f"- max_tokens: `{cfg.get('max_tokens')}`  ")
    lines.append(f"- TTFT stream tokens: `{cfg.get('ttft_tokens')}`  ")
    lines.append(f"- server n_ctx: `{cfg.get('context_limit')}`  ")
    lines.append(f"- shard hash mode: `{cfg.get('hash_mode')}`  ")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "`prompt tok/s` is prefill throughput, `output tok/s` is decode throughput, "
        "`TTFT` is time to first streamed token. Values are the median of the "
        "repeats; `cv` is the coefficient of variation (stdev/median)."
    )
    lines.append("")
    lines.append("| context | prompt tokens | prompt tok/s | output tok/s | TTFT ms | prefill ms | repeats | cv(ttft) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in m.get("results", []):
        s = r.get("summary", {})
        pt = s.get("prompt_tokens") or {}
        pp = s.get("prompt_tps") or {}
        op = s.get("output_tps") or {}
        tt = s.get("ttft_ms") or {}
        first = r.get("runs", [{}])[0] if r.get("runs") else {}
        lines.append(
            "| {ctx} | {ptok} | {ptps} | {otps} | {ttft} | {pm} | {n} | {cv} |".format(
                ctx=r.get("target_context"),
                ptok=num(pt.get("median"), 0),
                ptps=num(pp.get("median")),
                otps=num(op.get("median")),
                ttft=num(tt.get("median"), 1),
                pm=num(first.get("prompt_ms"), 1),
                n=(op or pt or tt).get("n", 0),
                cv=num(tt.get("cv"), 3) if tt.get("cv") is not None else "n/a",
            )
        )
    lines.append("")

    lines.append("## Memory")
    lines.append("")
    mem = m.get("memory", {})
    lines.append(f"- peak VRAM: **{human_bytes(mem.get('vram_peak_bytes'))}** (method: `{mem.get('vram_method')}`)")
    lines.append(
        f"- peak system RAM (process RSS): **{human_bytes(mem.get('system_ram_peak_bytes'))}** "
        f"(method: `{mem.get('system_ram_method')}`)"
    )
    lines.append(
        f"- process RSS high-water mark: {human_bytes(mem.get('system_ram_process_hwm_bytes'))}"
    )
    lines.append("")

    lines.append("## 128K needle")
    lines.append("")
    nd = m.get("needle", {})
    lines.append(f"- status: **{nd.get('status')}**")
    lines.append(f"- prompt tokens: {nd.get('prompt_tokens')}")
    lines.append(f"- expected token: `{nd.get('needle')}`")
    lines.append(f"- answer: `{(nd.get('answer') or '').strip()[:200]}`")
    if nd.get("error"):
        lines.append(f"- error: `{nd['error']}`")
    lines.append("")

    lines.append("## Machine spec")
    lines.append("")
    mac = m.get("machine", {})
    cpu_line = next((l.split(":", 1)[1].strip() for l in (mac.get("cpu") or "").splitlines() if "Model name" in l), None)
    if not cpu_line:
        cpu_line = (mac.get("cpu") or "").splitlines()[0] if mac.get("cpu") else "n/a"
    mem_line = next((l for l in (mac.get("memory") or "").splitlines() if l.startswith("Mem:")), None)
    lines.append(f"- uname: `{mac.get('uname')}`")
    lines.append(f"- CPU: `{cpu_line}`")
    lines.append(f"- memory: `{mem_line or 'n/a'}`")
    lines.append(f"- display devices:")
    lines.append("")
    lines.append("  ```")
    for line in (mac.get("pci_display") or "").splitlines():
        lines.append(f"  {line}")
    lines.append("  ```")
    lines.append(f"- DRM drivers: `{mac.get('drivers')}`")
    lines.append(f"- modules: `{mac.get('modules')}`")
    lines.append(f"- DRI nodes: `{mac.get('dri_nodes')}`")
    lines.append("")

    lines.append("## Server")
    lines.append("")
    srv = m.get("server", {})
    lines.append(f"- pids: `{srv.get('pids')}`")
    lines.append(f"- llama.cpp build: `{srv.get('llama_cpp_build_info')}`")
    lines.append(f"- flags:")
    lines.append("")
    lines.append("  ```")
    lines.append("  " + " ".join(srv.get("flags") or []) if srv.get("flags") else "  n/a")
    lines.append("  ```")
    lines.append("")

    model = m.get("model") or {}
    lines.append("## Model / shards")
    lines.append("")
    lines.append(f"- gguf dir: `{model.get('gguf_dir')}`")
    if m.get("bongo_config"):
        lines.append(f"- bongo config: `{m['bongo_config'].get('path')}`")
    shards = model.get("shards")
    if shards:
        lines.append("")
        lines.append("| shard | size | sha256 |")
        lines.append("| --- | ---: | --- |")
        for s in shards:
            lines.append(f"| `{s['path']}` | {human_bytes(s['size_bytes'])} | `{s.get('sha256')}` |")
    else:
        lines.append("- shards: not found (pass `--gguf-dir`)")
    lines.append("")

    lines.append("## Error / malformed-prompt cases")
    lines.append("")
    lines.append("| case | HTTP | error status | excerpt |")
    lines.append("| --- | ---: | --- | --- |")
    for c in m.get("error_cases", []):
        excerpt = (c.get("response_excerpt") or "").replace("\n", " ")[:120].replace("|", "\\|")
        lines.append(f"| {c.get('name')} | {c.get('http_status')} | {c.get('is_error_status')} | `{excerpt}` |")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="bongo benchmark harness")
    p.add_argument("--base-url", default=os.environ.get("BONGO_BASE_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--model", default=os.environ.get("BONGO_MODEL"))
    p.add_argument("--tier", default=os.environ.get("BONGO_TIER", "iq2_xs"))
    p.add_argument(
        "--contexts",
        default=os.environ.get("BONGO_CONTEXTS", "1024,4096,32768,131072"),
        help="comma-separated context targets",
    )
    p.add_argument("--repeats", type=int, default=int(os.environ.get("BONGO_REPEATS", "3")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("BONGO_MAX_TOKENS", "128")))
    p.add_argument("--ttft-tokens", type=int, default=int(os.environ.get("BONGO_TTFT_TOKENS", "8")))
    p.add_argument("--needle-context", type=int, default=int(os.environ.get("BONGO_NEEDLE_CONTEXT", "131072")))
    p.add_argument("--needle-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=float(os.environ.get("BONGO_TIMEOUT", "3600")))
    p.add_argument("--server-pid", type=int, action="append", default=None)
    p.add_argument("--gguf-dir", default=None)
    p.add_argument("--hash-mode", choices=["full", "sampled", "none"], default="full")
    p.add_argument("--memory-interval", type=float, default=0.25)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    args.contexts = [int(x) for x in str(args.contexts).replace(" ", "").split(",") if x]
    if not args.out_dir:
        args.out_dir = str(Path(args.repo_root) / "bench" / "results" / f"{time.strftime('%Y-%m-%d')}-baseline")
    args.command = " ".join(["bench/run.sh"] + (argv or sys.argv[1:]))
    return args


def main(argv=None):
    args = parse_args(argv)
    return benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
