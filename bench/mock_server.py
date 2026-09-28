#!/usr/bin/env python3
"""A tiny OpenAI-compatible mock server used to self-test the harness.

It emulates the parts of ``llama-server`` that ``bench/harness.py`` relies on:
``/v1/models``, ``/v1/completions`` (stream and non-stream), ``/tokenize`` and
``/props``.  It is *not* a model; timings are synthetic.  Only used by
``bench/selftest.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "bongo-mock/1.0"

    # -- helpers -----------------------------------------------------------
    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw.decode("utf-8", "replace"))

    def _send(self, status, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def log_message(self, *_args):  # keep the test output clean
        pass

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": self.server.model_id, "object": "model"}]})
        elif path == "/props":  # real llama-server exposes /props at the root
            self._send(
                200,
                {
                    "default_generation_settings": {"n_ctx": self.server.ctx},
                    "build_info": "mock-build (deadbeef)",
                    "model_path": "/mock/model.gguf",
                },
            )
        elif path == "/health":
            self._send(200, {"status": "ok"})
        elif path == "/slots":
            # Field names follow the b11223 Vulkan build, which reports
            # n_prompt_tokens{,_processed,_cache} and no longer the older
            # `n_past`. Scripts that read only `n_past` see None here too.
            self._send(
                200,
                [
                    {
                        "id": 0,
                        "n_ctx": self.server.ctx,
                        "is_processing": False,
                        "n_prompt_tokens": len(self.server.cache_tokens),
                        "n_prompt_tokens_processed": len(self.server.cache_tokens),
                        "n_prompt_tokens_cache": len(self.server.cache_tokens),
                    }
                ],
            )
        else:
            self._error(404, "not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/tokenize":  # root only, like llama-server
                obj = self._read_json()
                content = obj.get("content", "")
                tokens = content.split()
                self._send(200, {"tokens": list(range(len(tokens)))})
                return
            if path == "/v1/completions":
                self._completions()
                return
            if path.startswith("/slots/"):
                self._slots(parsed)
                return
            self._error(404, "not found")
        except json.JSONDecodeError:
            self._error(400, "invalid JSON body")
        except Exception as exc:  # noqa: BLE001
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _slots(self, parsed):
        """Minimal /slots/{id}?action=save|restore|erase, saving the cached tokens."""
        srv = self.server
        if not srv.slot_dir:
            self._error(501, "This server does not support slots action. Start it with `--slot-save-path`")
            return
        try:
            slot_id = int(parsed.path.rsplit("/", 1)[-1])
        except ValueError:
            self._error(400, "Invalid slot ID")
            return
        action = (parse_qs(parsed.query).get("action") or [""])[0]
        obj = self._read_json() if int(self.headers.get("Content-Length", "0") or 0) else {}
        filename = os.path.basename(str(obj.get("filename", "")))
        path = os.path.join(srv.slot_dir, filename) if filename else None
        if action == "save":
            if not filename:
                self._error(400, "Invalid filename")
                return
            with open(path, "w") as fh:
                json.dump({"slot_id": slot_id, "tokens": srv.cache_tokens}, fh)
            self._send(200, {"n_saved": len(srv.cache_tokens)})
        elif action == "restore":
            if not filename or not os.path.isfile(path):
                self._error(400, "Invalid filename")
                return
            with open(path) as fh:
                srv.cache_tokens = json.load(fh).get("tokens", [])
            self._send(200, {"n_restored": len(srv.cache_tokens)})
        elif action == "erase":
            n_erased = len(srv.cache_tokens)
            srv.cache_tokens = []
            self._send(200, {"n_erased": n_erased})
        else:
            self._error(400, "Invalid action")

    def _completions(self):
        srv = self.server
        obj = self._read_json()
        if "prompt" not in obj:
            self._error(400, "missing required field: prompt")
            return
        prompt = obj.get("prompt")
        if not isinstance(prompt, str):
            self._error(400, "prompt must be a string")
            return
        if prompt == "":
            self._error(400, "prompt is empty")
            return
        max_tokens = obj.get("max_tokens", 16)
        if not isinstance(max_tokens, int) or max_tokens < 0:
            self._error(400, "max_tokens must be a non-negative integer")
            return
        stream = bool(obj.get("stream", False))
        tokens = prompt.split()
        total = len(tokens)

        # Emulate llama-server prefix reuse: cache_n is the common token prefix
        # with the prompt currently held by the slot. cache_prompt defaults to
        # true, matching the server default (--cache-prompt is on by default).
        cache_prompt = bool(obj.get("cache_prompt", True))
        cache_n = 0
        if cache_prompt:
            cached = srv.cache_tokens
            while cache_n < total and cache_n < len(cached) and tokens[cache_n] == cached[cache_n]:
                cache_n += 1
        srv.cache_tokens = tokens
        prompt_n = total - cache_n
        if prompt_n + max_tokens > srv.ctx:
            self._error(400, f"the prompt is too long ({total} + {max_tokens} > n_ctx {srv.ctx})")
            return

        if srv.fail_above is not None and total > srv.fail_above:
            self._error(
                500,
                f"failed to process prompt: out of memory (prompt_n={total} > fail_above={srv.fail_above})",
            )
            return

        prefill_ms = (prompt_n / srv.prefill_tps) * 1000.0
        decode_ms = (max_tokens / srv.decode_tps) * 1000.0
        text = " 7391" if "vault" in prompt.lower() and srv.needle_ok else " mock completion"
        if srv.needle_ok and "7391" in obj.get("prompt", ""):
            text = "VAULT-COORD-7391-QXZ"

        timings = {
            "prompt_n": prompt_n,
            "cache_n": cache_n,
            "prompt_ms": prefill_ms,
            "prompt_per_token_ms": (prefill_ms / prompt_n) if prompt_n else 0.0,
            "prompt_per_second": srv.prefill_tps if prompt_n else 0.0,
            "predicted_n": max_tokens,
            "predicted_ms": decode_ms,
            "predicted_per_token_ms": decode_ms / max_tokens if max_tokens else None,
            "predicted_per_second": srv.decode_tps,
        }
        usage = {
            "prompt_tokens": total,
            "completion_tokens": max_tokens,
            "total_tokens": total + max_tokens,
            "prompt_tokens_details": {"cached_tokens": cache_n},
        }

        if not stream:
            self._send(
                200,
                {
                    "id": "cmpl-mock",
                    "object": "text_completion",
                    "model": srv.model_id,
                    "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
                    "usage": usage,
                    "timings": timings,
                },
            )
            return

        # streaming: first token after prompt processing (scaled for test speed)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(min(prefill_ms / 1000.0, srv.max_ttft_delay))
        for i in range(max_tokens):
            chunk = {
                "id": "cmpl-mock",
                "object": "text_completion",
                "choices": [{"index": 0, "text": ("VAULT-COORD-7391-QXZ " if (srv.needle_ok and i == 0) else "tok"), "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        final = {
            "id": "cmpl-mock",
            "object": "text_completion",
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
            "usage": usage,
            "timings": timings,
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--fail-above", type=int, default=None)
    ap.add_argument("--prefill-tps", type=float, default=1000.0)
    ap.add_argument("--decode-tps", type=float, default=1000.0)
    ap.add_argument("--max-ttft-delay", type=float, default=0.05)
    ap.add_argument("--model-id", default="bongo-mock")
    ap.add_argument("--slot-save-path", default=None)
    ap.add_argument("--needle-ok", dest="needle_ok", action="store_true", default=True)
    ap.add_argument("--needle-fail", dest="needle_ok", action="store_false")
    args = ap.parse_args()

    if args.slot_save_path:
        os.makedirs(args.slot_save_path, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.ctx = args.ctx
    httpd.fail_above = args.fail_above
    httpd.prefill_tps = args.prefill_tps
    httpd.decode_tps = args.decode_tps
    httpd.max_ttft_delay = args.max_ttft_delay
    httpd.model_id = args.model_id
    httpd.needle_ok = args.needle_ok
    httpd.slot_dir = args.slot_save_path
    httpd.cache_tokens = []
    print(f"mock listening on http://{args.host}:{args.port}/v1 ctx={args.ctx}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
