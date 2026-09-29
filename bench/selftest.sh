#!/usr/bin/env bash
# Self-test for the benchmark harness.  Runs bench/run.sh against
# bench/mock_server.py (no GPU, no model) and checks the produced matrix.
#
#   ./bench/selftest.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "${here}")"
scratch="${PAPERCLIP_RUN_SCRATCH_DIR:-${PAPERCLIP_SCRATCH_DIR:-$(mktemp -d)}}"
work="$(mktemp -d "${scratch%/}/bongo-bench-selftest.XXXXXX")"
happy_port=18080
fail_port=18081
dead_port=19999
pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  rm -rf "$work"
}
trap cleanup EXIT

wait_ready() {
  local url="$1" tries=50
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  echo "mock at $url did not become ready" >&2
  return 1
}

fail() { echo "SELFTEST FAIL: $*" >&2; exit 1; }

# ---- 1. happy path ---------------------------------------------------------
python3 "${here}/mock_server.py" --port "$happy_port" --ctx 262144 --needle-ok >/dev/null 2>&1 &
pids+=("$!")
wait_ready "http://127.0.0.1:${happy_port}"

"${here}/run.sh" \
  --base-url "http://127.0.0.1:${happy_port}/v1" \
  --contexts 1024,4096,32768,131072 --repeats 1 \
  --needle-context 131072 --hash-mode none \
  --server-pid "$(pgrep -f "mock_server.py --port ${happy_port}" | head -1)" \
  --out-dir "${work}/happy" || fail "happy-path run exited non-zero"

python3 - "${work}/happy/matrix.json" <<'PY' || fail "happy-path matrix assertions"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["schema"] == "bongo-bench/1", m["schema"]
assert m["verdict"]["ok"] is True, m["verdict"]
assert m["needle"]["status"] == "pass", m["needle"]
assert m["highest_working_context"] == 131072, m["highest_working_context"]
assert len(m["results"]) == 4, len(m["results"])
for r in m["results"]:
    assert r["status"] == "ok", (r["target_context"], r["status"])
    assert r["summary"]["prompt_tps"], r["target_context"]
    assert r["summary"]["output_tps"], r["target_context"]
    assert r["summary"]["ttft_ms"], r["target_context"]
    assert r["actual_prompt_tokens_median"] > 0
names = {c["name"] for c in m["error_cases"]}
assert "prompt_exceeds_context" in names
assert len(m["error_cases"]) >= 6
long = next(c for c in m["error_cases"] if c["name"] == "prompt_exceeds_context")
assert long["http_status"] and long["http_status"] >= 400, long
print("happy-path assertions OK")
PY
test -s "${work}/happy/matrix.md" || fail "matrix.md missing/empty"
grep -q "bongo baseline benchmark" "${work}/happy/matrix.md" || fail "matrix.md title missing"
echo "PASS: happy path"

# ---- 2. negative path (128K-style OOM) ------------------------------------
python3 "${here}/mock_server.py" --port "$fail_port" --ctx 262144 --fail-above 8000 >/dev/null 2>&1 &
pids+=("$!")
wait_ready "http://127.0.0.1:${fail_port}"

set +e
"${here}/run.sh" \
  --base-url "http://127.0.0.1:${fail_port}/v1" \
  --contexts 1024,4096,32768,131072 --repeats 1 \
  --needle-context 131072 --hash-mode none \
  --out-dir "${work}/negative" >/dev/null 2>&1
neg_rc=$?
set -e
[ "$neg_rc" -eq 3 ] || fail "negative run exit code $neg_rc (expected 3)"

python3 - "${work}/negative/matrix.json" <<'PY' || fail "negative-path matrix assertions"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["verdict"]["ok"] is False
assert m["highest_working_context"] == 4096, m["highest_working_context"]
assert m["needle"]["status"] == "fail", m["needle"]
assert m["needle"]["http_status"] == 500, m["needle"]
assert m["failures"], "expected recorded failures"
assert "out of memory" in (m["failures"][0]["error"] or "")
skipped = [r for r in m["results"] if r["status"] == "skipped"]
assert skipped, "expected skipped contexts after the failure"
print("negative-path assertions OK")
PY
echo "PASS: negative path (OOM recorded with exact error)"

# ---- 3. unreachable endpoint ----------------------------------------------
set +e
"${here}/run.sh" \
  --base-url "http://127.0.0.1:${dead_port}/v1" \
  --contexts 1024 --repeats 1 --needle-context 1024 --hash-mode none \
  --out-dir "${work}/dead" >/dev/null 2>&1
dead_rc=$?
set -e
[ "$dead_rc" -eq 2 ] || fail "dead-endpoint exit code $dead_rc (expected 2)"
python3 - "${work}/dead/matrix.json" <<'PY' || fail "dead-endpoint matrix assertions"
import json, sys
m = json.load(open(sys.argv[1]))
assert "fatal_error" in m, "fatal matrix missing fatal_error"
assert m["verdict"]["ok"] is False
print("dead-endpoint assertions OK")
PY
echo "PASS: unreachable endpoint recorded as fatal"

# ---- 4. tight context: prompt must never overflow n_ctx -------------------
python3 "${here}/mock_server.py" --port 18082 --ctx 4096 >/dev/null 2>&1 &
pids+=("$!")
wait_ready "http://127.0.0.1:18082"

"${here}/run.sh" \
  --base-url "http://127.0.0.1:18082/v1" \
  --contexts 4096 --repeats 1 \
  --needle-context 4096 --hash-mode none \
  --out-dir "${work}/tight" || fail "tight-context run exited non-zero"
python3 - "${work}/tight/matrix.json" <<'PY' || fail "tight-context matrix assertions"
import json, sys
m = json.load(open(sys.argv[1]))
ctx = m["config"]["context_limit"]
max_tokens = m["config"]["max_tokens"]
assert ctx == 4096, ctx
assert m["verdict"]["ok"] is True, m["verdict"]
r = m["results"][0]
assert r["status"] == "ok", r
assert r["actual_prompt_tokens_median"] + max_tokens <= ctx, (
    r["actual_prompt_tokens_median"], max_tokens, ctx)
assert m["needle"]["status"] == "pass", m["needle"]
assert m["needle"]["http_status"] == 200, m["needle"]
assert m["needle"]["prompt_tokens"] <= ctx, m["needle"]
print("tight-context assertions OK")
PY
echo "PASS: prompt fits tight n_ctx (no overflow)"

# ---- 5. per-context repeat budget -----------------------------------------
python3 "${here}/mock_server.py" --port 18083 --ctx 262144 >/dev/null 2>&1 &
pids+=("$!")
wait_ready "http://127.0.0.1:18083"

"${here}/run.sh" \
  --base-url "http://127.0.0.1:18083/v1" \
  --contexts 1024,4096 --repeats 3 --repeats-deep 1 --deep-threshold 4096 \
  --needle-context 4096 --hash-mode none \
  --out-dir "${work}/deep" || fail "deep-repeat run exited non-zero"
python3 - "${work}/deep/matrix.json" <<'PY' || fail "deep-repeat matrix assertions"
import json, sys
m = json.load(open(sys.argv[1]))
by = {r["target_context"]: r for r in m["results"]}
assert m["config"]["repeats"] == 3, m["config"]
assert m["config"]["repeats_deep"] == 1, m["config"]
assert m["config"]["deep_threshold"] == 4096, m["config"]
assert len(by[1024]["runs"]) == 3, len(by[1024]["runs"])
assert len(by[4096]["runs"]) == 1, len(by[4096]["runs"])
for r in by.values():
    assert r["summary"]["prompt_tps"]["n"] == len(r["runs"]), r["target_context"]
print("deep-repeat assertions OK")
PY
echo "PASS: per-context repeat budget (deep contexts repeat less)"

# ---- 6. cache_prompt mode (prefix reuse) ----------------------------------
cache_port=18084
cache_slot_dir="${work}/slots"
python3 "${here}/mock_server.py" --port "$cache_port" --ctx 262144 --slot-save-path "$cache_slot_dir" >/dev/null 2>&1 &
pids+=("$!")
wait_ready "http://127.0.0.1:${cache_port}"

"${here}/run.sh" \
  --base-url "http://127.0.0.1:${cache_port}/v1" \
  --contexts 1024,4096 --repeats 1 --needle-context 4096 --hash-mode none \
  --out-dir "${work}/cache" || fail "cache-path run exited non-zero"
python3 - "${work}/cache/matrix.json" <<'PY' || fail "cache-path matrix assertions"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["config"]["cache_prompt"] is True, m["config"]
assert m["config"]["profile"] == "agentic", m["config"]
by = {r["target_context"]: r for r in m["results"]}
assert by[1024]["summary"]["cache_n"]["median"] == 0, by[1024]["summary"]["cache_n"]
assert by[4096]["summary"]["cache_n"]["median"] > 0, by[4096]["summary"]["cache_n"]
assert by[4096]["summary"]["cached_tokens"]["median"] > 0, by[4096]["summary"]["cached_tokens"]
print("cache-path assertions OK")
PY
echo "PASS: cache_prompt mode records cache_n / cached_tokens"

# ---- 7. cold baseline (--no-cache-prompt) ---------------------------------
"${here}/run.sh" \
  --base-url "http://127.0.0.1:${cache_port}/v1" --no-cache-prompt \
  --contexts 1024,4096 --repeats 1 --needle-context 4096 --hash-mode none \
  --out-dir "${work}/cold" || fail "cold-path run exited non-zero"
python3 - "${work}/cold/matrix.json" <<'PY' || fail "cold-path matrix assertions"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["config"]["cache_prompt"] is False, m["config"]
assert m["config"]["profile"] == "baseline", m["config"]
for r in m["results"]:
    assert (r["summary"].get("cache_n") or {}).get("median") == 0, r["target_context"]
print("cold-path assertions OK")
PY
echo "PASS: --no-cache-prompt reproduces the cold baseline"

# ---- 8. slot save/restore measurement -------------------------------------
BONGO_BASE_URL="http://127.0.0.1:${cache_port}/v1" BONGO_MODEL=bongo-mock \
  python3 "${here}/measure-prefix-cache.py" \
  --prefixes 1024 --delta 64 --slot-id 0 --slot-save-dir "$cache_slot_dir" \
  --out "${work}/pcache" >/dev/null || fail "measure-prefix-cache run exited non-zero"
python3 - "${work}/pcache/prefix-cache.json" <<'PY' || fail "prefix-cache slot assertions"
import json, sys
d = json.load(open(sys.argv[1]))
labels = {r["label"]: r for r in d["runs"]}
assert labels["hit_p1024"]["cache_n"] > 0, labels["hit_p1024"]
assert labels["grow_p1024_d64"]["cache_n"] > 0, labels["grow_p1024_d64"]
assert labels["cold_p1024"]["cache_prompt"] is False
slot = d["slot"]
assert slot and slot["save"]["status"] == 200, slot
assert slot["restore"]["status"] == 200, slot
assert slot["save"]["file_bytes"] > 0, slot["save"]
assert slot["restore_verified"] is True, slot
print("prefix-cache slot assertions OK")
PY
echo "PASS: slot save/restore timed and restore verified"

# ---- 9. cached-only delta turn (no cold pass, VRAM window recorded) --------
BONGO_BASE_URL="http://127.0.0.1:${cache_port}/v1" BONGO_MODEL=bongo-mock \
  python3 "${here}/measure-prefix-cache.py" \
  --prefixes 1024 --delta 64 --no-slot --cached-only --timeout 1234 \
  --out "${work}/pcache-cached" >/dev/null || fail "cached-only run exited non-zero"
python3 - "${work}/pcache-cached/prefix-cache.json" <<'PY' || fail "cached-only assertions"
import json, sys
d = json.load(open(sys.argv[1]))
labels = {r["label"]: r for r in d["runs"]}
assert "cold_p1024" not in labels, labels.keys()
assert labels["prime_p1024"]["cache_prompt"] is True, labels["prime_p1024"]
assert labels["prime_p1024"]["prompt_tokens"] > 0, labels["prime_p1024"]
assert labels["hit_p1024"]["cache_n"] > 0, labels["hit_p1024"]
assert labels["grow_p1024_d64"]["cache_n"] > 0, labels["grow_p1024_d64"]
assert d["cached_only"] is True, d["cached_only"]
assert d["request_timeout_s"] == 1234, d["request_timeout_s"]
assert "memory" in d and "vram_method" in d["memory"], d.get("memory")
assert all("memory" in r for r in d["runs"]), "per-case memory window missing"
print("cached-only assertions OK")
PY
echo "PASS: --cached-only primes the prefix and records a per-case VRAM window"

echo "ALL SELFTESTS PASSED"
