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

echo "ALL SELFTESTS PASSED"
