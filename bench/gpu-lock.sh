#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-TBD
#
# bench/gpu-lock.sh — advisory single-GPU lock for bongo (BAS-80).
#
# The reference box has one Intel Arc Pro B70 (32 GiB). Two `llama-server`
# processes cannot fit: the `xe` driver evicts VRAM, and every number measured
# while they overlap is corrupted (a ~20x prompt-tok/s drop was observed). This
# helper makes a server start mutually exclusive across worktrees and agent
# runs, so measurement tasks serialise instead of thrashing.
#
# Source it and hold the lock for the *whole* measured run — server start and
# measurement, not just process launch:
#
#   . "$repo/bench/gpu-lock.sh"
#   bongo_gpu_lock_acquire "run-speculation-ab baseline" || exit 3
#   trap 'bongo_gpu_lock_release' EXIT INT TERM
#
# If the script starts the server by calling ./bongo.sh, the acquire function
# exports BONGO_GPU_LOCK_HELD=1, so bongo.sh will not try to take the same lock
# twice. A wrapper must pass the marker through `systemd-run --scope` with
# `--setenv=BONGO_GPU_LOCK_HELD=1`.
#
# Command form (for manual / ad-hoc use, wraps the system `flock`):
#
#   bench/gpu-lock.sh --timeout 0 -- some-command arg...
#
# Environment:
#   BONGO_GPU_LOCK          lock file path
#                           (default: ${BONGO_HOME:-$HOME/.bongo}/gpu.lock)
#   BONGO_GPU_LOCK_TIMEOUT  seconds to wait for a busy lock.
#                           0 (default) = fail fast; -1 = wait forever.
#
# The lock is advisory. A `llama-server` started without it (for example an
# older detached server) is not excluded by flock; keep the existing
# `pgrep`/port guards in the callers as the backstop.

# Absolute path of the lock file (may not exist yet).
bongo_gpu_lock_path() {
  printf '%s' "${BONGO_GPU_LOCK:-${BONGO_HOME:-$HOME/.bongo}/gpu.lock}"
}

# Path of the sidecar file that names the current holder (diagnostics only).
bongo_gpu_lock_holder_path() {
  printf '%s.holder' "$(bongo_gpu_lock_path)"
}

# bongo_gpu_lock_acquire [label]
# Returns 0 when the lock is held by this process tree, non-zero otherwise.
# When BONGO_GPU_LOCK_HELD=1 is already set (a wrapper already holds it) this
# is a no-op.
bongo_gpu_lock_acquire() {
  local label="${1:-${BASH_SOURCE[1]:-$0}}"

  if [[ "${BONGO_GPU_LOCK_HELD:-0}" == "1" ]]; then
    return 0
  fi

  local lock
  local holder
  lock="$(bongo_gpu_lock_path)"
  holder="$(bongo_gpu_lock_holder_path)"

  if ! command -v flock >/dev/null 2>&1; then
    echo "gpu-lock: 'flock' is required to serialise the single GPU (util-linux)." >&2
    return 1
  fi

  mkdir -p "$(dirname "$lock")" 2>/dev/null || true

  # Append so the file is never truncated; open on a high descriptor.
  if ! exec {BONGO_GPU_LOCK_FD}>>"$lock"; then
    echo "gpu-lock: cannot open lock file '$lock'." >&2
    return 1
  fi

  local timeout="${BONGO_GPU_LOCK_TIMEOUT:-0}"
  local rc=0
  if [[ "$timeout" == "0" ]]; then
    flock -n "$BONGO_GPU_LOCK_FD" || rc=$?
  else
    flock -w "$timeout" "$BONGO_GPU_LOCK_FD" || rc=$?
  fi

  if (( rc != 0 )); then
    local owner=""
    if [[ -r "$holder" ]]; then owner="$(cat "$holder" 2>/dev/null || true)"; fi
    echo "gpu-lock: another run holds the single GPU lock ($lock)." >&2
    [[ -n "$owner" ]] && echo "gpu-lock: holder: $owner" >&2
    echo "gpu-lock: refusing to start a second llama-server; do not overlap measurements (BAS-80)." >&2
    echo "gpu-lock: set BONGO_GPU_LOCK_TIMEOUT=<seconds> to queue, or retry once the holder exits." >&2
    exec {BONGO_GPU_LOCK_FD}>&- 2>/dev/null || true
    BONGO_GPU_LOCK_FD=""
    return 1
  fi

  BONGO_GPU_LOCK_HELD=1
  export BONGO_GPU_LOCK_HELD
  printf 'pid=%s host=%s since=%s label=%s\n' \
    "$$" "$(hostname 2>/dev/null || echo unknown)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" >"$holder" 2>/dev/null || true
  return 0
}

# bongo_gpu_lock_release
# Idempotent. Releases only a lock this process tree took.
bongo_gpu_lock_release() {
  [[ -n "${BONGO_GPU_LOCK_FD:-}" ]] || return 0

  local holder
  holder="$(bongo_gpu_lock_holder_path)"
  rm -f "$holder" 2>/dev/null || true
  flock -u "$BONGO_GPU_LOCK_FD" 2>/dev/null || true
  exec {BONGO_GPU_LOCK_FD}>&- 2>/dev/null || true
  BONGO_GPU_LOCK_FD=""
  BONGO_GPU_LOCK_HELD=0
  return 0
}

# ---------------------------------------------------------------------------
# Command form: bench/gpu-lock.sh [--timeout N] [--] command [args...]
# ---------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  timeout="${BONGO_GPU_LOCK_TIMEOUT:-0}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --timeout) timeout="${2:?--timeout needs a value}"; shift 2;;
      --) shift; break;;
      -h|--help)
        sed -n '2,45p' "$0"; exit 0;;
      *) break;;
    esac
  done
  [[ $# -gt 0 ]] || { echo "usage: $0 [--timeout N] [--] command [args...]" >&2; exit 2; }

  lock="$(bongo_gpu_lock_path)"
  command -v flock >/dev/null 2>&1 || { echo "gpu-lock: 'flock' is required." >&2; exit 1; }
  mkdir -p "$(dirname "$lock")" 2>/dev/null || true
  if [[ "$timeout" == "0" ]]; then
    exec flock -n "$lock" "$@"
  elif [[ "$timeout" == "-1" ]]; then
    exec flock "$lock" "$@"
  else
    exec flock -w "$timeout" "$lock" "$@"
  fi
fi
