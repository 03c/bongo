#!/usr/bin/env bash
# Self-test for bench/gpu-lock.sh (BAS-80).
#
# Proves single-GPU mutual exclusion without a GPU or a llama-server:
#   1. first acquire succeeds and records a holder
#   2. a second in-process acquire is a no-op
#   3. a child process is refused while the lock is held (fail fast)
#   4. a child can acquire after the holder releases
#   5. BONGO_GPU_LOCK_HELD=1 skips acquisition (wrapper -> bongo.sh hand-off)
#   6. the command form is mutually exclusive too
#
# Run: bash bench/gpu-lock-selftest.sh
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

export BONGO_GPU_LOCK="$scratch/gpu.lock"
export BONGO_GPU_LOCK_TIMEOUT=0
# shellcheck disable=SC1090
. "$here/gpu-lock.sh"

pass=0
fail=0
ok() { pass=$((pass + 1)); printf 'ok   - %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf 'FAIL - %s\n' "$1"; }

holder="$(bongo_gpu_lock_holder_path)"

# 1. first acquire
if bongo_gpu_lock_acquire "selftest-parent"; then
  if grep -q "^pid=$$ " "$holder" 2>/dev/null; then
    ok "first acquire succeeds and records the holder"
  else
    bad "first acquire succeeds and records the holder (no holder file)"
  fi
else
  bad "first acquire succeeds and records the holder"
fi

# 2. re-entrant in-process acquire is a no-op
if bongo_gpu_lock_acquire "selftest-parent-again"; then
  ok "re-entrant in-process acquire is a no-op"
else
  bad "re-entrant in-process acquire is a no-op"
fi

# 3. a child is refused while the parent holds the lock
if BONGO_GPU_LOCK_HELD=0 bash -c '
    . "'"$here"'/gpu-lock.sh"
    bongo_gpu_lock_acquire child >/dev/null 2>&1
  '; then
  bad "child is refused while the lock is held"
else
  ok "child is refused while the lock is held (fail fast)"
fi

# 4. release then a child can acquire and release
bongo_gpu_lock_release
if [[ -e "$holder" ]]; then bad "release removes the holder file"; else ok "release removes the holder file"; fi
if bash -c '
    . "'"$here"'/gpu-lock.sh"
    bongo_gpu_lock_acquire child && bongo_gpu_lock_release
  '; then
  ok "child can acquire after the holder releases"
else
  bad "child can acquire after the holder releases"
fi

# 5. BONGO_GPU_LOCK_HELD=1 skips acquisition (wrapper already holds it)
if BONGO_GPU_LOCK_HELD=1 bash -c '
    . "'"$here"'/gpu-lock.sh"
    bongo_gpu_lock_acquire child && [[ -z "${BONGO_GPU_LOCK_FD:-}" ]]
  '; then
  ok "BONGO_GPU_LOCK_HELD=1 skips acquisition"
else
  bad "BONGO_GPU_LOCK_HELD=1 skips acquisition"
fi

# 6. the lock follows an inherited fd (server lifetime), not the wrapper
bongo_gpu_lock_acquire "selftest-server-lifetime" || bad "server-lifetime: acquire"
( sleep 3 ) &
server_fd_child=$!
sleep 0.3
bongo_gpu_lock_release   # close the wrapper descriptor only
if BONGO_GPU_LOCK_HELD=0 bash -c '
    . "'"$here"'/gpu-lock.sh"
    bongo_gpu_lock_acquire child >/dev/null 2>&1
  '; then
  bad "lock stays held while the inherited server fd lives"
else
  ok "lock stays held while the inherited server fd lives"
fi
wait "$server_fd_child" 2>/dev/null
if BONGO_GPU_LOCK_HELD=0 bash -c '
    . "'"$here"'/gpu-lock.sh"
    bongo_gpu_lock_acquire child && bongo_gpu_lock_release
  '; then
  ok "lock frees when the inherited holder exits"
else
  bad "lock frees when the inherited holder exits"
fi

# 7. command form is mutually exclusive
if "$here/gpu-lock.sh" --timeout 0 -- true >/dev/null 2>&1; then
  ok "command form runs when free"
else
  bad "command form runs when free"
fi
BONGO_GPU_LOCK_FD=""
(
  . "$here/gpu-lock.sh"
  bongo_gpu_lock_acquire "selftest-hold" || exit 1
  if "$here/gpu-lock.sh" --timeout 0 -- true >/dev/null 2>&1; then
    exit 7      # unexpectedly acquired
  fi
  exit 0        # correctly refused
)
case $? in
  0) ok "command form is refused while the lock is held" ;;
  *) bad "command form is refused while the lock is held" ;;
esac

printf '\n%d passed, %d failed\n' "$pass" "$fail"
(( fail == 0 ))
