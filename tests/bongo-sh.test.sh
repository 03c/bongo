#!/usr/bin/env bash
# Regression tests for the BAS-58 `bongo.sh` fixes.
#
# Covers idempotent user-local runtime provisioning (F1), the --uninstall fixes
# (F3/F4), and the fail-fast need_cmd guards (F5). No network access is needed:
# install_runtime_user() is exercised with dnf/rpm2cpio/cpio stubs.
#
# Run: bash tests/bongo-sh.test.sh
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/bongo.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

# Load bongo.sh without running main() so the helpers can be unit-tested.
LIB="$TMP/bongo-lib.sh"
sed 's/^main "\$@"$/:/' "$SRC" > "$LIB"
# shellcheck disable=SC1090
source "$LIB"
trap - ERR   # bongo.sh installs an ERR trap; keep test output clean

# Test helpers are defined after sourcing: bongo.sh defines its own ok().
t_ok() { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
t_bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n' "$1"; }
t_check() { # <description> <command...>
  local desc="$1"; shift
  if "$@"; then t_ok "$desc"; else t_bad "$desc"; fi
}

# --- F1: sycl_runtime_present honours the provisioning sentinel -------------
home="$TMP/home-sentinel"
BONGO_HOME="$home"; RUNTIME_DIR="$home/runtime"
mkdir -p "$RUNTIME_DIR"
: > "$RUNTIME_DIR/.bongo-provisioned"
t_check "F1 sentinel marks the runtime present" sycl_runtime_present

# --- F1: Level Zero / OpenCL loaders also count as present ------------------
home="$TMP/home-levelzero"
BONGO_HOME="$home"; RUNTIME_DIR="$home/runtime"
mkdir -p "$RUNTIME_DIR"
: > "$RUNTIME_DIR/libze_loader.so.1"
LD_LIBRARY_PATH="$RUNTIME_DIR"
t_check "F1 libze_loader.so counts as present" sycl_runtime_present

# --- F1: an empty prefix is still absent ------------------------------------
if command -v sycl-ls >/dev/null 2>&1 || [[ -e /opt/intel/oneapi/setvars.sh ]]; then
  echo "skip - absent-prefix check (host already has a SYCL runtime)"
else
  home="$TMP/home-absent"
  BONGO_HOME="$home"; RUNTIME_DIR="$home/runtime"
  mkdir -p "$RUNTIME_DIR"
  LD_LIBRARY_PATH="$RUNTIME_DIR"
  if sycl_runtime_present; then t_bad "F1 empty prefix is absent"; else t_ok "F1 empty prefix is absent"; fi
fi

# --- F1: install_runtime_user writes the sentinel, resolves deps, fetches SYCL
home="$TMP/home-install"
BONGO_HOME="$home"; RUNTIME_DIR="$home/runtime"
mkdir -p "$home"
DNF_LOG="$TMP/dnf.log"
dnf() { printf 'dnf %s\n' "$*" >> "$DNF_LOG"; return 0; }
rpm2cpio() { cat >/dev/null; return 0; }
cpio() {
  mkdir -p "$RUNTIME_DIR/usr/lib64"
  : > "$RUNTIME_DIR/usr/lib64/libze_loader.so.1"
  cat >/dev/null
  return 0
}
install_runtime_user >/dev/null 2>&1
if [[ -e "$RUNTIME_DIR/.bongo-provisioned" ]]; then
  t_ok "F1 install_runtime_user writes the sentinel"
else
  t_bad "F1 install_runtime_user writes the sentinel"
fi
if grep -q -- '--resolve' "$DNF_LOG"; then
  t_ok "F1 oneAPI download resolves dependencies"
else
  t_bad "F1 oneAPI download resolves dependencies"
fi
if grep -q 'intel-oneapi-runtime-dpcpp-sycl-core' "$DNF_LOG"; then
  t_ok "F1 oneAPI download requests the SYCL core package"
else
  t_bad "F1 oneAPI download requests the SYCL core package"
fi
if sycl_runtime_present; then
  t_ok "F1 provisioned prefix is detected on the next run"
else
  t_bad "F1 provisioned prefix is detected on the next run"
fi
unset -f dnf rpm2cpio cpio

# --- F3/F4: --uninstall output is correct and --yes performs removal --------
out="$(BONGO_HOME="$TMP/does-not-exist" bash "$SRC" --uninstall 2>&1)"
case "$out" in
  *clinfo*) t_ok "F3 --uninstall lists clinfo" ;;
  *) t_bad "F3 --uninstall lists clinfo" ;;
esac
case "$out" in
  *clinch*) t_bad "F3 --uninstall no longer lists clinch" ;;
  *) t_ok "F3 --uninstall no longer lists clinch" ;;
esac

home="$TMP/home-uninstall"
mkdir -p "$home/sub"
: > "$home/sub/marker"
out="$(BONGO_HOME="$home" bash "$SRC" --uninstall --yes 2>&1)"
if [[ ! -e "$home" ]]; then
  t_ok "F4 --uninstall --yes removes BONGO_HOME"
else
  t_bad "F4 --uninstall --yes removes BONGO_HOME"
fi
case "$out" in
  *"dnf remove"*|*"apt-get remove"*) t_ok "F4 --uninstall --yes prints the package removal command" ;;
  *) t_bad "F4 --uninstall --yes prints the package removal command" ;;
esac

# Print-only mode must not delete anything.
home="$TMP/home-print-only"
mkdir -p "$home"
bash "$SRC" --uninstall >/dev/null 2>&1 || true
if [[ -e "$home" ]]; then
  t_ok "F4 --uninstall without --yes keeps BONGO_HOME"
else
  t_bad "F4 --uninstall without --yes keeps BONGO_HOME"
fi

# --- F5: a host without curl gets an actionable message, not a traceback -----
emptybin="$TMP/emptybin"
mkdir -p "$emptybin"
set +e
out="$(env -u BASH_ENV PATH="$emptybin" /bin/bash "$SRC" --dry-run 2>&1)"
rc=$?
set -e
if (( rc != 0 )) && [[ "$out" == *"Required command 'curl' is not installed"* ]] \
   && [[ "$out" != *"failed at line"* ]]; then
  t_ok "F5 missing curl fails with an actionable message"
else
  t_bad "F5 missing curl fails with an actionable message"
  printf '%s\n' "$out" | sed 's/^/     /'
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
(( FAIL == 0 ))
