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

# --- BAS-57 F1: oneAPI ABI packages are pinned and extracted last ----------
# The pinned llama.cpp SYCL asset links libsycl.so.8 (oneAPI 2025.3). An
# unpinned download now also resolves to 2026.1 (libsycl.so.9), so the pinned
# packages must be requested by version and extracted after any newer copy.
home="$TMP/home-pin"
BONGO_HOME="$home"; RUNTIME_DIR="$home/runtime"
mkdir -p "$home"
PIN_DNF_LOG="$TMP/dnf-pin.log"
RPM_LOG="$TMP/rpm-pin.log"
LDCONF_LOG="$TMP/ldconfig-pin.log"
dnf() {
  printf 'dnf %s\n' "$*" >> "$PIN_DNF_LOG"
  # Simulate the download dir: finer pinned packages plus newer transitive
  # copies that dnf --resolve pulls in through the meta-packages.
  mkdir -p "$BONGO_HOME/tmp/runtime"
  : > "$BONGO_HOME/tmp/runtime/intel-oneapi-runtime-dpcpp-sycl-core-2026.1.1-325.x86_64.rpm"
  : > "$BONGO_HOME/tmp/runtime/intel-oneapi-runtime-mkl-2026.1.1-325.x86_64.rpm"
  : > "$BONGO_HOME/tmp/runtime/intel-oneapi-runtime-mkl-2025.3.1-8.x86_64.rpm"
  : > "$BONGO_HOME/tmp/runtime/intel-oneapi-runtime-dpcpp-sycl-core-2025.3.3-30.x86_64.rpm"
  return 0
}
rpm2cpio() { printf '%s\n' "$1" >> "$RPM_LOG"; cat >/dev/null; return 0; }
cpio() {
  mkdir -p "$RUNTIME_DIR/usr/lib64" "$RUNTIME_DIR/usr/bin" "$RUNTIME_DIR/opt/intel/oneapi/redist/lib"
  : > "$RUNTIME_DIR/opt/intel/oneapi/redist/lib/libsycl.so.8"
  # The prefix's ldconfig recreates SONAME symlinks for extracted libs.
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" >> "%s"\n' "$LDCONF_LOG" > "$RUNTIME_DIR/usr/bin/ldconfig"
  chmod +x "$RUNTIME_DIR/usr/bin/ldconfig"
  cat >/dev/null 2>/dev/null || true
  return 0
}
install_runtime_user >/dev/null 2>&1
if grep -q 'intel-oneapi-runtime-dpcpp-sycl-core-2025.3.3-30' "$PIN_DNF_LOG"; then
  t_ok "BAS-57 pins the SYCL core package to the 2025.3 ABI"
else
  t_bad "BAS-57 pins the SYCL core package to the 2025.3 ABI"
fi
if grep -q 'intel-oneapi-runtime-mkl-2025.3.1-8' "$PIN_DNF_LOG"; then
  t_ok "BAS-57 pins the MKL runtime package to the .so.5 ABI"
else
  t_bad "BAS-57 pins the MKL runtime package to the .so.5 ABI"
fi
pin_sycl_line="$(grep -n 'intel-oneapi-runtime-dpcpp-sycl-core-2025.3.3-30' "$RPM_LOG" | tail -n1 | cut -d: -f1)"
new_sycl_line="$(grep -n 'intel-oneapi-runtime-dpcpp-sycl-core-2026.1.1-325' "$RPM_LOG" | tail -n1 | cut -d: -f1)"
pin_mkl_line="$(grep -n 'intel-oneapi-runtime-mkl-2025.3.1-8' "$RPM_LOG" | tail -n1 | cut -d: -f1)"
new_mkl_line="$(grep -n 'intel-oneapi-runtime-mkl-2026.1.1-325' "$RPM_LOG" | tail -n1 | cut -d: -f1)"
if [[ -n "$pin_sycl_line" && -n "$new_sycl_line" && "$pin_sycl_line" -gt "$new_sycl_line" ]] \
   && [[ -n "$pin_mkl_line" && -n "$new_mkl_line" && "$pin_mkl_line" -gt "$new_mkl_line" ]]; then
  t_ok "BAS-57 extracts the pinned packages after any newer copies"
else
  t_bad "BAS-57 extracts the pinned packages after any newer copies"
fi
if [[ -e "$RUNTIME_DIR/opt/intel/oneapi/redist/lib/libsycl.so.8" ]]; then
  t_ok "BAS-57 pinned prefix carries libsycl.so.8"
else
  t_bad "BAS-57 pinned prefix carries libsycl.so.8"
fi
if grep -q 'intel-oneapi-umf-1.0' "$PIN_DNF_LOG"; then
  t_ok "BAS-57 requests libumf (intel-oneapi-umf-1.0)"
else
  t_bad "BAS-57 requests libumf (intel-oneapi-umf-1.0)"
fi
if grep -q -- '-n' "$LDCONF_LOG" 2>/dev/null; then
  t_ok "BAS-57 repairs SONAME symlinks with prefix ldconfig"
else
  t_bad "BAS-57 repairs SONAME symlinks with prefix ldconfig"
fi
unset -f dnf rpm2cpio cpio

# --- BAS-57: setup_runtime_env exposes UMF and IGC's LLVM libraries ---------
home="$TMP/home-env"
BONGO_HOME="$home"; RUNTIME_DIR="$home/runtime"
mkdir -p "$RUNTIME_DIR/opt/intel/oneapi/redist/lib" \
         "$RUNTIME_DIR/opt/intel/oneapi/umf/1.0/lib" \
         "$RUNTIME_DIR/usr/lib64/llvm15/lib" \
         "$RUNTIME_DIR/usr/lib64"
LD_LIBRARY_PATH=""
setup_runtime_env >/dev/null 2>&1
for probe_dir in "opt/intel/oneapi/umf/1.0/lib" "usr/lib64/llvm15/lib"; do
  case ":$LD_LIBRARY_PATH:" in
    *":$RUNTIME_DIR/$probe_dir:"*) t_ok "BAS-57 LD_LIBRARY_PATH includes $probe_dir" ;;
    *) t_bad "BAS-57 LD_LIBRARY_PATH includes $probe_dir" ;;
  esac
done
LD_LIBRARY_PATH=""

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

# --- M3.0a: prefix-cache serving flags -------------------------------------
reset_cache_state() {
  CACHE_PROMPT=1; CACHE_IDLE_SLOTS=""; CTX_CHECKPOINTS=""; SLOT_SAVE_PATH=""; SLOT_SAVE_PATH_SET=0; WARMUP=1
  TIER=iq2_xs; MODEL_SHARDS=(/m/model.gguf); N_GPU_LAYERS=99; N_CPU_MOE=16; CTX=131072
  N_CPU_MOE_SET=0; N_CPU_MOE_REQUESTED=""; PLACEMENT_FALLBACK=0; PLACEMENT_FALLBACK_REASON=""
  HOST=127.0.0.1; PORT=8080; PARALLEL=1; SELECTED_BACKEND=Vulkan; SERVER_BIN=""
  FLASH_ATTN=on; CACHE_TYPE_K=q8_0; CACHE_TYPE_V=q8_0; THREADS=""; LOAD_MODE=""; KEEP_ALIVE=1
  # M4.3 (BAS-158) shipped default: the M4.2-patched engine with the levers on.
  ENGINE_MODE=m42; M42_UPLOAD=1; ENGINE_PATCHED=1; SERVER_ENV=()
}
flags_have() { local needle="$1" f; for f in "${SERVER_FLAGS[@]}"; do [[ "$f" == "$needle" ]] && return 0; done; return 1; }
flags_pair() {
  local a="$1" b="$2" i
  for ((i = 0; i < ${#SERVER_FLAGS[@]} - 1; i++)); do
    [[ "${SERVER_FLAGS[i]}" == "$a" && "${SERVER_FLAGS[i + 1]}" == "$b" ]] && return 0
  done
  return 1
}

reset_cache_state
validate_args; build_server_flags
t_check "M3.0a default emits --cache-prompt" flags_have --cache-prompt
t_check "M3.0a default slot path lives under RUN_DIR" test "$SLOT_SAVE_PATH" = "$RUN_DIR/slots"

reset_cache_state; CACHE_PROMPT=0; SLOT_SAVE_PATH_SET=1; SLOT_SAVE_PATH=""
validate_args; build_server_flags
t_check "M3.0a --no-cache-prompt emits --no-cache-prompt" flags_have --no-cache-prompt
if flags_have --cache-prompt; then t_bad "M3.0a cold mode omits --cache-prompt"; else t_ok "M3.0a cold mode omits --cache-prompt"; fi
t_check "M3.0a --no-slot-save-path omits the flag" test -z "$SLOT_SAVE_PATH"

reset_cache_state; CACHE_IDLE_SLOTS=0; CTX_CHECKPOINTS=8
validate_args; build_server_flags
t_check "M3.0a emits --no-cache-idle-slots" flags_have --no-cache-idle-slots
t_check "M3.0a emits --ctx-checkpoints" flags_have --ctx-checkpoints
t_check "M3.0a emits --cache-prompt with idle/checkpoint overrides" flags_have --cache-prompt

# --- M4.3: the shipped default is the M4.2 engine + upload levers -----------
# M4.3 (BAS-158) turns the M4.2 win into the default: the patched Vulkan engine,
# --load-mode none, and the two GGML_VK upload env vars. '--engine stage0' is the
# no-rebuild opt-out that restores the Stage 0 Vulkan baseline (no --load-mode,
# no env). The --no-mmap spelling of --load-mode none (BAS-144) still works.
env_has() { local needle="$1" e; for e in "${SERVER_ENV[@]}"; do [[ "$e" == "$needle" ]] && return 0; done; return 1; }

reset_cache_state
validate_args; build_server_flags
t_check "M4.3 default emits --load-mode none" flags_pair --load-mode none
t_check "M4.3 default sets the device-local host buffer lever" env_has GGML_VK_HOST_BUFT_PER_DEVICE=1
t_check "M4.3 default sets the transfer-queue lever" env_has GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1

# The opt-out restores the Stage 0 Vulkan baseline without a rebuild: stock
# engine (or the patched one with the env-gated patch off), no --load-mode, no
# upload env. It also clears the upload levers whatever the argument order.
reset_cache_state; parse_args --engine stage0
validate_args; build_server_flags
t_check "M4.3 --engine stage0 selects the stock engine" test "$ENGINE_MODE" = stage0
t_check "M4.3 --engine stage0 forces the upload levers off" test "$M42_UPLOAD" = 0
if flags_have --load-mode; then t_bad "M4.3 --engine stage0 omits --load-mode"; else t_ok "M4.3 --engine stage0 omits --load-mode"; fi
if (( ${#SERVER_ENV[@]} == 0 )); then t_ok "M4.3 --engine stage0 omits the upload env"; else t_bad "M4.3 --engine stage0 omits the upload env"; fi

reset_cache_state; parse_args --no-m42-upload
validate_args; build_server_flags
if flags_have --load-mode; then t_bad "M4.3 --no-m42-upload omits --load-mode"; else t_ok "M4.3 --no-m42-upload omits --load-mode"; fi
if (( ${#SERVER_ENV[@]} == 0 )); then t_ok "M4.3 --no-m42-upload omits the upload env"; else t_bad "M4.3 --no-m42-upload omits the upload env"; fi

reset_cache_state; ENGINE_PATCHED=0
validate_args; build_server_flags
if flags_have --load-mode; then t_bad "M4.3 an unpatched engine omits --load-mode"; else t_ok "M4.3 an unpatched engine omits --load-mode"; fi
if (( ${#SERVER_ENV[@]} == 0 )); then t_ok "M4.3 an unpatched engine omits the upload env"; else t_bad "M4.3 an unpatched engine omits the upload env"; fi

reset_cache_state; SELECTED_BACKEND=SYCL0
validate_args; build_server_flags
if flags_have --load-mode; then t_bad "M4.3 the SYCL backend omits --load-mode"; else t_ok "M4.3 the SYCL backend omits --load-mode"; fi
if (( ${#SERVER_ENV[@]} == 0 )); then t_ok "M4.3 the SYCL backend omits the upload env"; else t_bad "M4.3 the SYCL backend omits the upload env"; fi

# --- M4.1: the host-expert RAM load mode stays selectable -------------------
# The pinned b11223 llama-server rejects --no-mmap; --load-mode none is the
# supported spelling of the same request (BAS-144).
reset_cache_state; parse_args --no-mmap
validate_args; build_server_flags
t_check "M4.1 --no-mmap maps to load mode none" test "$LOAD_MODE" = none
t_check "M4.1 --no-mmap emits --load-mode none" flags_pair --load-mode none

# An explicit --load-mode overrides the M4.2 default (mmap+mlock is not none).
reset_cache_state; parse_args --load-mode mmap+mlock
validate_args; build_server_flags
t_check "M4.1 --load-mode passes the mode through" flags_pair --load-mode mmap+mlock
if flags_pair --load-mode none; then t_bad "M4.3 explicit --load-mode is not overridden"; else t_ok "M4.3 explicit --load-mode is not overridden"; fi

reset_cache_state; LOAD_MODE=bogus
if (validate_args) >/dev/null 2>&1; then t_bad "M4.1 invalid load mode is rejected"; else t_ok "M4.1 invalid load mode is rejected"; fi

reset_cache_state; ENGINE_MODE=bogus; M42_UPLOAD=1
if (validate_args) >/dev/null 2>&1; then t_bad "M4.3 invalid engine mode is rejected"; else t_ok "M4.3 invalid engine mode is rejected"; fi

# --- M3.0: setup_runtime_env puts the IGC/LLVM libs on the link path -------
# Without usr/lib64/llvm15/lib the Level Zero probe aborts in gmm_helper and
# --backend sycl reports "no SYCL device was found" (BAS-72).
home="$TMP/home-llvm15"
RUNTIME_DIR="$home/runtime"
mkdir -p "$RUNTIME_DIR/opt/intel/oneapi/redist/lib" "$RUNTIME_DIR/usr/lib64/llvm15/lib"
LD_LIBRARY_PATH=""
setup_runtime_env >/dev/null 2>&1
if [[ ":$LD_LIBRARY_PATH:" == *":$RUNTIME_DIR/usr/lib64/llvm15/lib:"* ]]; then
  t_ok "M3.0 llvm15/lib is on LD_LIBRARY_PATH"
else
  t_bad "M3.0 llvm15/lib is on LD_LIBRARY_PATH"
fi
if [[ ":$LD_LIBRARY_PATH:" == *":$RUNTIME_DIR/opt/intel/oneapi/redist/lib:"* ]]; then
  t_ok "M3.0 oneAPI redist is still on LD_LIBRARY_PATH"
else
  t_bad "M3.0 oneAPI redist is still on LD_LIBRARY_PATH"
fi
# A prefix without the LLVM dir (older oneAPI set) must still work.
home="$TMP/home-no-llvm15"
RUNTIME_DIR="$home/runtime"
mkdir -p "$RUNTIME_DIR/usr/lib64"
LD_LIBRARY_PATH=""
setup_runtime_env >/dev/null 2>&1
if [[ -n "$LD_LIBRARY_PATH" ]]; then
  t_ok "M3.0 prefix without llvm15/lib still gets a library path"
else
  t_bad "M3.0 prefix without llvm15/lib still gets a library path"
fi

# --- M3.0: ZEL_LIBRARY_PATH stays a single directory -----------------------
# A colon-separated ZEL_LIBRARY_PATH makes the Level Zero driver enumerate zero
# devices, and setup_runtime_env() runs twice per bongo.sh invocation, so the
# old prepend produced "<prefix>/usr/lib64:<prefix>/usr/lib64" and the SYCL
# backend reported "no SYCL device was found" on a healthy Arc B70 (BAS-72).
RUNTIME_DIR="$TMP/home-zel/runtime"
mkdir -p "$RUNTIME_DIR/opt/intel/oneapi/redist/lib" "$RUNTIME_DIR/usr/lib64/llvm15/lib"
ZEL_LIBRARY_PATH=""
setup_runtime_env >/dev/null 2>&1
first="$ZEL_LIBRARY_PATH"
setup_runtime_env >/dev/null 2>&1   # second call, same invocation
if [[ "$ZEL_LIBRARY_PATH" == "$first" ]] && [[ "$ZEL_LIBRARY_PATH" != *:* ]]; then
  t_ok "M3.0 ZEL_LIBRARY_PATH is one dir and idempotent"
else
  t_bad "M3.0 ZEL_LIBRARY_PATH is one dir and idempotent (got '$ZEL_LIBRARY_PATH')"
fi
ZEL_LIBRARY_PATH="$RUNTIME_DIR/usr/lib64:$RUNTIME_DIR/usr/lib64"  # the old broken value
setup_runtime_env >/dev/null 2>&1
if [[ "$ZEL_LIBRARY_PATH" == "$RUNTIME_DIR/usr/lib64" ]]; then
  t_ok "M3.0 a multi-entry ZEL_LIBRARY_PATH is repaired"
else
  t_bad "M3.0 a multi-entry ZEL_LIBRARY_PATH is repaired (got '$ZEL_LIBRARY_PATH')"
fi


# --- M3.0: --backend auto prefers Vulkan (the measured default) -------------
# The M3.0 A/B (BAS-72) found SYCL slower at 128K on prefill, decode and the
# cached-turn TTFT, so `auto` resolves to Vulkan and keeps SYCL reachable both
# as its fallback and explicitly via --backend sycl.
mkdir -p "$TMP/bin"
# select_backend() calls these; stub them so the test needs no GPU.
setup_runtime_env() { :; }
DETECT_RESULT=""
detect_vulkan_device() { [[ -n "$DETECT_RESULT" ]] && printf '%s' "$DETECT_RESULT" || return 1; }
auto_select() { # <DETECT_RESULT value>
  DETECT_RESULT="$1"
  BACKEND="auto"; SERVER_BIN=""; SELECTED_BACKEND=""
  select_backend "$TMP/bin" >/dev/null 2>&1 || true
  printf '%s' "$SELECTED_BACKEND"
}
probe_sycl() { return 0; }
got="$(auto_select Vulkan1)"
if [[ "$got" == "Vulkan" ]]; then
  t_ok "M3.0 auto prefers Vulkan when the build sees an Intel GPU"
else
  t_bad "M3.0 auto prefers Vulkan when the build sees an Intel GPU (got '$got')"
fi
got="$(auto_select '')"
if [[ "$got" == "SYCL0" ]]; then
  t_ok "M3.0 auto falls back to SYCL when no Vulkan device is reported"
else
  t_bad "M3.0 auto falls back to SYCL when no Vulkan device is reported (got '$got')"
fi
probe_sycl() { return 1; }
got="$(auto_select '')"
if [[ "$got" == "Vulkan" ]]; then
  t_ok "M3.0 auto keeps Vulkan when neither device can be confirmed"
else
  t_bad "M3.0 auto keeps Vulkan when neither device can be confirmed (got '$got')"
fi
# --backend sycl must still win over a visible Vulkan device.
probe_sycl() { return 0; }
DETECT_RESULT="Vulkan1"
BACKEND="sycl"; SERVER_BIN=""; SELECTED_BACKEND=""
select_backend "$TMP/bin" >/dev/null 2>&1 || true
if [[ "$SELECTED_BACKEND" == "SYCL0" ]]; then
  t_ok "M3.0 --backend sycl still selects SYCL0 with a Vulkan device present"
else
  t_bad "M3.0 --backend sycl still selects SYCL0 with a Vulkan device present (got '$SELECTED_BACKEND')"
fi
# --backend vulkan also still wins over SYCL.
BACKEND="vulkan"; SERVER_BIN=""; SELECTED_BACKEND=""
select_backend "$TMP/bin" >/dev/null 2>&1 || true
if [[ "$SELECTED_BACKEND" == "Vulkan" ]]; then
  t_ok "M3.0 --backend vulkan still selects Vulkan"
else
  t_bad "M3.0 --backend vulkan still selects Vulkan (got '$SELECTED_BACKEND')"
fi

# --- M4.4: --placement expert residency policy ------------------------------
# 'tier' (default) keeps the fixed per-tier CPU/GPU split and is the 256K-safe
# value.  'auto' spends the VRAM a small context leaves free on expert residency.
# An explicit --n-cpu-moe wins over both.
m44_reset() {
  reset_cache_state
  N_CPU_MOE=""; PLACEMENT=tier
}

m44_reset
validate_args
if [[ "$N_CPU_MOE" == "16" ]]; then t_ok "M4.4 --placement tier keeps the per-tier split"; else t_bad "M4.4 --placement tier keeps the per-tier split (got '$N_CPU_MOE')"; fi

m44_reset; PLACEMENT=auto
validate_args
if [[ "$N_CPU_MOE" == "12" ]]; then t_ok "M4.4 --placement auto uses the measured 128K split"; else t_bad "M4.4 --placement auto uses the measured 128K split (got '$N_CPU_MOE')"; fi

# Above 131072 the KV cache is larger, so the context-aware value must keep more
# experts on the CPU.  M4.5 uses the measured large-context split (18 for
# iq2_xs), because the tier value 16 device-losts at 262144 on the reference box.
m44_reset; PLACEMENT=auto; CTX=262144
validate_args
if [[ "$N_CPU_MOE" == "18" ]]; then t_ok "M4.5 --placement auto uses the large-context split above 131072"; else t_bad "M4.5 --placement auto uses the large-context split above 131072 (got '$N_CPU_MOE')"; fi

# An explicit --n-cpu-moe overrides the policy in both modes.
m44_reset; PLACEMENT=auto; N_CPU_MOE=20
validate_args
if [[ "$N_CPU_MOE" == "20" ]]; then t_ok "M4.4 --n-cpu-moe overrides --placement auto"; else t_bad "M4.4 --n-cpu-moe overrides --placement auto (got '$N_CPU_MOE')"; fi

# An unmeasured tier stays on the safe per-tier split under auto.
m44_reset; PLACEMENT=auto; TIER=iq3_xxs
validate_args
if [[ "$N_CPU_MOE" == "22" ]]; then t_ok "M4.4 --placement auto falls back for an unmeasured tier"; else t_bad "M4.4 --placement auto falls back for an unmeasured tier (got '$N_CPU_MOE')"; fi

# A bad policy is refused instead of silently picking a split.
m44_reset; PLACEMENT=nonsense
if ( validate_args ) >/dev/null 2>&1; then
  t_bad "M4.4 --placement rejects an unknown mode"
else
  t_ok "M4.4 --placement rejects an unknown mode"
fi

# --- M4.5: auto is the default, with an OOM fallback to the tier split -------
# 'auto' becomes the shipped default (BAS-163) but must not sit on the VRAM load
# edge: a failed load at the auto split retries once at the tier split and keeps
# serving. An explicit --n-cpu-moe is never overridden.
m45_reset() {
  reset_cache_state
  N_CPU_MOE=""; PLACEMENT=auto
  N_CPU_MOE_SET=0; N_CPU_MOE_REQUESTED=""; PLACEMENT_FALLBACK=0; PLACEMENT_FALLBACK_REASON=""
  RUN_DIR="$TMP/m45-run"; mkdir -p "$RUN_DIR"
  LOG_FILE="$RUN_DIR/llama-server.log"; PID_FILE="$RUN_DIR/llama-server.pid"
}

if grep -qE '^PLACEMENT="auto"' "$SRC"; then
  t_ok "M4.5 shipped default is --placement auto"
else
  t_bad "M4.5 shipped default is --placement auto"
fi

# No flag at all resolves to the measured 128K split and records the request.
m45_reset; validate_args
if [[ "$N_CPU_MOE" == "12" && "$N_CPU_MOE_REQUESTED" == "12" ]]; then
  t_ok "M4.5 default uses the auto 128K split (12)"
else
  t_bad "M4.5 default uses the auto 128K split (got '$N_CPU_MOE' requested '$N_CPU_MOE_REQUESTED')"
fi

# The explicit opt-out still pins the tier split.
m45_reset; PLACEMENT=tier; validate_args
if [[ "$N_CPU_MOE" == "16" ]]; then
  t_ok "M4.5 --placement tier opts out to the tier split (16)"
else
  t_bad "M4.5 --placement tier opts out to the tier split (got '$N_CPU_MOE')"
fi

# A forced OOM at the auto split retries once at 16 and serves.
m45_reset; validate_args
: > "$LOG_FILE"; printf 'vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n' >> "$LOG_FILE"
server_attempts=0
start_server() { server_attempts=$((server_attempts + 1)); return $(( server_attempts < 2 ? 1 : 0 )); }
if start_server_guarded >"$TMP/m45-plan.out" 2>/dev/null; then
  t_ok "M4.5 auto fallback serves on the retry"
else
  t_bad "M4.5 auto fallback serves on the retry"
fi
if grep -q 'auto -> tier' "$TMP/m45-plan.out"; then
  t_ok "M4.5 fallback is shown in the plan output"
else
  t_bad "M4.5 fallback is shown in the plan output"
fi
if [[ "$server_attempts" == "2" ]]; then
  t_ok "M4.5 auto fallback retries exactly once"
else
  t_bad "M4.5 auto fallback retries exactly once (got $server_attempts)"
fi
if [[ "$N_CPU_MOE" == "16" && "$PLACEMENT_FALLBACK" == "1" ]]; then
  t_ok "M4.5 fallback switches the active split to the tier value"
else
  t_bad "M4.5 fallback switches the active split to the tier value (got '$N_CPU_MOE' fallback '$PLACEMENT_FALLBACK')"
fi
if [[ "$PLACEMENT_FALLBACK_REASON" == "out_of_device_memory" ]]; then
  t_ok "M4.5 fallback classifies the OOM reason"
else
  t_bad "M4.5 fallback classifies the OOM reason (got '$PLACEMENT_FALLBACK_REASON')"
fi
if grep -q '"auto_fallback": true' "$RUN_DIR/bongo-config.json" \
   && grep -q '"n_cpu_moe_requested": 12' "$RUN_DIR/bongo-config.json" \
   && grep -q '"n_cpu_moe": 16' "$RUN_DIR/bongo-config.json"; then
  t_ok "M4.5 fallback is recorded in bongo-config.json"
else
  t_bad "M4.5 fallback is recorded in bongo-config.json"
fi
if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$RUN_DIR/bongo-config.json" 2>/dev/null; then
  t_ok "M4.5 fallback config is valid JSON"
else
  t_bad "M4.5 fallback config is valid JSON"
fi
if [[ -f "$RUN_DIR/llama-server-auto-12.log" ]]; then
  t_ok "M4.5 keeps the failed auto-placement log"
else
  t_bad "M4.5 keeps the failed auto-placement log"
fi

# An explicit --n-cpu-moe is never overridden by the fallback.
m45_reset; N_CPU_MOE=12; N_CPU_MOE_SET=1; validate_args
server_attempts=0
start_server() { server_attempts=$((server_attempts + 1)); return 1; }
if start_server_guarded; then
  t_bad "M4.5 explicit --n-cpu-moe failure is not retried"
else
  t_ok "M4.5 explicit --n-cpu-moe failure is not retried"
fi
if [[ "$server_attempts" == "1" && "$N_CPU_MOE" == "12" ]]; then
  t_ok "M4.5 explicit --n-cpu-moe is preserved"
else
  t_bad "M4.5 explicit --n-cpu-moe is preserved (attempts $server_attempts, n_cpu_moe '$N_CPU_MOE')"
fi

# Above 131072 auto already uses the measured large-context safe split, so
# there is nothing (safer) to fall back to.
m45_reset; CTX=262144; validate_args
if [[ "$N_CPU_MOE" == "18" ]] && ! placement_fallback_eligible; then
  t_ok "M4.5 256K auto is already at the safe split (no fallback target)"
else
  t_bad "M4.5 256K auto is already at the safe split (got '$N_CPU_MOE')"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
(( FAIL == 0 ))
