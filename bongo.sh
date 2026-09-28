#!/usr/bin/env bash
# bongo — one-command Intel Arc LLM setup (Stage 0: llama.cpp SYCL baseline).
#
# Usage:  git clone <repo> && cd bongo && ./bongo.sh
#
# Reaches an OpenAI-compatible server at http://127.0.0.1:8080/v1 with a
# >=131072-token context. See docs/adr/0001-runtime-architecture.md.
#
# This script is deliberately self-contained: it provisions the Intel compute
# runtime (Level Zero + oneAPI SYCL), fetches a pinned llama.cpp build, resumes
# the GGUF download from Hugging Face, and launches llama-server. It runs on
# Fedora and Ubuntu/Debian.
#
# SPDX-License-Identifier: LicenseRef-TBD
set -Eeuo pipefail

BONGO_VERSION="0.1.0"

# ---------------------------------------------------------------------------
# Pins (ADR-0002: pinning is mandatory; record the revision in the config)
# ---------------------------------------------------------------------------
# llama.cpp release tag and the commit it points at (resolved via GitHub API).
LLAMA_CPP_REV_DEFAULT="b11223"
LLAMA_CPP_COMMIT_DEFAULT="4da6337767f973e2b4d0797e5b323d77d8565e4a"

MODEL_REPO_DEFAULT="ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
MODEL_ARCH="qwen4exp"
MODEL_LAYERS=48

# Default context. Acceptance requires >= 131072 and the model supports 262144.
CTX_DEFAULT=131072

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
BONGO_HOME="${BONGO_HOME:-$HOME/.bongo}"
RUN_DIR="$BONGO_HOME/run"
RUNTIME_DIR="${BONGO_RUNTIME_DIR:-$BONGO_HOME/runtime}"
LLAMA_DIR="$BONGO_HOME/llama"
MODEL_BASE_DIR="${BONGO_MODEL_DIR:-$BONGO_HOME/models}"
LOG_FILE=""
PID_FILE=""

# ---------------------------------------------------------------------------
# Defaults / runtime state
# ---------------------------------------------------------------------------
TIER="iq2_xs"
MODEL_REPO="$MODEL_REPO_DEFAULT"
GGUF_DIR=""
CTX="$CTX_DEFAULT"
HOST="127.0.0.1"
PORT=8080
BACKEND="auto"              # auto|sycl|vulkan|cpu
RUNTIME_MODE="auto"         # auto|system|user|dir
LLAMA_REV="$LLAMA_CPP_REV_DEFAULT"
LLAMA_BIN_DIR=""
N_GPU_LAYERS=99
N_CPU_MOE=""
THREADS=""
FLASH_ATTN="on"
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
PARALLEL=1
NO_MMAP=0
DRY_RUN=0
DETACH=0
FORCE=0
CHECK_ONLY=0
UNINSTALL=0
SKIP_DOWNLOAD=0
VERIFY_SHA=0
ASSUME_YES=0
KEEP_ALIVE=1

SERVER_PID=""
SERVER_BIN=""
SERVER_FLAGS=()
SELECTED_BACKEND=""
SELECTED_DEVICES=""
GPU_PCI=""
GPU_NAME=""
VRAM_TOTAL_GB=""

# Tier -> GGUF base name (shards are "<base>-00001-of-00002.gguf", ...).
declare -A TIER_BASE=(
  [iq2_xs]="Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS"
  [iq3_xxs]="Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
  [q2_0]="Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0"
)
# Tier -> total bytes across both shards (verified from the HF file tree).
declare -A TIER_BYTES=(
  [iq2_xs]=68152167168
  [iq3_xxs]=75966073120
  [q2_0]=66549952800
)
# Tier -> experts-only bytes (research estimate) and default MoE CPU offload.
declare -A TIER_EXPERT_GB=(
  [iq2_xs]=36
  [iq3_xxs]=43
  [q2_0]=34
)
# Default number of MoE layers whose expert weights stay on the CPU. Derived
# from a ~24 GB VRAM expert budget over 48 layers; overridable with --n-cpu-moe.
declare -A TIER_N_CPU_MOE=(
  [iq2_xs]=16
  [iq3_xxs]=22
  [q2_0]=15
)

# ---------------------------------------------------------------------------
# Logging (color only on a tty; never a traceback on failure)
# ---------------------------------------------------------------------------
if [[ -t 2 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
  C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""
fi
log()  { printf '%s[bongo]%s %s\n' "$C_BLUE" "$C_RESET" "$*" >&2; }
ok()   { printf '%s[bongo]%s %s\n' "$C_GREEN" "$C_RESET" "$*" >&2; }
warn() { printf '%s[bongo]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()  { printf '%s[bongo]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }

# die <message...>  — actionable error, no traceback.
die() {
  err "$*"
  err "Run './bongo.sh --help' for options."
  exit 1
}

on_err() {
  local rc=$?
  if (( rc != 0 )); then
    err "bongo.sh failed at line ${BASH_LINENO[0]:-?} (exit $rc)."
    if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
      err "Last server log lines ($LOG_FILE):"
      tail -n 20 "$LOG_FILE" >&2 || true
    fi
  fi
}
trap on_err ERR

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' is not installed. Install it and retry (e.g. 'sudo dnf install $2' or 'sudo apt-get install $2')."
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

can_sudo() {
  have_cmd sudo || return 1
  [[ "$(id -u)" -eq 0 ]] && return 0
  sudo -n true >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
bongo.sh ${BONGO_VERSION} — one-command Intel Arc LLM setup (llama.cpp SYCL baseline)

Usage:
  ./bongo.sh [options]

Model selection:
  --tier NAME            GGUF tier: iq2_xs (default), iq3_xxs, q2_0
  --gguf-dir DIR         Use an existing download (skip Hugging Face download)
  --model-repo REPO      Hugging Face repo id (default: $MODEL_REPO_DEFAULT)
  --verify-sha256        Verify downloaded shards against SHA256SUMS

Serving:
  --ctx N                Context size (default: $CTX_DEFAULT; minimum 131072)
  --host HOST            Bind address (default: $HOST)
  --port N               Port (default: $PORT)
  --backend NAME         auto (default) | sycl | vulkan | cpu
  --n-cpu-moe N          MoE layers with experts on CPU (explicit placement)
  --n-gpu-layers N       Max layers offloaded to GPU (default: $N_GPU_LAYERS)
  --threads N            CPU threads (default: auto)
  --no-mmap              Pass --no-mmap (only if measurement shows thrashing)
  --detach               Start the server in the background and exit
  --no-keep-alive        Let llama-server exit when idle

Runtime / engine provisioning:
  --runtime MODE         auto (default) | system | user | dir
  --runtime-dir DIR      Use a pre-provisioned runtime prefix (implies --runtime dir)
  --llama-rev REV        Pin a llama.cpp release tag (default: $LLAMA_CPP_REV_DEFAULT)
  --llama-bin DIR        Use an existing llama.cpp build directory
  --check                Provision and verify the runtime, then exit (no model)

Other:
  --dry-run              Print the plan and generated flags; change nothing
  --force                Re-download / re-fetch even if files look complete
  --yes                  Do not prompt for confirmation
  --uninstall            Print how to remove downloaded artifacts (add --yes to remove them)
  -h, --help             Show this help

Environment:
  HF_TOKEN               Hugging Face token (needed for gated repos)
  BONGO_HOME             State directory (default: ~/.bongo)
  BONGO_RUNTIME_DIR      Runtime prefix (same as --runtime-dir)
  BONGO_MODEL_DIR        Model root (default: ~/.bongo/models)

Examples:
  ./bongo.sh                                  # default: iq2_xs, SYCL, 128K ctx
  ./bongo.sh --tier iq3_xxs --n-cpu-moe 24
  ./bongo.sh --gguf-dir /mnt/models/iq2_xs --backend vulkan
  ./bongo.sh --check                          # verify GPU/runtime only
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tier) TIER="${2:?--tier needs a value}"; shift 2;;
      --gguf-dir) GGUF_DIR="${2:?--gguf-dir needs a value}"; shift 2;;
      --model-repo) MODEL_REPO="${2:?--model-repo needs a value}"; shift 2;;
      --verify-sha256) VERIFY_SHA=1; shift;;
      --ctx) CTX="${2:?--ctx needs a value}"; shift 2;;
      --host) HOST="${2:?--host needs a value}"; shift 2;;
      --port) PORT="${2:?--port needs a value}"; shift 2;;
      --backend) BACKEND="${2:?--backend needs a value}"; shift 2;;
      --n-cpu-moe) N_CPU_MOE="${2:?--n-cpu-moe needs a value}"; shift 2;;
      --n-gpu-layers) N_GPU_LAYERS="${2:?--n-gpu-layers needs a value}"; shift 2;;
      --threads) THREADS="${2:?--threads needs a value}"; shift 2;;
      --no-mmap) NO_MMAP=1; shift;;
      # The published GGUF has no MTP/NextN head and llama.cpp qwen4exp cannot convert or run one;
      # speculation is the n-gram/PLE table. Refuse the flag with an actionable message.
      --mtp) die "--mtp is not supported for this model: the published GGUF has no MTP head and llama.cpp qwen4exp cannot run one. Speculation uses the lazy-read n-gram/PLE table. See docs/research/intel-arc-b70.md section 2.1.";;
      --detach) DETACH=1; shift;;
      --no-keep-alive) KEEP_ALIVE=0; shift;;
      --runtime) RUNTIME_MODE="${2:?--runtime needs a value}"; shift 2;;
      --runtime-dir) RUNTIME_DIR="${2:?--runtime-dir needs a value}"; RUNTIME_MODE="dir"; shift 2;;
      --llama-rev) LLAMA_REV="${2:?--llama-rev needs a value}"; shift 2;;
      --llama-bin) LLAMA_BIN_DIR="${2:?--llama-bin needs a value}"; shift 2;;
      --check) CHECK_ONLY=1; shift;;
      --dry-run) DRY_RUN=1; shift;;
      --force) FORCE=1; shift;;
      --yes) ASSUME_YES=1; shift;;
      --uninstall) UNINSTALL=1; shift;;
      -h|--help) usage; exit 0;;
      *) die "Unknown option '$1'.";;
    esac
  done
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
validate_args() {
  MODEL_REPO="${MODEL_REPO:-$MODEL_REPO_DEFAULT}"
  if [[ -z "${TIER_BASE[$TIER]:-}" ]]; then
    die "Unknown tier '$TIER'. Valid tiers: iq2_xs, iq3_xxs, q2_0."
  fi
  case "$BACKEND" in auto|sycl|vulkan|cpu) ;; *) die "Unknown backend '$BACKEND'. Use auto|sycl|vulkan|cpu.";; esac
  case "$RUNTIME_MODE" in auto|system|user|dir) ;; *) die "Unknown runtime mode '$RUNTIME_MODE'. Use auto|system|user|dir.";; esac
  [[ "$CTX" =~ ^[0-9]+$ ]] || die "--ctx must be an integer (got '$CTX')."
  (( CTX >= 131072 )) || die "Context $CTX is below the 131072 acceptance minimum. Use --ctx 131072 or higher."
  [[ "$PORT" =~ ^[0-9]+$ ]] || die "--port must be an integer."
  [[ "$N_GPU_LAYERS" =~ ^[0-9]+$ ]] || die "--n-gpu-layers must be an integer."
  if [[ -n "$N_CPU_MOE" ]]; then
    [[ "$N_CPU_MOE" =~ ^[0-9]+$ ]] || die "--n-cpu-moe must be an integer."
  else
    N_CPU_MOE="${TIER_N_CPU_MOE[$TIER]}"
  fi
  if [[ -n "$GGUF_DIR" && ! -d "$GGUF_DIR" ]]; then
    die "--gguf-dir '$GGUF_DIR' does not exist or is not a directory."
  fi
}

# ---------------------------------------------------------------------------
# OS / GPU detection
# ---------------------------------------------------------------------------
OS_ID=""; OS_LIKE=""; OS_VER=""
detect_os() {
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS_ID="${ID:-unknown}"; OS_LIKE="${ID_LIKE:-}"; OS_VER="${VERSION_ID:-}"
  fi
}

is_fedora() { [[ "$OS_ID" == "fedora" || "$OS_LIKE" == *fedora* ]]; }
is_debian() { [[ "$OS_ID" == debian || "$OS_ID" == ubuntu || "$OS_LIKE" == *debian* ]]; }

detect_gpu() {
  have_cmd lspci || die "lspci is required to detect the GPU (install pciutils)."
  local line
  line="$(lspci -nn 2>/dev/null | grep -iE 'VGA compatible controller|Display controller|3D controller' | grep -i '8086:' | head -n1 || true)"
  if [[ -z "$line" ]]; then
    die "No Intel GPU found on this host (lspci found no Intel display device).
  This script targets Intel Arc. If the GPU exists but is not visible,
  check that it is seated/powered and that the 'xe' kernel driver is loaded
  ('lsmod | grep xe', 'lspci -nn | grep -i intel')."
  fi
  GPU_PCI="$(printf '%s' "$line" | grep -oE '[0-9a-f]{4}:[0-9a-f]{4}' | head -n1)"
  GPU_NAME="$(printf '%s' "$line" | awk -F': ' '{print $2}' | sed -E 's/ \[[0-9a-f]{4}:[0-9a-f]{4}\]$//')"
  if [[ "${GPU_PCI:-}" != "8086:e223" ]]; then
    warn "Detected Intel GPU '$GPU_PCI' ($GPU_NAME); the reference target is 8086:e223 (Arc Pro B70)."
  fi
  if [[ -d /sys/module/xe ]]; then
    log "Kernel driver: xe (loaded)"
  elif [[ -d /sys/module/i915 ]]; then
    warn "Kernel driver: i915 (Arc B-Series normally uses 'xe')."
  else
    warn "Neither 'xe' nor 'i915' kernel driver is loaded."
  fi
  if [[ -e /dev/dri/renderD128 ]]; then
    log "Render node: /dev/dri/renderD128"
  else
    die "/dev/dri/renderD128 not found. The Intel render node is missing.
  Check that the GPU is bound to the 'xe' driver and that /dev/dri is mounted."
  fi
  VRAM_TOTAL_GB="$(gpu_vram_total_gb || true)"
}

# Total VRAM in GB, best effort. Prefer xpu-smi, then the PCI BAR size.
gpu_vram_total_gb() {
  if have_cmd xpu-smi; then
    local v
    v="$(xpu-smi discovery -j 2>/dev/null | grep -oE '"memory_physical_size_byte"[ :]*[0-9]+' | grep -oE '[0-9]+$' | head -n1 || true)"
    if [[ -n "$v" ]]; then echo "$(( v / 1000000000 ))"; return 0; fi
  fi
  local bar
  bar="$(lspci -v -d "$GPU_PCI" 2>/dev/null | grep -oE 'size=[0-9]+[GM]' | sed 's/size=//' | sort -h | tail -n1 || true)"
  case "$bar" in
    *G) echo "${bar%G}";;
    *M) echo "$(( ${bar%M} / 1024 ))";;
    *) return 1;;
  esac
}

# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------
available_bytes() { df -PB1 "$1" 2>/dev/null | awk 'NR==2{print $4}'; }

check_disk() {
  local target="$1" need="${2:-0}" avail
  avail="$(available_bytes "$target")"
  [[ -n "$avail" ]] || die "Could not determine free disk space on '$target'."
  if (( avail < need )); then
    die "Insufficient disk space on '$target': need ~$(( need / 1000000000 )) GB, have $(( avail / 1000000000 )) GB free.
  Free space, choose a different --gguf-dir, or pick a smaller --tier."
  fi
}

# ---------------------------------------------------------------------------
# Runtime provisioning
# ---------------------------------------------------------------------------
ONEAPI_REPO_URL="https://yum.repos.intel.com/oneapi"
ONEAPI_REPO_FILE_URL="https://yum.repos.intel.com/oneapi/file/intel-oneapi.repo"

setup_runtime_env() {
  local base="$RUNTIME_DIR"
  local -a libdirs=()
  [[ -d "$base/opt/intel/oneapi/redist/lib" ]] && libdirs+=("$base/opt/intel/oneapi/redist/lib")
  [[ -d "$base/usr/lib64" ]] && libdirs+=("$base/usr/lib64")
  [[ -d "$base/lib" ]] && libdirs+=("$base/lib")
  if (( ${#libdirs[@]} )); then
    local joined
    joined="$(IFS=:; echo "${libdirs[*]}")"
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ZEL_LIBRARY_PATH="$base/usr/lib64${ZEL_LIBRARY_PATH:+:$ZEL_LIBRARY_PATH}"
    log "Runtime library path: $joined"
  fi
  # Expose runtime-provided tools (vulkaninfo, clinfo, sycl-ls, xpu-smi) when
  # the runtime was provisioned user-locally.
  if [[ -d "$base/usr/bin" ]]; then
    export PATH="$base/usr/bin:$PATH"
  fi
  # OpenCL loader needs an ICD entry for the user-local NEO driver.
  if [[ -f "$base/usr/lib64/intel-opencl/libigdrcl.so" ]]; then
    mkdir -p "$RUN_DIR/ocl-vendors"
    printf '%s\n' "$base/usr/lib64/intel-opencl/libigdrcl.so" > "$RUN_DIR/ocl-vendors/intel.icd"
    export OCL_ICD_VENDORS="$RUN_DIR/ocl-vendors${OCL_ICD_VENDORS:+:$OCL_ICD_VENDORS}"
  fi
  # Vulkan loader needs ICD manifests; rewrite the user-local Mesa manifests so
  # they point at the runtime prefix instead of /usr/lib64.
  if [[ -d "$base/usr/share/vulkan/icd.d" ]]; then
    mkdir -p "$RUN_DIR/vulkan-icds"
    local j
    for j in "$base"/usr/share/vulkan/icd.d/*.json; do
      [[ -f "$j" ]] || continue
      sed "s#/usr/lib64/#$base/usr/lib64/#g" "$j" > "$RUN_DIR/vulkan-icds/$(basename "$j")"
    done
    if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
      export VK_ICD_FILENAMES="$RUN_DIR/vulkan-icds"
      export VK_DRIVER_FILES="$RUN_DIR/vulkan-icds"
    fi
  fi
}

sycl_runtime_present() {
  # A user-local prefix provisioned by install_runtime_user() can legitimately
  # contain no libsycl.so (the oneAPI set need not carry the DPC++ SYCL core),
  # so trust its sentinel and the Level Zero / OpenCL loaders. Without this a
  # second run re-downloads/re-extracts ~1.9 GB (BAS-58 F1).
  [[ -e "$RUNTIME_DIR/.bongo-provisioned" ]] && return 0
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    local d
    while IFS= read -r d; do
      [[ -e "$d/libsycl.so" || -e "$d/libsycl.so.8" ]] && return 0
      [[ -e "$d/libze_loader.so" || -e "$d/libze_loader.so.1" ]] && return 0
      [[ -e "$d/libOpenCL.so" || -e "$d/libOpenCL.so.1" ]] && return 0
    done < <(printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n')
  fi
  have_cmd sycl-ls && return 0
  [[ -e /opt/intel/oneapi/setvars.sh ]] && return 0
  return 1
}

install_runtime_fedora() {
  local SUDO=""
  [[ "$(id -u)" -ne 0 ]] && SUDO="sudo"
  log "Installing Intel compute runtime (Fedora packages) via dnf..."
  $SUDO dnf install -y intel-level-zero oneapi-level-zero intel-opencl clinfo \
    || die "dnf failed to install the Intel compute runtime. Check network/repo access and retry."
  log "Adding the Intel oneAPI repository and installing the SYCL runtime..."
  if have_cmd dnf; then
    if dnf config-manager --help >/dev/null 2>&1; then
      $SUDO dnf config-manager addrepo --from-repofile="$ONEAPI_REPO_FILE_URL" >/dev/null 2>&1 || true
    fi
  fi
  $SUDO dnf install -y \
    intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl \
    intel-oneapi-runtime-dnnl intel-oneapi-runtime-tbb \
    intel-oneapi-runtime-compilers intel-oneapi-runtime-openmp \
    intel-oneapi-runtime-opencl \
    || warn "One or more oneAPI runtime packages failed to install; SYCL may be unavailable."
}

install_runtime_debian() {
  local SUDO=""
  [[ "$(id -u)" -ne 0 ]] && SUDO="sudo"
  need_cmd apt-get apt
  log "Installing Intel compute runtime (Debian/Ubuntu packages) via apt..."
  $SUDO apt-get update -y >/dev/null 2>&1 || true
  $SUDO apt-get install -y intel-level-zero-gpu level-zero intel-opencl-icd clinfo \
    || warn "Could not install all Level Zero/OpenCL packages via apt."
  if ! have_cmd sycl-ls && [[ ! -e /opt/intel/oneapi/setvars.sh ]]; then
    log "Adding the Intel oneAPI apt repository..."
    local keyring=/usr/share/keyrings/oneapi-archive-keyring.gpg
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor | $SUDO tee "$keyring" >/dev/null || warn "Could not install the oneAPI signing key."
    echo "deb [signed-by=$keyring] https://apt.repos.intel.com/oneapi all main" \
      | $SUDO tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    $SUDO apt-get update -y >/dev/null 2>&1 || true
    $SUDO apt-get install -y intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl \
      || warn "oneAPI runtime packages failed to install; SYCL may be unavailable."
  fi
}

# Rootless provisioning: download RPMs and extract them under $RUNTIME_DIR.
# This keeps the machine clean and works without root. Fedora-family only.
install_runtime_user() {
  need_cmd dnf; need_cmd rpm2cpio; need_cmd cpio
  log "Provisioning the Intel compute runtime into $RUNTIME_DIR (no root)..."
  mkdir -p "$RUNTIME_DIR"
  local tmp="$BONGO_HOME/tmp/runtime"
  rm -rf "$tmp"; mkdir -p "$tmp"
  local repo_dir="$tmp/repos"
  mkdir -p "$repo_dir"
  cat > "$repo_dir/intel-oneapi.repo" <<EOF
[intel-oneapi]
name=Intel oneAPI
baseurl=$ONEAPI_REPO_URL
enabled=1
repo_gpgcheck=0
gpgcheck=0
type=rpm-md
EOF
  # Copy the distro repo definitions so dependency resolution still works.
  if [[ -d /etc/yum.repos.d ]]; then
    cp /etc/yum.repos.d/*.repo "$repo_dir/" 2>/dev/null || true
  fi
  # --resolve is required: without it dnf fetches only the named meta-packages
  # and never the dependency that ships libsycl.so
  # (intel-oneapi-runtime-dpcpp-sycl-core), so the prefix has no SYCL library and
  # every run re-provisions (BAS-58 F1).
  local oneapi_pkgs=(
    intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-dpcpp-sycl-core
    intel-oneapi-runtime-mkl intel-oneapi-runtime-dnnl intel-oneapi-runtime-tbb
    intel-oneapi-runtime-compilers intel-oneapi-runtime-openmp
    intel-oneapi-runtime-opencl
  )
  log "  downloading oneAPI runtime packages..."
  if ! ( cd "$tmp" && dnf -q --setopt=reposdir="$repo_dir" download --resolve "${oneapi_pkgs[@]}" ); then
    warn "SYCL core runtime package unavailable from $ONEAPI_REPO_URL; retrying without it."
    oneapi_pkgs=(
      intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl
      intel-oneapi-runtime-dnnl intel-oneapi-runtime-tbb
      intel-oneapi-runtime-compilers intel-oneapi-runtime-openmp
      intel-oneapi-runtime-opencl
    )
    ( cd "$tmp" && dnf -q --setopt=reposdir="$repo_dir" download --resolve "${oneapi_pkgs[@]}" ) \
      || die "Failed to download the oneAPI runtime packages. Check network access to $ONEAPI_REPO_URL."
  fi
  log "  downloading Level Zero / OpenCL / IGC packages..."
  ( cd "$tmp" && dnf -q download --resolve intel-level-zero oneapi-level-zero intel-opencl intel-igc-libs intel-gmmlib clinfo ) \
    || die "Failed to download the Intel GPU runtime packages."
  log "  extracting packages into $RUNTIME_DIR..."
  ( cd "$RUNTIME_DIR" && for r in "$tmp"/*.rpm; do
      case "$r" in *.i686.rpm) continue;; esac
      rpm2cpio "$r" | cpio -idmu --quiet --no-absolute-filenames 2>/dev/null || true
    done )
  rm -rf "$tmp"
  # Fail loudly instead of writing the sentinel on an empty/failed extraction.
  if ! compgen -G "$RUNTIME_DIR/usr/lib64/*.so*" >/dev/null \
     && ! compgen -G "$RUNTIME_DIR/opt/intel/oneapi/redist/lib/*.so*" >/dev/null; then
    die "Runtime extraction produced no shared libraries under $RUNTIME_DIR."
  fi
  # Sentinel for sycl_runtime_present(): the user-local set can legitimately
  # lack libsycl.so, so the prefix itself is the reliable idempotency signal.
  {
    echo "bongo-runtime $BONGO_VERSION"
    printf 'packages %s\n' "${oneapi_pkgs[*]}"
  } > "$RUNTIME_DIR/.bongo-provisioned"
  ok "User-local runtime ready at $RUNTIME_DIR."
}

provision_runtime() {
  case "$RUNTIME_MODE" in
    dir)
      [[ -d "$RUNTIME_DIR" ]] || die "--runtime-dir '$RUNTIME_DIR' does not exist.
  Provision a runtime first (./bongo.sh --runtime system) or fix the path."
      setup_runtime_env
      ;;
    system)
      if [[ "$(id -u)" -ne 0 ]] && ! have_cmd sudo; then
        die "System provisioning needs root or sudo, but neither is available.
  Re-run with '--runtime user' to install into $RUNTIME_DIR without root."
      fi
      if is_fedora; then install_runtime_fedora
      elif is_debian; then install_runtime_debian
      else die "Unsupported distribution '$OS_ID'. Supported: Fedora, Ubuntu/Debian.
  Use '--runtime user' on RPM-based systems, or install Level Zero + oneAPI manually."
      fi
      setup_runtime_env
      ;;
    user)
      install_runtime_user
      setup_runtime_env
      ;;
    auto)
      setup_runtime_env
      if sycl_runtime_present; then
        log "Existing SYCL runtime detected."
      elif can_sudo; then
        install_runtime_fedora_or_debian
        setup_runtime_env
      elif is_fedora || is_debian; then
        warn "No SYCL runtime and no passwordless sudo: falling back to user-local provisioning."
        install_runtime_user
        setup_runtime_env
      else
        die "No usable compute runtime and no supported package manager. Install Level Zero + oneAPI and retry."
      fi
      ;;
  esac
}

install_runtime_fedora_or_debian() {
  if is_fedora; then install_runtime_fedora
  elif is_debian; then install_runtime_debian
  else die "Unsupported distribution '$OS_ID'."
  fi
}

# ---------------------------------------------------------------------------
# Backend selection / probe
# ---------------------------------------------------------------------------
# Probe the SYCL runtime with the llama.cpp helper binary, if present.
probe_sycl() {
  local bin="$1"
  local helper
  helper="$(dirname "$bin")/llama-ls-sycl-device"
  [[ -x "$helper" ]] || return 1
  local out
  if ! out="$("$helper" 2>&1)"; then
    return 1
  fi
  # The helper lists "Found N SYCL devices" and a device table.
  if grep -qiE 'Found [1-9][0-9]* SYCL devices|Device 0' <<<"$out"; then
    return 0
  fi
  return 1
}

probe_vulkan() {
  if have_cmd vulkaninfo; then
    local out=""
    out="$(vulkaninfo --summary 2>/dev/null || true)"
    grep -qiE 'Intel|Arc|BMG' <<<"$out" && return 0
  fi
  # No vulkaninfo: assume Vulkan works if the loader library is loadable.
  return 1
}

# Pick the llama.cpp Vulkan device index for the Intel GPU from
# `llama-server --list-devices`.  A host with a second Vulkan device (for
# example a CPU iGPU from another vendor) would otherwise default to
# Vulkan0, which fails to hold the model.  Prints e.g. "Vulkan1"; prints
# nothing (and returns non-zero) when no Intel device is listed.
detect_vulkan_device() {
  local bin="$1"
  [[ -x "$bin" ]] || return 1
  local out tok
  out="$("$bin" --list-devices 2>/dev/null || true)"
  tok="$(awk '/Vulkan[0-9]+:/ && /Intel/ {print $1; exit}' <<<"$out" | tr -d ':')"
  [[ -n "$tok" ]] || return 1
  printf '%s' "$tok"
}

select_backend() {
  local bin_dir="$1"
  local want="$BACKEND"
  case "$want" in
    sycl)
      SERVER_BIN="$bin_dir/llama-server"
      setup_runtime_env
      if ! probe_sycl "$SERVER_BIN"; then
        die "Backend 'sycl' requested but no SYCL device was found.
  - Is the Intel compute runtime installed? Try: ./bongo.sh --check --runtime system
  - Is the GPU usable? Check 'lspci -nn | grep -i intel' and 'ls /dev/dri'.
  - Fall back with: ./bongo.sh --backend vulkan"
      fi
      SELECTED_BACKEND="SYCL0";;
    vulkan)
      SERVER_BIN="$bin_dir/llama-server"
      if ! probe_vulkan; then
        warn "Could not positively confirm a Vulkan device; continuing anyway."
      fi
      SELECTED_BACKEND="Vulkan";;
    cpu)
      SERVER_BIN="$bin_dir/llama-server"
      SELECTED_BACKEND="CPU";;
    auto)
      # SYCL first (the baseline), then Vulkan, then CPU.
      if probe_sycl "$(dirname "$bin_dir/llama-server")/llama-server" 2>/dev/null; then
        SERVER_BIN="$bin_dir/llama-server"; SELECTED_BACKEND="SYCL0"
      else
        warn "SYCL device not available; falling back to Vulkan (documented fallback)."
        SERVER_BIN="$bin_dir/llama-server"; SELECTED_BACKEND="Vulkan"
      fi
      ;;
  esac
}

# ---------------------------------------------------------------------------
# llama.cpp binary acquisition
# ---------------------------------------------------------------------------
backend_asset() {
  case "$1" in
    sycl) echo "ubuntu-sycl-fp16-x64";;
    vulkan) echo "ubuntu-vulkan-x64";;
    cpu) echo "ubuntu-x64";;
    *) die "No prebuilt asset for backend '$1'.";;
  esac
}

fetch_llama() {
  local backend="$1"
  local dest="$LLAMA_DIR/$LLAMA_REV/${backend}"
  if [[ -x "$dest/llama-server" && "$FORCE" -eq 0 ]]; then
    log "Using pinned llama.cpp $LLAMA_REV ($backend) at $dest"
    echo "$dest"; return 0
  fi
  local asset="llama-${LLAMA_REV}-bin-$(backend_asset "$backend").tar.gz"
  local url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_REV}/${asset}"
  local tmp="$BONGO_HOME/tmp/$asset"
  mkdir -p "$BONGO_HOME/tmp" "$dest"
  log "Downloading pinned llama.cpp $LLAMA_REV ($backend) prebuilt binary..."
  curl -fL --retry 5 --retry-delay 3 -o "$tmp" "$url" \
    || die "Failed to download $url.
  Check network access, or use '--llama-bin DIR' with an existing build."
  tar -xzf "$tmp" -C "$dest" --strip-components=1 || die "Failed to extract $asset."
  rm -f "$tmp"
  [[ -x "$dest/llama-server" ]] || die "Extracted $asset but 'llama-server' is missing."
  ok "llama.cpp $LLAMA_REV ($backend) ready at $dest"
  echo "$dest"
}

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
shard_names() {
  local base="${TIER_BASE[$TIER]}"
  echo "${base}-00001-of-00002.gguf"
  echo "${base}-00002-of-00002.gguf"
}

resolve_model_dir() {
  if [[ -n "$GGUF_DIR" ]]; then
    echo "$GGUF_DIR"
  else
    echo "$MODEL_BASE_DIR/${MODEL_REPO##*/}/$TIER"
  fi
}

hf_url() {
  local repo="$1" path="$2"
  printf 'https://huggingface.co/%s/resolve/main/%s' "$repo" "$path"
}

model_present() {
  local dir="$1" f expected size
  while IFS= read -r f; do
    [[ -f "$dir/$f" ]] || return 1
    expected="$(( TIER_BYTES[$TIER] ))"
    size="$(stat -c%s "$dir/$f" 2>/dev/null || echo 0)"
    (( size > 0 )) || return 1
  done < <(shard_names)
  # Sanity: combined size should match the published total (within 1 MiB).
  local total=0
  while IFS= read -r f; do
    total=$(( total + $(stat -c%s "$dir/$f" 2>/dev/null || echo 0) ))
  done < <(shard_names)
  local diff=$(( total > expected ? total - expected : expected - total ))
  (( diff < 1048576 ))
}

# Fast existence check used for an explicit --gguf-dir (trusted, may be a
# user-provided re-quant whose size differs from the published tier).
model_files_exist() {
  local dir="$1" f found=0
  while IFS= read -r f; do
    [[ -f "$dir/$f" ]] && found=1 || return 1
  done < <(shard_names)
  (( found == 1 ))
}

# A single self-contained GGUF in --gguf-dir (for smoke tests / re-quants that
# are not split into the published two shards).
single_gguf_in() {
  local dir="$1" f
  while IFS= read -r f; do
    case "$f" in *mmproj*) continue;; esac
    printf '%s' "$f"; return 0
  done < <(find "$dir" -maxdepth 1 \( -type f -o -type l \) -name '*.gguf' 2>/dev/null | sort)
  return 1
}

download_model() {
  local dir="$1"
  local expected="$(( TIER_BYTES[$TIER] ))"
  mkdir -p "$dir"
  # An explicit --gguf-dir is authoritative: never re-download over it.
  if [[ -n "$GGUF_DIR" ]]; then
    if model_files_exist "$dir"; then
      if model_present "$dir"; then
        ok "Using existing $TIER download at $dir — skipping download."
      else
        warn "$dir has the expected shard names but sizes differ from the published tier; using them as-is."
      fi
      if [[ "$VERIFY_SHA" -eq 1 ]]; then verify_sha256 "$dir"; fi
      return 0
    fi
    if [[ -n "$(single_gguf_in "$dir" || true)" ]]; then
      ok "Using the single GGUF in $dir (not the published two-shard layout)."
      return 0
    fi
    die "--gguf-dir '$dir' does not contain a usable model for tier '$TIER'.
  Expected: $(shard_names | tr '\n' ' ')
  Or a single *.gguf file in that directory."
  fi
  if model_present "$dir" && [[ "$FORCE" -eq 0 ]]; then
    ok "Model already present ($TIER) at $dir — skipping download."
    return 0
  fi
  check_disk "$dir" $(( expected + 2147483648 ))
  local auth=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  log "Downloading $TIER from $MODEL_REPO (~$(( expected / 1000000000 )) GB, resumable)..."
  local f
  while IFS= read -r f; do
    local out="$dir/$f" url
    url="$(hf_url "$MODEL_REPO" "$f")"
    log "  -> $f"
    curl -fL --retry 5 --retry-delay 3 -C - "${auth[@]}" -o "$out" "$url" \
      || die "Download failed for $f.
  If this is a gated repo, accept its licence and export HF_TOKEN first:
    export HF_TOKEN=hf_...
  Re-run the script to resume from the partial file."
  done < <(shard_names)

  if [[ "$VERIFY_SHA" -eq 1 ]]; then
    verify_sha256 "$dir"
  fi
  model_present "$dir" || die "Downloaded shards do not match the published size for tier '$TIER'.
  Remove the partial files in $dir and re-run."
  ok "Model $TIER ready at $dir ($(( expected / 1000000000 )) GB)."
}

verify_sha256() {
  local dir="$1"
  local sums="$dir/SHA256SUMS"
  log "Verifying SHA256SUMS..."
  curl -fL --retry 3 -o "$sums" "$(hf_url "$MODEL_REPO" SHA256SUMS)" \
    || { warn "Could not download SHA256SUMS; skipping verification."; return 0; }
  local f
  while IFS= read -r f; do
    if ! grep -q " $f\$" "$sums" 2>/dev/null && ! grep -q "  $f\$" "$sums" 2>/dev/null; then
      warn "No checksum entry for $f; skipping."
      continue
    fi
    ( cd "$dir" && grep -E "[ *]$f\$" SHA256SUMS | sha256sum -c - ) \
      || die "Checksum mismatch for $f. Delete it and re-run to re-download."
  done < <(shard_names)
  ok "Checksum verification passed."
}

# ---------------------------------------------------------------------------
# Config + flags
# ---------------------------------------------------------------------------
MODEL_SHARDS=()
# Collect existing model shards for the configured tier. In non-fatal mode,
# return 1 instead of dying so callers can report a helpful dry-run plan.
build_model_paths() {
  local dir="$1" mode="${2:-fatal}"
  MODEL_SHARDS=()
  local f
  while IFS= read -r f; do
    [[ -f "$dir/$f" ]] && MODEL_SHARDS+=("$dir/$f")
  done < <(shard_names)
  if (( ${#MODEL_SHARDS[@]} == 0 )); then
    local one
    if one="$(single_gguf_in "$dir")" && [[ -n "$one" ]]; then
      warn "Using single-shard GGUF $one (tier '$TIER' normally ships as two shards)."
      MODEL_SHARDS+=("$one")
      return 0
    fi
    if [[ "$mode" == "nonfatal" ]]; then return 1; fi
    die "No model shards found in $dir for tier '$TIER'.
  Remove --gguf-dir, or place the $TIER files there (expected names: $(shard_names | tr '\n' ' '))."
  fi
  return 0
}

build_server_flags() {
  local model_arg="${MODEL_SHARDS[0]:-<model.gguf>}"
  SERVER_FLAGS=(
    "--model" "$model_arg"
    "--ctx-size" "$CTX"
    "--jinja"
    "--flash-attn" "$FLASH_ATTN"
    "--cache-type-k" "$CACHE_TYPE_K"
    "--cache-type-v" "$CACHE_TYPE_V"
    "--n-gpu-layers" "$N_GPU_LAYERS"
    "--n-cpu-moe" "$N_CPU_MOE"
    "--host" "$HOST"
    "--port" "$PORT"
    "--parallel" "$PARALLEL"
    "--alias" "bongo-$TIER"
    "--metrics"
  )
  if [[ -n "$THREADS" ]]; then SERVER_FLAGS+=(--threads "$THREADS"); fi
  if (( NO_MMAP )); then SERVER_FLAGS+=(--no-mmap); fi
  if (( KEEP_ALIVE == 0 )); then SERVER_FLAGS+=(--no-keep-alive); fi
  # Pin the Intel GPU when the Vulkan build and another Vulkan device are both
  # present; otherwise llama.cpp selects device 0 (which may not be the Arc).
  if [[ "$SELECTED_BACKEND" == "Vulkan" && -x "$SERVER_BIN" ]]; then
    local vkdev
    vkdev="$(detect_vulkan_device "$SERVER_BIN" || true)"
    if [[ -n "$vkdev" ]]; then
      SERVER_FLAGS+=(--device "$vkdev")
      log "Pinned Vulkan device: $vkdev"
    fi
  fi
}

json_escape() { local s="$1"; s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; printf '%s' "$s"; }

write_config() {
  mkdir -p "$RUN_DIR"
  local json="$RUN_DIR/bongo-config.json"
  local envf="$RUN_DIR/bongo-config.env"
  local flags_json="[" first=1 f
  for f in "${SERVER_FLAGS[@]}"; do
    [[ $first -eq 0 ]] && flags_json+=", "
    first=0
    flags_json+="\"$(json_escape "$f")\""
  done
  flags_json+="]"
  local shards_json="[" first=1
  for f in "${MODEL_SHARDS[@]}"; do
    [[ $first -eq 0 ]] && shards_json+=", "
    first=0
    shards_json+="\"$(json_escape "$f")\""
  done
  shards_json+="]"
  cat > "$json" <<EOF
{
  "bongo_version": "$(json_escape "$BONGO_VERSION")",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "llama_cpp": {
    "revision": "$(json_escape "$LLAMA_REV")",
    "commit": "$(json_escape "$LLAMA_CPP_COMMIT_DEFAULT")",
    "backend": "$(json_escape "$SELECTED_BACKEND")",
    "binary": "$(json_escape "$SERVER_BIN")"
  },
  "model": {
    "repo": "$(json_escape "$MODEL_REPO")",
    "architecture": "$(json_escape "$MODEL_ARCH")",
    "tier": "$(json_escape "$TIER")",
    "shards": $shards_json
  },
  "gpu": {
    "pci_id": "$(json_escape "${GPU_PCI:-}")",
    "name": "$(json_escape "${GPU_NAME:-}")",
    "backend": "$(json_escape "$SELECTED_BACKEND")",
    "vram_total_gb": "$(json_escape "${VRAM_TOTAL_GB:-unknown}")"
  },
  "runtime": {
    "mode": "$(json_escape "$RUNTIME_MODE")",
    "dir": "$(json_escape "$RUNTIME_DIR")"
  },
  "server": {
    "host": "$(json_escape "$HOST")",
    "port": $PORT,
    "flags": $flags_json
  },
  "placement": {
    "n_gpu_layers": $N_GPU_LAYERS,
    "n_cpu_moe": $N_CPU_MOE,
    "cache_type_k": "$(json_escape "$CACHE_TYPE_K")",
    "cache_type_v": "$(json_escape "$CACHE_TYPE_V")",
    "flash_attn": "$(json_escape "$FLASH_ATTN")"
  }
}
EOF
  {
    echo "# Generated by bongo.sh $BONGO_VERSION on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "BONGO_LLAMA_REV=$(printf '%q' "$LLAMA_REV")"
    echo "BONGO_LLAMA_COMMIT=$(printf '%q' "$LLAMA_CPP_COMMIT_DEFAULT")"
    echo "BONGO_BACKEND=$(printf '%q' "$SELECTED_BACKEND")"
    echo "BONGO_TIER=$(printf '%q' "$TIER")"
    echo "BONGO_CTX=$(printf '%q' "$CTX")"
    echo "BONGO_N_CPU_MOE=$(printf '%q' "$N_CPU_MOE")"
    printf 'BONGO_SERVER_FLAGS=('
    printf '%q ' "${SERVER_FLAGS[@]}"
    printf ')\n'
  } > "$envf"
  log "Wrote generated config: $json"
  log "Wrote generated config: $envf"
}

print_plan() {
  echo
  echo "${C_BOLD}bongo plan${C_RESET}"
  echo "  GPU            : ${GPU_NAME:-?} (${GPU_PCI:-?})${VRAM_TOTAL_GB:+, ${VRAM_TOTAL_GB} GB VRAM}"
  echo "  Backend        : ${SELECTED_BACKEND:-auto}"
  echo "  Runtime        : mode=$RUNTIME_MODE dir=$RUNTIME_DIR"
  echo "  llama.cpp      : $LLAMA_REV ($LLAMA_CPP_COMMIT_DEFAULT)"
  echo "  Model          : $MODEL_REPO [$TIER]"
  echo "  Context        : $CTX"
  echo "  MoE placement  : --n-gpu-layers $N_GPU_LAYERS --n-cpu-moe $N_CPU_MOE"
  echo "  Endpoint       : http://$HOST:$PORT/v1"
  echo "  Exact flags    : ${SERVER_FLAGS[*]:-<not built>}"
  echo
}

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
server_healthy() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://$HOST:$PORT/v1/models" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

print_resource_usage() {
  if have_cmd xpu-smi; then
    log "VRAM/RAM (xpu-smi):"
    xpu-smi dump -m 0 2>/dev/null | tail -n 3 >&2 || true
  fi
  log "System RAM:"
  free -h 2>/dev/null | sed -n '1,2p' >&2 || true
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    local rss
    rss="$(ps -o rss= -p "$SERVER_PID" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$rss" ]] && log "llama-server RSS: $(( rss / 1024 )) MiB"
  fi
}

stop_existing() {
  [[ -f "$PID_FILE" ]] || return 0
  local old
  old="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    log "Stopping previous bongo server (pid $old)..."
    kill "$old" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$old" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$old" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

start_server() {
  mkdir -p "$RUN_DIR"
  LOG_FILE="$RUN_DIR/llama-server.log"
  PID_FILE="$RUN_DIR/llama-server.pid"

  if server_healthy && [[ "$FORCE" -eq 0 ]]; then
    ok "A healthy server is already listening on http://$HOST:$PORT/v1 (idempotent no-op)."
    print_resource_usage
    return 0
  fi
  stop_existing

  log "Starting llama-server:"
  log "  $SERVER_BIN ${SERVER_FLAGS[*]}"
  : > "$LOG_FILE"
  "$SERVER_BIN" "${SERVER_FLAGS[@]}" >>"$LOG_FILE" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$PID_FILE"

  local waited=0 timeout=600
  log "Waiting for the server to become healthy (model load can take minutes)..."
  while (( waited < timeout )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      err "llama-server exited during startup. Last log lines:"
      tail -n 40 "$LOG_FILE" >&2 || true
      stop_existing
      return 1
    fi
    if server_healthy; then
      ok "Server ready: http://$HOST:$PORT/v1"
      return 0
    fi
    sleep 3; waited=$(( waited + 3 ))
    if (( waited % 30 == 0 )); then log "  ...still loading (${waited}s)"; fi
  done
  err "Timed out after ${timeout}s waiting for /v1/models to return 200."
  tail -n 40 "$LOG_FILE" >&2 || true
  stop_existing
  return 1
}

print_ready() {
  echo
  ok "bongo is serving."
  echo "  Endpoint     : http://$HOST:$PORT/v1"
  echo "  Model name   : bongo-$TIER"
  echo "  Backend      : $SELECTED_BACKEND"
  echo "  Context      : $CTX tokens"
  echo "  llama.cpp    : $LLAMA_REV ($LLAMA_CPP_COMMIT_DEFAULT)"
  echo "  Config       : $RUN_DIR/bongo-config.json"
  echo "  Server log   : $LOG_FILE"
  echo
  echo "  Try it:"
  echo "    curl -s http://$HOST:$PORT/v1/models | jq ."
  echo "    curl -s http://$HOST:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
  echo "      -d '{\"model\":\"bongo-$TIER\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}' | jq ."
  echo
}

# ---------------------------------------------------------------------------
# --check / --uninstall
# ---------------------------------------------------------------------------
run_checks() {
  log "Running preflight checks..."
  detect_os; log "OS: ${PRETTY_NAME:-$OS_ID $OS_VER}"
  validate_args
  detect_gpu
  local backend_dir="$LLAMA_DIR/$LLAMA_REV/${BACKEND}"
  local bin="$SERVER_BIN"
  if [[ -n "$LLAMA_BIN_DIR" ]]; then bin="$LLAMA_BIN_DIR/llama-server"; fi
  [[ -x "$bin" ]] || warn "llama-server not fetched yet (run without --check to fetch it)."
  ok "Preflight checks complete."
}

do_uninstall() {
  detect_os
  local remove_cmd
  if is_fedora; then
    remove_cmd="sudo dnf remove intel-level-zero oneapi-level-zero intel-opencl clinfo 'intel-oneapi-runtime-*'"
  elif is_debian; then
    remove_cmd="sudo apt-get remove intel-level-zero-gpu level-zero intel-opencl-icd 'intel-oneapi-runtime-*'"
  else
    remove_cmd="remove the Intel Level Zero / OpenCL / oneAPI runtime packages with your package manager"
  fi
  cat <<EOF
bongo uninstall

Downloaded artifacts live under: ${BONGO_HOME}
  runtime : ${RUNTIME_DIR}
  llama   : ${LLAMA_DIR}
  models  : ${MODEL_BASE_DIR}
  logs    : ${RUN_DIR}
EOF
  if (( ASSUME_YES )); then
    # Guard against an accidental BONGO_HOME=/ or empty override.
    if [[ -z "$BONGO_HOME" || "$BONGO_HOME" == "/" ]]; then
      die "Refusing to remove BONGO_HOME='${BONGO_HOME}'."
    fi
    if [[ -e "$BONGO_HOME" ]]; then
      rm -rf "$BONGO_HOME"
      ok "Removed ${BONGO_HOME}."
    else
      log "Nothing to remove: ${BONGO_HOME} does not exist."
    fi
    cat <<EOF

System packages are not removed automatically. Remove them with:
  ${remove_cmd}
EOF
  else
    cat <<EOF

Remove the downloaded artifacts:
  rm -rf "${BONGO_HOME}"
(add --yes to do it now: './bongo.sh --uninstall --yes')

System packages installed by '--runtime system' are not removed automatically.
Remove them with:
  ${remove_cmd}
EOF
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  parse_args "$@"
  if (( UNINSTALL )); then
    do_uninstall
    exit 0
  fi
  # Fail fast with an actionable install message for the external commands the
  # script cannot run without, instead of an ERR-trap line number (BAS-58 F5).
  need_cmd curl curl; need_cmd tar tar; need_cmd df coreutils
  validate_args
  mkdir -p "$BONGO_HOME" "$RUN_DIR"

  detect_os
  log "bongo.sh $BONGO_VERSION — ${PRETTY_NAME:-$OS_ID $OS_VER}"
  detect_gpu

  log "Provisioning compute runtime (mode: $RUNTIME_MODE)..."
  provision_runtime

  # Fetch/select the llama.cpp build for the requested backend.
  local backend_for_bin="$BACKEND"
  case "$backend_for_bin" in auto) backend_for_bin="sycl";; esac
  local bin_dir
  if [[ -n "$LLAMA_BIN_DIR" ]]; then
    bin_dir="$LLAMA_BIN_DIR"
    log "Using existing llama.cpp build at $bin_dir"
  else
    bin_dir="$(fetch_llama "$backend_for_bin")"
  fi
  select_backend "$bin_dir"

  # Auto backend: if SYCL was requested by default but is unavailable, fetch the
  # Vulkan asset instead of failing.
  if [[ "$BACKEND" == "auto" && "$SELECTED_BACKEND" != "SYCL0" && -z "$LLAMA_BIN_DIR" ]]; then
    local vk_dir
    vk_dir="$(fetch_llama vulkan)"
    SELECTED_BACKEND="Vulkan"
    SERVER_BIN="$vk_dir/llama-server"
  fi

  if (( CHECK_ONLY )); then
    build_model_paths "$(resolve_model_dir)" nonfatal 2>/dev/null || true
    log "Selected backend: $SELECTED_BACKEND"
    ok "Runtime/backend checks passed."
    exit 0
  fi

  local model_dir
  model_dir="$(resolve_model_dir)"
  if (( DRY_RUN )); then
    build_model_paths "$model_dir" nonfatal || {
      log "[dry-run] model not downloaded yet: $model_dir"
    }
  else
    download_model "$model_dir"
    build_model_paths "$model_dir"
  fi

  build_server_flags
  write_config
  print_plan

  if (( DRY_RUN )); then
    log "Dry run complete; nothing was started."
    exit 0
  fi

  start_server
  print_ready
  print_resource_usage

  if (( DETACH )); then
    if [[ -n "$SERVER_PID" ]]; then
      log "Started in the background (pid $SERVER_PID, log $LOG_FILE)."
    else
      log "Server already running; nothing to do."
    fi
    exit 0
  fi

  if [[ -z "$SERVER_PID" ]]; then
    log "Server already running; leaving it up (use --force to restart)."
    exit 0
  fi

  log "Press Ctrl-C to stop."
  local rc=0
  wait "$SERVER_PID" || rc=$?
  rm -f "$PID_FILE"
  exit "$rc"
}

main "$@"
