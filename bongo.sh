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

# Directory of this script. Used to locate the in-repo engine build helper when
# bongo.sh is run from a checkout (a standalone copy degrades gracefully).
# Builtins only: bongo.sh must start on a host with an almost-empty PATH (BAS-58 F5).
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]] && SCRIPT_DIR="."
if [[ -d "$SCRIPT_DIR" ]]; then
  SCRIPT_DIR="$(cd -- "$SCRIPT_DIR" 2>/dev/null && pwd -P)" || SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
fi

# Single-GPU serialisation (BAS-80). `bench/gpu-lock.sh` lives beside this
# script in the repo; when bongo.sh is copied standalone the lock degrades to a
# no-op, which is acceptable for a dev box without the bench tree.
GPU_LOCK_LIB="${BONGO_GPU_LOCK_LIB:-${BASH_SOURCE[0]%/*}/bench/gpu-lock.sh}"
if [[ -r "$GPU_LOCK_LIB" ]]; then
  # shellcheck disable=SC1090
  . "$GPU_LOCK_LIB"
fi

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
N_CPU_MOE_SET=0          # 1 once --n-cpu-moe is given (disables the auto fallback)
N_CPU_MOE_REQUESTED=""   # the split the placement policy chose, before any fallback
PLACEMENT_FALLBACK=0     # 1 once --placement auto fell back to the tier split
PLACEMENT_FALLBACK_REASON=""
# M4.5 (BAS-163) expert placement policy. 'auto' (default) = spend the VRAM that
# the configured context leaves free on expert residency, then retry once at the
# tier split if the server fails to load there (M4.4/BAS-159 measured the gain).
# 'tier' = the fixed per-tier split (the large-context-safe value). An explicit
# --n-cpu-moe overrides either and is never overridden by the fallback.
# 'auto' is context-aware: it picks the small- or large-context split and can fall
# back to the safe split for the active context.
PLACEMENT="auto"
THREADS=""
FLASH_ATTN="on"
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
PARALLEL=1
LOAD_MODE=""             # "" = engine default (auto/mmap); --load-mode MODE or --no-mmap sets it
DRY_RUN=0
DETACH=0
FORCE=0
CHECK_ONLY=0
UNINSTALL=0
SKIP_DOWNLOAD=0
VERIFY_SHA=0
ASSUME_YES=0
KEEP_ALIVE=1
# Prefix-cache serving (agentic turns: a small prompt that grows).
CACHE_PROMPT=1            # llama.cpp --cache-prompt is on by default; keep it explicit
CACHE_IDLE_SLOTS=""       # "" = engine default (on); 1/0 force --[no-]cache-idle-slots
CTX_CHECKPOINTS=""        # "" = engine default (32); N = --ctx-checkpoints N
SLOT_SAVE_PATH=""         # "" = disabled; set by --slot-save-path or the run-dir default
SLOT_SAVE_PATH_SET=0       # 1 once --slot-save-path/--no-slot-save-path was given
SAVE_SLOT_CHECKPOINTS=0    # 1 persist context checkpoints alongside the slot file (BAS-86)
WARMUP=1                   # warm the server after load (pays shader/kernel compile once)
# Engine + M4.2 host-expert upload (BAS-158). The shipped default is the patched
# Vulkan engine with the two upload levers on; `--engine stage0` is the no-rebuild
# opt-out that restores the Stage 0 Vulkan baseline.
ENGINE_MODE="${BONGO_ENGINE:-m42}"   # m42 (pinned llama.cpp + M4.2 patch) | stage0 (stock)
M42_UPLOAD=1                          # 1 = --load-mode none + the two GGML_VK levers
ENGINE_PATCHED=0                      # 1 once the M4.2-patched engine is selected

SERVER_PID=""
SERVER_BIN=""
SERVER_FLAGS=()
SERVER_ENV=()            # env assignments for llama-server (the M4.2 upload levers)
EFFECTIVE_LOAD_MODE=""  # the load mode actually emitted (default or explicit)
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
# This is the --placement tier value (the fixed per-tier split at the default
# 131072 context) and the fallback target for a small-context auto run.
declare -A TIER_N_CPU_MOE=(
  [iq2_xs]=16
  [iq3_xxs]=22
  [q2_0]=15
)
# --placement auto: the split for a context of 131072 or less.  Measured on the
# reference box with the M4.2 levers on (BAS-159): --n-cpu-moe 12 is the last
# value that loads at --ctx 131072 (10 and 8 fail with ErrorOutOfDeviceMemory)
# and it takes 4K decode from 17.0 to 19.6 tok/s.  Only iq2_xs is measured; other
# tiers fall back to TIER_N_CPU_MOE.
declare -A TIER_N_CPU_MOE_SMALL_CTX=(
  [iq2_xs]=12
)
# --placement auto for a context above 131072.  The larger KV cache leaves less
# VRAM for experts, so the split must move more experts to the CPU: for iq2_xs
# the 256K fit measured n=18 as the safe value (30.35-30.65 GiB, ~1.5 GiB margin),
# while the tier value n=16 (31.79 GiB) device-losts during load on the reference
# box (BAS-163, observed 2026-09-29).  Only iq2_xs is measured; other tiers fall
# back to TIER_N_CPU_MOE.
declare -A TIER_N_CPU_MOE_LARGE_CTX=(
  [iq2_xs]=18
)

# The expert split the --placement auto policy picks for the active context.
# Falls back to the tier value for an unmeasured tier/context.
auto_n_cpu_moe() {
  if (( CTX <= 131072 )); then
    printf '%s' "${TIER_N_CPU_MOE_SMALL_CTX[$TIER]:-${TIER_N_CPU_MOE[$TIER]}}"
  else
    printf '%s' "${TIER_N_CPU_MOE_LARGE_CTX[$TIER]:-${TIER_N_CPU_MOE[$TIER]}}"
  fi
}

# The safe split to fall back to when the auto value fails to load: the tier
# value for a <=131072 context, the large-context value above it.  Above 131072
# auto already selects this value, so the fallback is a small-context mechanism.
placement_fallback_target() {
  if (( CTX > 131072 )) && [[ -n "${TIER_N_CPU_MOE_LARGE_CTX[$TIER]:-}" ]]; then
    printf '%s' "${TIER_N_CPU_MOE_LARGE_CTX[$TIER]}"
  else
    printf '%s' "${TIER_N_CPU_MOE[$TIER]}"
  fi
}

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
  --backend NAME         auto (default: Vulkan, SYCL fallback) | sycl | vulkan | cpu
  --engine MODE          Engine build: m42 (default) = the pinned llama.cpp Vulkan build with the
                         M4.2 host-expert upload patch; stage0 = the stock prebuilt Stage 0 build.
                         '--engine stage0' is the no-rebuild opt-out (BAS-158).
  --m42-upload           Enable the M4.2 host-expert upload levers (default: on): --load-mode none
                         plus GGML_VK_HOST_BUFT_PER_DEVICE=1 and GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1
  --no-m42-upload        Keep the selected engine but turn the upload levers off (Stage 0 defaults)
  --n-cpu-moe N          MoE layers with experts on CPU (explicit placement)
  --placement MODE       Expert placement policy. auto (default) = spend the VRAM the
                         configured context leaves free on expert residency: iq2_xs keeps
                         12 instead of 16 layers of experts on the CPU at --ctx 131072
                         and below, which measures 17.0 -> 19.6 tok/s on 4K decode
                         (BAS-159), and the measured large-context split above (18 at
                         262144; the tier value 16 device-losts there, BAS-163). If the
                         auto placement fails to load, bongo.sh retries once at the safe
                         split for the context (16 at 131072) and keeps serving. tier =
                         the fixed per-tier split (the opt-out). An explicit --n-cpu-moe
                         overrides either and is never overridden by the fallback.
  --n-gpu-layers N       Max layers offloaded to GPU (default: $N_GPU_LAYERS)
  --threads N            CPU threads (default: auto)
  --load-mode MODE       Model loading mode: none (default when the M4.2 upload levers are on),
                         or auto|mmap|mlock|mmap+mlock|dio. 'none' reads the model into anonymous
                         RAM instead of an mmap and removes the in-turn page-cache re-reads of the
                         host-resident MoE experts. The opt-out ('--engine stage0') omits it.
  --no-mmap              Alias for --load-mode none.
  --detach               Start the server in the background and exit
  --no-keep-alive        Let llama-server exit when idle

Prefix cache / agentic turns:
  --cache-prompt         Force llama.cpp --cache-prompt on (default: on)
  --no-cache-prompt      Pass --no-cache-prompt (cold prefill; baseline A/B)
  --slot-save-path DIR   Save/restore slot KV under DIR (default: $RUN_DIR/slots)
  --no-slot-save-path    Disable the slot save/restore endpoint (engine default)
  --cache-idle-slots     Force --cache-idle-slots on (engine default: on)
  --no-cache-idle-slots  Pass --no-cache-idle-slots
  --ctx-checkpoints N    Pass --ctx-checkpoints N (engine default: 32)
  --save-slot-checkpoints  Persist context checkpoints alongside the slot file so a
                           restored slot on hybrid/recurrent models reuses the prefix
                           (default: off; requires --slot-save-path)
  --no-save-slot-checkpoints
  --no-warmup            Do not send the post-load warmup request

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
  BONGO_ENGINE_DIR       Pinned llama.cpp source/build tree for the M4.2 engine
                         (default: $BONGO_HOME/engine/llama.cpp-pin)
  BONGO_ENGINE           Default for --engine (m42 | stage0)
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
      --engine) ENGINE_MODE="${2:?--engine needs a value}"; shift 2;;
      --m42-upload) M42_UPLOAD=1; shift;;
      --no-m42-upload) M42_UPLOAD=0; shift;;
      --n-cpu-moe) N_CPU_MOE="${2:?--n-cpu-moe needs a value}"; N_CPU_MOE_SET=1; shift 2;;
      --placement) PLACEMENT="${2:?--placement needs a value}"; shift 2;;
      --n-gpu-layers) N_GPU_LAYERS="${2:?--n-gpu-layers needs a value}"; shift 2;;
      --threads) THREADS="${2:?--threads needs a value}"; shift 2;;
      --load-mode) LOAD_MODE="${2:?--load-mode needs a value}"; shift 2;;
      --no-mmap) LOAD_MODE="none"; shift;;
      # The published GGUF has no MTP/NextN head and llama.cpp qwen4exp cannot convert or run one;
      # speculation is the n-gram/PLE table. Refuse the flag with an actionable message.
      --mtp) die "--mtp is not supported for this model: the published GGUF has no MTP head and llama.cpp qwen4exp cannot run one. Speculation uses the lazy-read n-gram/PLE table. See docs/research/intel-arc-b70.md section 2.1.";;
      --detach) DETACH=1; shift;;
      --no-keep-alive) KEEP_ALIVE=0; shift;;
      --cache-prompt) CACHE_PROMPT=1; shift;;
      --no-cache-prompt) CACHE_PROMPT=0; shift;;
      --slot-save-path) SLOT_SAVE_PATH="${2:?--slot-save-path needs a value}"; SLOT_SAVE_PATH_SET=1; shift 2;;
      --no-slot-save-path) SLOT_SAVE_PATH=""; SLOT_SAVE_PATH_SET=1; shift;;
      --cache-idle-slots) CACHE_IDLE_SLOTS=1; shift;;
      --no-cache-idle-slots) CACHE_IDLE_SLOTS=0; shift;;
      --ctx-checkpoints) CTX_CHECKPOINTS="${2:?--ctx-checkpoints needs a value}"; shift 2;;
      --save-slot-checkpoints) SAVE_SLOT_CHECKPOINTS=1; shift;;
      --no-save-slot-checkpoints) SAVE_SLOT_CHECKPOINTS=0; shift;;
      --no-warmup) WARMUP=0; shift;;
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
  case "$ENGINE_MODE" in
    m42|stage0) ;;
    *) die "Unknown --engine '$ENGINE_MODE'. Use m42 (the M4.2-patched Vulkan engine) or stage0 (stock).";;
  esac
  # The stage0 engine is the Stage 0 behaviour: no --load-mode none and no M4.2
  # upload env, so the upload levers go with it.
  if [[ "$ENGINE_MODE" == "stage0" ]]; then M42_UPLOAD=0; fi
  case "$RUNTIME_MODE" in auto|system|user|dir) ;; *) die "Unknown runtime mode '$RUNTIME_MODE'. Use auto|system|user|dir.";; esac
  [[ "$CTX" =~ ^[0-9]+$ ]] || die "--ctx must be an integer (got '$CTX')."
  (( CTX >= 131072 )) || die "Context $CTX is below the 131072 acceptance minimum. Use --ctx 131072 or higher."
  [[ "$PORT" =~ ^[0-9]+$ ]] || die "--port must be an integer."
  [[ "$N_GPU_LAYERS" =~ ^[0-9]+$ ]] || die "--n-gpu-layers must be an integer."
  if [[ -n "$LOAD_MODE" ]]; then
    case "$LOAD_MODE" in
      auto|none|mmap|mlock|mmap+mlock|dio) ;;
      *) die "Unknown --load-mode '$LOAD_MODE'. Use auto|none|mmap|mlock|mmap+mlock|dio.";;
    esac
  fi
  if [[ -n "$CTX_CHECKPOINTS" ]]; then
    [[ "$CTX_CHECKPOINTS" =~ ^[0-9]+$ ]] || die "--ctx-checkpoints must be an integer (got '$CTX_CHECKPOINTS')."
  fi
  if (( SAVE_SLOT_CHECKPOINTS )) && [[ -z "$SLOT_SAVE_PATH" ]]; then
    die "--save-slot-checkpoints requires --slot-save-path (checkpoints are persisted alongside the slot KV)."
  fi
  # The slot save/restore endpoint is on the product path by default so a long
  # agentic session can persist its KV across idle/restart. It only writes when a
  # client calls /slots/{id}?action=save; --no-slot-save-path restores the engine default.
  if (( SLOT_SAVE_PATH_SET == 0 )); then SLOT_SAVE_PATH="$RUN_DIR/slots"; fi
  if [[ -n "$N_CPU_MOE" ]]; then
    [[ "$N_CPU_MOE" =~ ^[0-9]+$ ]] || die "--n-cpu-moe must be an integer."
  else
    [[ "$PLACEMENT" == "tier" || "$PLACEMENT" == "auto" ]] \
      || die "--placement must be 'tier' or 'auto' (got '$PLACEMENT')."
    if [[ "$PLACEMENT" == "auto" ]]; then
      N_CPU_MOE="$(auto_n_cpu_moe)"
    else
      N_CPU_MOE="${TIER_N_CPU_MOE[$TIER]}"
    fi
  fi
  N_CPU_MOE_REQUESTED="$N_CPU_MOE"
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
  # NEO only keeps the IGC/LLVM backend loaded when llvm15/lib is on the link
  # path. Without it the Level Zero probe dies in gmm_helper/resource_info.cpp
  # (SIGABRT, no device) and --backend sycl reports "no SYCL device was found"
  # even though the GPU is healthy. See bench/micro/levelzero_probe.py.
  [[ -d "$base/usr/lib64/llvm15/lib" ]] && libdirs+=("$base/usr/lib64/llvm15/lib")
  [[ -d "$base/lib" ]] && libdirs+=("$base/lib")
  if (( ${#libdirs[@]} )); then
    local joined
    joined="$(IFS=:; echo "${libdirs[*]}")"
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # ZEL_LIBRARY_PATH must name exactly one directory. The Level Zero driver
    # treats a colon-separated list as a single (invalid) path, enumerates zero
    # devices and every SYCL call fails with "No device of requested type
    # available". setup_runtime_env() can run twice in one invocation
    # (provision_runtime + select_backend), so a naive prepend produced
    # "<prefix>/usr/lib64:<prefix>/usr/lib64" and --backend sycl reported
    # "no SYCL device was found" on a healthy Arc B70 (BAS-72).
    if [[ "${ZEL_LIBRARY_PATH:-}" != "$base/usr/lib64" ]]; then
      export ZEL_LIBRARY_PATH="$base/usr/lib64"
    fi
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
    PROBE_SYCL_OUTPUT="$out"
    return 1
  fi
  # The helper lists "Found N SYCL devices" and a device table.
  if grep -qiE 'Found [1-9][0-9]* SYCL devices|Device 0' <<<"$out"; then
    return 0
  fi
  PROBE_SYCL_OUTPUT="$out"
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
        if [[ -n "${PROBE_SYCL_OUTPUT:-}" ]]; then
          warn "Last probe output: $(tail -n 3 <<<"$PROBE_SYCL_OUTPUT")"
        fi
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
      # Vulkan first, then SYCL, then CPU.  Measurement on the reference box
      # (M3.0 A/B, BAS-72, ADR-0002 amendment) found SYCL slower at 128K on
      # prefill, decode and the cached-turn TTFT, so Vulkan is the shipped
      # default; SYCL stays fully selectable with --backend sycl.
      # detect_vulkan_device() (not probe_vulkan()) is the discriminator: it
      # asks the llama.cpp build itself whether it can see the Intel GPU, which
      # is the same evidence build_server_flags() pins the device with.
      if [[ -n "$(detect_vulkan_device "$bin_dir/llama-server" || true)" ]]; then
        SERVER_BIN="$bin_dir/llama-server"; SELECTED_BACKEND="Vulkan"
      elif probe_sycl "$bin_dir/llama-server" 2>/dev/null; then
        warn "No Vulkan device found; using SYCL instead (ADR-0002 amendment, BAS-72)."
        SERVER_BIN="$bin_dir/llama-server"; SELECTED_BACKEND="SYCL0"
      else
        warn "Neither a Vulkan nor a SYCL device could be confirmed; continuing with Vulkan."
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
# M4.2 patched Vulkan engine (BAS-158)
# ---------------------------------------------------------------------------
# The M4.2 host-expert upload fix is an engine patch, not a server flag, so the
# shipped default needs a patched build. The pinned source/build tree is cached
# under $BONGO_ENGINE_DIR; when it is missing and docker is available the tree is
# cloned at the pinned commit and built with tools/build-llama-vulkan.sh.
m42_engine_src() { printf '%s' "${BONGO_ENGINE_DIR:-$BONGO_HOME/engine/llama.cpp-pin}"; }
m42_engine_bindir() { printf '%s/build-vulkan/bin' "$(m42_engine_src)"; }

# Is the engine at BIN_DIR the M4.2-patched build? The lever literal is compiled
# into libggml-vulkan.so, not the server binary, so test the sibling Vulkan lib.
# grep reads the library directly (no pipe): a `strings | grep -q` pipeline trips
# `set -o pipefail` on the early-exit SIGPIPE and would report a false negative.
binary_is_m42_patched() {
  local bin="$1" dir lib
  [[ -x "$bin" ]] || return 1
  have_cmd grep || return 0   # cannot inspect; assume the caller knows
  dir="${bin%/*}"
  for lib in "$dir"/libggml-vulkan.so*; do
    [[ -e "$lib" ]] || continue
    if grep -qa -m1 'GGML_VK_HOST_BUFT_PER_DEVICE' "$lib" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

# Materialise the patched M4.2 Vulkan engine and print its bin directory.
# Reuses the cached build; otherwise clones the pinned llama.cpp tree and builds
# it in the Vulkan build container. Returns non-zero (with a warning) when it
# cannot, so the caller can fall back to the stock prebuilt with the levers off.
build_m42_engine() {
  local src bin_dir builder image
  src="$(m42_engine_src)"
  bin_dir="$src/build-vulkan/bin"
  builder="$SCRIPT_DIR/tools/build-llama-vulkan.sh"

  if [[ -x "$bin_dir/llama-server" ]] && binary_is_m42_patched "$bin_dir/llama-server"; then
    echo "$bin_dir"; return 0
  fi
  if [[ ! -x "$builder" ]]; then
    warn "The M4.2 engine builder is missing ($builder); cannot build the patched engine."
    return 1
  fi
  if ! have_cmd docker; then
    warn "docker is required to build the M4.2 patched Vulkan engine; it is not installed."
    return 1
  fi
  image="${BONGO_VULKAN_BUILD_IMAGE:-bongo-llama-build:vulkan}"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    warn "The Vulkan build image '$image' is missing; cannot build the M4.2 engine."
    return 1
  fi
  if [[ ! -d "$src/.git" ]]; then
    need_cmd git
    log "Cloning pinned llama.cpp $LLAMA_CPP_COMMIT_DEFAULT into $src ..."
    mkdir -p "$(dirname "$src")"
    if ! git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$src"; then
      warn "Could not clone llama.cpp; cannot build the M4.2 engine."
      return 1
    fi
    if ! git -C "$src" checkout --quiet "$LLAMA_CPP_COMMIT_DEFAULT"; then
      warn "Could not check out llama.cpp $LLAMA_CPP_COMMIT_DEFAULT."
      return 1
    fi
  fi
  log "Building the M4.2 patched Vulkan engine (first run only; this can take several minutes)..."
  if ! BONGO_APPLY_M42_PATCH=1 "$builder" "$src" llama-server; then
    warn "The M4.2 engine build failed."
    return 1
  fi
  if [[ -x "$bin_dir/llama-server" ]] && binary_is_m42_patched "$bin_dir/llama-server"; then
    echo "$bin_dir"; return 0
  fi
  warn "The M4.2 engine build produced no patched llama-server."
  return 1
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
  if [[ -n "$LOAD_MODE" ]]; then SERVER_FLAGS+=(--load-mode "$LOAD_MODE"); fi
  EFFECTIVE_LOAD_MODE="$LOAD_MODE"
  # M4.2 host-expert upload (BAS-158): the shipped default. Active only on the
  # patched engine's Vulkan path; a fallback to the stock prebuilt or to SYCL
  # clears M42_UPLOAD in main()/validate_args().
  local m42_active=0
  if (( M42_UPLOAD )) && (( ENGINE_PATCHED )) && [[ "$SELECTED_BACKEND" == "Vulkan" ]]; then
    m42_active=1
  fi
  if (( m42_active )) && [[ -z "$LOAD_MODE" ]]; then
    SERVER_FLAGS+=(--load-mode none)
    EFFECTIVE_LOAD_MODE="none"
  fi
  SERVER_ENV=()
  if (( m42_active )); then
    SERVER_ENV+=(GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1)
  fi
  if (( KEEP_ALIVE == 0 )); then SERVER_FLAGS+=(--no-keep-alive); fi
  # Prefix-cache serving: explicit in the config so the cached path is the
  # reproducible product path (not the engine default that a reader has to know).
  if (( CACHE_PROMPT )); then SERVER_FLAGS+=(--cache-prompt); else SERVER_FLAGS+=(--no-cache-prompt); fi
  if [[ -n "$SLOT_SAVE_PATH" ]]; then SERVER_FLAGS+=(--slot-save-path "$SLOT_SAVE_PATH"); fi
  if (( SAVE_SLOT_CHECKPOINTS )); then SERVER_FLAGS+=(--save-slot-checkpoints); fi
  if [[ -n "$CACHE_IDLE_SLOTS" ]]; then
    if (( CACHE_IDLE_SLOTS )); then SERVER_FLAGS+=(--cache-idle-slots); else SERVER_FLAGS+=(--no-cache-idle-slots); fi
  fi
  if [[ -n "$CTX_CHECKPOINTS" ]]; then SERVER_FLAGS+=(--ctx-checkpoints "$CTX_CHECKPOINTS"); fi
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
  local cp_json warmup_json
  if (( CACHE_PROMPT )); then cp_json=true; else cp_json=false; fi
  if (( WARMUP )); then warmup_json=true; else warmup_json=false; fi
  local flags_json="[" first=1 f
  for f in "${SERVER_FLAGS[@]}"; do
    [[ $first -eq 0 ]] && flags_json+=", "
    first=0
    flags_json+="\"$(json_escape "$f")\""
  done
  flags_json+="]"
  local env_json="[" first=1 f
  for f in "${SERVER_ENV[@]}"; do
    [[ $first -eq 0 ]] && env_json+=", "
    first=0
    env_json+="\"$(json_escape "$f")\""
  done
  env_json+="]"
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
    "engine": "$(json_escape "$ENGINE_MODE")",
    "m42_patched": $(if (( ENGINE_PATCHED )); then echo true; else echo false; fi),
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
    "flags": $flags_json,
    "env": $env_json
  },
  "placement": {
    "n_gpu_layers": $N_GPU_LAYERS,
    "policy": "$(json_escape "$PLACEMENT")",
    "auto_fallback": $(if (( PLACEMENT_FALLBACK )); then echo true; else echo false; fi),
    "auto_fallback_reason": $(if (( PLACEMENT_FALLBACK )); then printf '"%s"' "$(json_escape "$PLACEMENT_FALLBACK_REASON")"; else echo null; fi),
    "n_cpu_moe_requested": ${N_CPU_MOE_REQUESTED:-$N_CPU_MOE},
    "n_cpu_moe": $N_CPU_MOE,
    "load_mode": "$(json_escape "$EFFECTIVE_LOAD_MODE")",
    "cache_type_k": "$(json_escape "$CACHE_TYPE_K")",
    "cache_type_v": "$(json_escape "$CACHE_TYPE_V")",
    "flash_attn": "$(json_escape "$FLASH_ATTN")"
  },
  "serving": {
    "cache_prompt": $cp_json,
    "slot_save_path": "$(json_escape "$SLOT_SAVE_PATH")",
    "save_slot_checkpoints": $(if (( SAVE_SLOT_CHECKPOINTS )); then echo true; else echo false; fi),
    "cache_idle_slots": $(if [[ -z "$CACHE_IDLE_SLOTS" ]]; then echo 'null'; elif (( CACHE_IDLE_SLOTS )); then echo true; else echo false; fi),
    "ctx_checkpoints": $(if [[ -z "$CTX_CHECKPOINTS" ]]; then echo 'null'; else echo "$CTX_CHECKPOINTS"; fi),
    "warmup": $warmup_json
  }
}
EOF
  {
    echo "# Generated by bongo.sh $BONGO_VERSION on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "BONGO_LLAMA_REV=$(printf '%q' "$LLAMA_REV")"
    echo "BONGO_LLAMA_COMMIT=$(printf '%q' "$LLAMA_CPP_COMMIT_DEFAULT")"
    echo "BONGO_BACKEND=$(printf '%q' "$SELECTED_BACKEND")"
    echo "BONGO_ENGINE=$(printf '%q' "$ENGINE_MODE")"
    echo "BONGO_M42_PATCHED=$(printf '%q' "$ENGINE_PATCHED")"
    echo "BONGO_M42_UPLOAD=$(printf '%q' "$M42_UPLOAD")"
    echo "BONGO_TIER=$(printf '%q' "$TIER")"
    echo "BONGO_CTX=$(printf '%q' "$CTX")"
    echo "BONGO_N_CPU_MOE=$(printf '%q' "$N_CPU_MOE")"
    echo "BONGO_N_CPU_MOE_REQUESTED=$(printf '%q' "${N_CPU_MOE_REQUESTED:-$N_CPU_MOE}")"
    echo "BONGO_PLACEMENT=$(printf '%q' "$PLACEMENT")"
    echo "BONGO_PLACEMENT_FALLBACK=$(printf '%q' "$PLACEMENT_FALLBACK")"
    echo "BONGO_PLACEMENT_FALLBACK_REASON=$(printf '%q' "$PLACEMENT_FALLBACK_REASON")"
    echo "BONGO_CACHE_PROMPT=$(printf '%q' "$CACHE_PROMPT")"
    echo "BONGO_SLOT_SAVE_PATH=$(printf '%q' "$SLOT_SAVE_PATH")"
    printf 'BONGO_SERVER_FLAGS=('
    printf '%q ' "${SERVER_FLAGS[@]}"
    printf ')\n'
    printf 'BONGO_SERVER_ENV=('
    printf '%q ' "${SERVER_ENV[@]}"
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
  if (( ENGINE_PATCHED )); then
    echo "  Engine         : m42 (pinned llama.cpp + M4.2 host-expert upload patch)"
  else
    echo "  Engine         : $ENGINE_MODE (M4.2 patch off)"
  fi
  echo "  Server env     : ${SERVER_ENV[*]:-<none>}"
  echo "  Model          : $MODEL_REPO [$TIER]"
  echo "  Context        : $CTX"
  local placement_note="$PLACEMENT"
  if (( PLACEMENT_FALLBACK )); then
    placement_note="$PLACEMENT -> tier (auto load fallback: ${PLACEMENT_FALLBACK_REASON:-load failure})"
  fi
  echo "  MoE placement  : --n-gpu-layers $N_GPU_LAYERS --n-cpu-moe $N_CPU_MOE ($placement_note)"
  echo "  Prefix cache   : cache_prompt=$CACHE_PROMPT slot_save_path=${SLOT_SAVE_PATH:-disabled} save_slot_checkpoints=$SAVE_SLOT_CHECKPOINTS warmup=$WARMUP"
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

# ---------------------------------------------------------------------------
# M4.5 (BAS-163) auto-placement load fallback
# ---------------------------------------------------------------------------
# --placement auto spends the VRAM that a small context leaves free
# (--n-cpu-moe 12 at --ctx 131072), which is the measured load edge. If the
# server cannot load or health-check there, retry once at the tier split and
# keep serving. The fallback never overrides an explicit --n-cpu-moe.
placement_fallback_eligible() {
  (( N_CPU_MOE_SET == 0 )) || return 1
  (( PLACEMENT_FALLBACK == 0 )) || return 1
  local target
  target="$(placement_fallback_target)"
  [[ -n "$target" ]] || return 1
  [[ "$N_CPU_MOE" != "$target" ]]
}

# Classify the failed load for the record. The fallback itself triggers on any
# load failure, so an unknown driver OOM message on another box is still guarded;
# this only labels the config and the log message.
load_failure_reason() {
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]] \
     && grep -qiE 'ErrorOutOfDeviceMemory|out of (device )?memory|OutOfMemory|failed to allocate|device.?lost|allocateMemory' "$LOG_FILE" 2>/dev/null; then
    printf 'out_of_device_memory'
  else
    printf 'load_failure'
  fi
}

# Switch the active split to the tier value, keep the failed attempt's log, and
# record the fallback so it lands in bongo-config.json / the plan output.
apply_placement_fallback() {
  local target
  target="$(placement_fallback_target)"
  PLACEMENT_FALLBACK_REASON="$(load_failure_reason)"
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    cp "$LOG_FILE" "$RUN_DIR/llama-server-auto-$N_CPU_MOE.log" 2>/dev/null || true
  fi
  warn "the auto placement (--n-cpu-moe $N_CPU_MOE) failed to load ($PLACEMENT_FALLBACK_REASON); falling back to the safe placement (--n-cpu-moe $target)."
  N_CPU_MOE="$target"
  PLACEMENT_FALLBACK=1
}

# Start the server; on an auto-placement load failure fall back once to the tier
# split, rebuild the flags + config, and try again.
start_server_guarded() {
  start_server && return 0
  placement_fallback_eligible || return 1
  apply_placement_fallback
  build_server_flags
  write_config
  print_plan
  start_server
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

  if [[ -n "$SLOT_SAVE_PATH" ]]; then
    mkdir -p "$SLOT_SAVE_PATH" || die "Could not create --slot-save-path '$SLOT_SAVE_PATH'."
  fi

  log "Starting llama-server:"
  log "  $SERVER_BIN ${SERVER_FLAGS[*]}"
  if (( ${#SERVER_ENV[@]} )); then log "  env: ${SERVER_ENV[*]}"; fi
  : > "$LOG_FILE"
  if (( ${#SERVER_ENV[@]} )); then
    env "${SERVER_ENV[@]}" "$SERVER_BIN" "${SERVER_FLAGS[@]}" >>"$LOG_FILE" 2>&1 &
  else
    "$SERVER_BIN" "${SERVER_FLAGS[@]}" >>"$LOG_FILE" 2>&1 &
  fi
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
      warm_server
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

# Warm the loaded server so shader/kernel compilation is paid once, at start,
# instead of on the user's first turn. The first request after load cost ~27 s on
# the reference box (docs/research/agentic-prefix-cache.md).
warm_server() {
  if (( WARMUP == 0 )); then
    log "Skipping warmup (--no-warmup)."
    return 0
  fi
  log "Warming the server (compiles the shader/kernel set)..."
  local body='{"prompt":"warmup","max_tokens":1,"temperature":0.0,"cache_prompt":false}'
  local t0 t1 code ms
  t0="$(date +%s%N)"
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 600 \
    -H 'Content-Type: application/json' -d "$body" \
    "http://$HOST:$PORT/v1/completions" 2>/dev/null || true)"
  t1="$(date +%s%N)"
  ms=$(( (t1 - t0) / 1000000 ))
  if [[ "$code" == "200" ]]; then
    ok "Warmup done in ${ms} ms."
  else
    warn "Warmup request returned HTTP ${code:-none} after ${ms} ms; the first turn may pay the compile cost."
  fi
}

print_ready() {
  echo
  ok "bongo is serving."
  echo "  Endpoint     : http://$HOST:$PORT/v1"
  echo "  Model name   : bongo-$TIER"
  echo "  Backend      : $SELECTED_BACKEND"
  echo "  Context      : $CTX tokens"
  if [[ -n "$SLOT_SAVE_PATH" ]]; then
    echo "  Slot KV      : save/restore enabled at $SLOT_SAVE_PATH"
  fi
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
  engine  : $(m42_engine_src)
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

  # Fetch/select the llama.cpp build for the requested backend. The default is
  # the M4.2-patched Vulkan engine; --engine stage0 uses the stock prebuilt.
  local backend_for_bin="$BACKEND"
  case "$backend_for_bin" in auto) backend_for_bin="vulkan";; esac
  local bin_dir
  if [[ -n "$LLAMA_BIN_DIR" ]]; then
    bin_dir="$LLAMA_BIN_DIR"
    log "Using existing llama.cpp build at $bin_dir"
    if [[ "$ENGINE_MODE" == "m42" ]]; then
      if binary_is_m42_patched "$bin_dir/llama-server"; then
        ENGINE_PATCHED=1
      else
        warn "--llama-bin is not the M4.2-patched engine; the host-expert upload levers stay off.
  Build it with: BONGO_APPLY_M42_PATCH=1 tools/build-llama-vulkan.sh <tree> llama-server"
        ENGINE_PATCHED=0; M42_UPLOAD=0
      fi
    fi
  elif [[ "$backend_for_bin" == "vulkan" && "$ENGINE_MODE" == "m42" ]]; then
    local m42_bin_dir
    m42_bin_dir="$(m42_engine_bindir)"
    if [[ -x "$m42_bin_dir/llama-server" ]] && binary_is_m42_patched "$m42_bin_dir/llama-server"; then
      bin_dir="$m42_bin_dir"; ENGINE_PATCHED=1
      log "Using the cached M4.2 patched Vulkan engine at $bin_dir"
    elif (( DRY_RUN )); then
      bin_dir="$m42_bin_dir"; ENGINE_PATCHED=1
      log "[dry-run] would build the M4.2 patched Vulkan engine at $m42_bin_dir"
    elif bin_dir="$(build_m42_engine)"; then
      ENGINE_PATCHED=1
    else
      warn "Falling back to the stock Vulkan prebuilt; the M4.2 host-expert upload levers stay off.
  Build the patched engine with: BONGO_APPLY_M42_PATCH=1 tools/build-llama-vulkan.sh $(m42_engine_src) llama-server"
      ENGINE_PATCHED=0; M42_UPLOAD=0
      bin_dir="$(fetch_llama vulkan)"
    fi
  else
    bin_dir="$(fetch_llama "$backend_for_bin")"
  fi
  select_backend "$bin_dir"

  # Auto backend: the SYCL fallback needs the SYCL asset, because the Vulkan
  # asset that was just fetched cannot run on the SYCL backend (BAS-72).
  if [[ "$BACKEND" == "auto" && "$SELECTED_BACKEND" == "SYCL0" && -z "$LLAMA_BIN_DIR" ]]; then
    local sycl_dir
    sycl_dir="$(fetch_llama sycl)"
    SELECTED_BACKEND="SYCL0"
    SERVER_BIN="$sycl_dir/llama-server"
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

  # Hold the single-GPU lock across the server lifetime so two measurements
  # cannot overlap and thrash VRAM (BAS-80). A wrapper that already holds it
  # exports BONGO_GPU_LOCK_HELD=1, and this is a no-op.
  if declare -F bongo_gpu_lock_acquire >/dev/null 2>&1; then
    bongo_gpu_lock_acquire "bongo.sh backend=$BACKEND tier=$TIER port=$PORT" \
      || die "the single GPU is busy; another measurement holds the lock (BAS-80)."
    trap 'bongo_gpu_lock_release' EXIT INT TERM
  fi

  start_server_guarded || die "llama-server failed to start; see ${LOG_FILE:-the server log}."
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
