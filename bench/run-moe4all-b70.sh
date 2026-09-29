#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-TBD
#
# bench/run-moe4all-b70.sh — quant x context x MTP benchmark matrix for
# MoE4All/INFR on the Intel Arc Pro B70 (BAS-181, child of BAS-179).
#
# One case = one `infr` process = one measurement. `infr bench` times either
# prefill (`-n 0`) or decode (`-p 0`) per process, so a (quant, ctx) cell needs
# two cases. Every raw stderr/stdout and the exact command land under
# `bench/results/2026-09-29-moe4all-b70/raw/`.
#
# Usage:
#   bench/run-moe4all-b70.sh --list
#   bench/run-moe4all-b70.sh --all [--cold] [--force]
#   bench/run-moe4all-b70.sh --case <name> [--cold] [--force]
#
# `--all` skips a case whose raw artifact already exists (resume). `--cold`
# clears ~/.cache/infr/vk-pipeline-cache-* before the first case so that run is
# a genuine cold-shader run; later cases are warm.
#
# The GPU is single-tenant: every case holds `bench/gpu-lock.sh`. Stop any
# running `llama-server` first.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFR="${INFR_BIN:-$HOME/.local/share/MoE4All/target/release/infr}"
OUT="$ROOT/bench/results/2026-09-29-moe4all-b70"
RAW="$OUT/raw"
MTP_SIDECAR="${INFR_MTP_SIDECAR:-$HOME/.local/share/MoE4All-models/mtp/mtp-shared-Q4_K_M.gguf}"

IQ2XS="$HOME/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
Q2_0="$HOME/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
IQ3XS="$HOME/.bongo/models/ukisai-Swift-1.5-Qwen3.8-Flash-Next-GGUF/IQ3_XS/Swift-1.5-Qwen3.8-Flash-Next-IQ3_XS-00001-of-00003.gguf"

MTP_PROMPT="List the first ten prime numbers as a comma-separated list, then stop."
MTP_MAX_NEW=32

mkdir -p "$RAW"
# shellcheck source=bench/gpu-lock.sh
. "$ROOT/bench/gpu-lock.sh"

model_path() {
  case "$1" in
    iq2xs) printf '%s' "$IQ2XS" ;;
    q2_0)  printf '%s' "$Q2_0" ;;
    iq3xs) printf '%s' "$IQ3XS" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

# All benchmark cases: <name>|<verb>|<model>|<ctx>|<cache>|<depth>
#   verb `bench-pp`  -> prefill pp512, 3 reps
#   verb `bench-tg`  -> decode  tg128, 3 reps, depth 0
#   verb `bench-tg-d`-> decode  tg128, 3 reps, at `depth` (supplementary)
bench_cases() {
  cat <<'EOF'
iq2xs_ctx4096_cache16G_pp512_cold|bench-pp|iq2xs|4096|16GiB|0
iq2xs_ctx4096_cache16G_pp512|bench-pp|iq2xs|4096|16GiB|0
iq2xs_ctx4096_cache16G_tg128|bench-tg|iq2xs|4096|16GiB|0
iq2xs_ctx32768_cache16G_pp512|bench-pp|iq2xs|32768|16GiB|0
iq2xs_ctx32768_cache16G_tg128|bench-tg|iq2xs|32768|16GiB|0
q2_0_ctx4096_cache16G_pp512|bench-pp|q2_0|4096|16GiB|0
q2_0_ctx4096_cache16G_tg128|bench-tg|q2_0|4096|16GiB|0
q2_0_ctx32768_cache16G_pp512|bench-pp|q2_0|32768|16GiB|0
q2_0_ctx32768_cache16G_tg128|bench-tg|q2_0|32768|16GiB|0
iq3xs_ctx4096_cache16G_pp512|bench-pp|iq3xs|4096|16GiB|0
iq3xs_ctx4096_cache16G_tg128|bench-tg|iq3xs|4096|16GiB|0
iq3xs_ctx32768_cache16G_pp512|bench-pp|iq3xs|32768|16GiB|0
iq3xs_ctx32768_cache16G_tg128|bench-tg|iq3xs|32768|16GiB|0
iq2xs_ctx4096_cache22G_tg128|bench-tg|iq2xs|4096|22GiB|0
iq2xs_ctx4096_cache16G_d4096_tg128|bench-tg-d|iq2xs|4096|16GiB|4096
q2_0_ctx4096_cache16G_d4096_tg128|bench-tg-d|q2_0|4096|16GiB|4096
iq3xs_ctx4096_cache16G_d4096_tg128|bench-tg-d|iq3xs|4096|16GiB|4096
iq2xs_ctx4096_cache16G_d8192_tg128|bench-tg-d|iq2xs|4096|16GiB|8192
iq2xs_ctx32768_cache16G_d32768_tg128|bench-tg-d|iq2xs|32768|16GiB|32768
q2_0_ctx32768_cache16G_d32768_tg128|bench-tg-d|q2_0|32768|16GiB|32768
iq2xs_ctx32768_cache16G_pp32000|bench-pp32000|iq2xs|32768|16GiB|0
q2_0_ctx32768_cache16G_pp32000|bench-pp32000|q2_0|32768|16GiB|0
EOF
}

# MTP A/B cases: <name>|<ordinary|mtp>|<rep>
mtp_cases() {
  for rep in 1 2 3; do printf 'iq2xs_ctx4096_ordinary_rep%s|ordinary|%s\n' "$rep" "$rep"; done
  for rep in 1 2 3; do printf 'iq2xs_ctx4096_mtp_rep%s|mtp|%s\n' "$rep" "$rep"; done
}

list_cases() {
  echo "# bench cases"; bench_cases | awk -F'|' '{print $1}'
  echo "# mtp cases";   mtp_cases   | awk -F'|' '{print $1}'
}

run_bench_case() {
  local name="$1" verb="$2" model="$3" ctx="$4" cache="$5" depth="${6:-0}" cold="${7:-0}"
  local path; path="$(model_path "$model")" || return 1
  if [[ ! -f "$path" ]]; then echo "SKIP $name: model not present: $path" >&2; return 3; fi
  local p=512 n=0
  case "$verb" in
    bench-pp)   p=512 n=0 ;;
    bench-tg)   p=0 n=128 ;;
    bench-tg-d) p=0 n=128 ;;
    bench-pp32000) p=32000 n=0 ;;
    *) echo "unknown verb $verb" >&2; return 1 ;;
  esac
  local depth_args=()
  (( depth > 0 )) && depth_args=(-d "$depth")
  local cmd_env=(env INFR_NO_HOST_DMA=1 INFR_DEV=Vulkan1)
  local args=(bench "$path" -p "$p" -n "$n" "${depth_args[@]}" -r 3 --ctx "$ctx" --dev Vulkan1 -u 512 --set "paging.cache=$cache" --json)

  if (( cold == 1 )); then
    { echo "cold: clearing ~/.cache/infr/vk-pipeline-cache-* before $name"; } > "$RAW/$name.log"
    rm -f "$HOME"/.cache/infr/vk-pipeline-cache-* 2>/dev/null || true
  else
    : > "$RAW/$name.log"
  fi
  { printf '%s %s' "${cmd_env[*]}" "$INFR"; printf ' %q' "${args[@]}"; printf '\n'; } | tee "$RAW/$name.cmd" >> "$RAW/commands.txt"

  echo "RUN $name  ($model ctx=$ctx cache=$cache depth=$depth)  $(date -u +%FT%TZ)"
  local t0 t1 rc
  t0=$(date +%s)
  "${cmd_env[@]}" "$INFR" "${args[@]}" >"$RAW/$name.out" 2>>"$RAW/$name.log"
  rc=$?
  t1=$(date +%s)
  grep -E '^\[\{' "$RAW/$name.out" > "$RAW/$name.json" 2>/dev/null || : > "$RAW/$name.json"
  printf '{"case":"%s","kind":"bench","model":"%s","ctx":%s,"cache":"%s","depth":%s,"prefill":%s,"decode":%s,"reps":3,"exit":%s,"wall_secs":%s,"json":%s}\n' \
    "$name" "$model" "$ctx" "$cache" "$depth" "$p" "$n" "$rc" "$((t1 - t0))" "$(cat "$RAW/$name.json")" > "$RAW/$name.meta.json"
  echo "  exit=$rc wall=$((t1 - t0))s  json=$(cat "$RAW/$name.json")"
  return $rc
}

run_mtp_case() {
  local name="$1" mode="$2" rep="${3:-1}"
  if [[ ! -f "$IQ2XS" ]]; then echo "SKIP $name: $IQ2XS missing" >&2; return 3; fi
  if [[ ! -f "$MTP_SIDECAR" ]]; then echo "SKIP $name: sidecar $MTP_SIDECAR missing" >&2; return 3; fi

  local mtp_env=(env INFR_NO_HOST_DMA=1 INFR_DEV=Vulkan1)
  if [[ "$mode" == "mtp" ]]; then
    mtp_env+=(INFR_MTP=1 "INFR_SPEC_DRAFT=$MTP_SIDECAR" INFR_PAGER_PROFILE=1)
  fi
  local args=(run "$IQ2XS" "$MTP_PROMPT" --max-new "$MTP_MAX_NEW" --temp 0 --no-think --ctx 4096 --dev Vulkan1 -u 512 --set paging.cache=16GiB)

  : > "$RAW/$name.log"
  { printf '%s %s' "${mtp_env[*]}" "$INFR"; printf ' %q' "${args[@]}"; printf '\n'; } | tee "$RAW/$name.cmd" >> "$RAW/commands.txt"

  echo "RUN $name  (mode=$mode rep=$rep)  $(date -u +%FT%TZ)"
  local t0 t1 rc
  t0=$(date +%s)
  "${mtp_env[@]}" "$INFR" "${args[@]}" >"$RAW/$name.out" 2>>"$RAW/$name.log"
  rc=$?
  t1=$(date +%s)
  local alpha
  alpha=$(grep -oE 'alpha=[0-9.]+' "$RAW/$name.log" | tail -1 | cut -d= -f2)
  printf '{"case":"%s","kind":"mtp-ab","mode":"%s","rep":%s,"exit":%s,"wall_secs":%s,"alpha":%s,"prompt":"%s","max_new":%s}\n' \
    "$name" "$mode" "$rep" "$rc" "$((t1 - t0))" "${alpha:-null}" "$MTP_PROMPT" "$MTP_MAX_NEW" > "$RAW/$name.meta.json"
  echo "  exit=$rc wall=$((t1 - t0))s alpha=${alpha:-n/a}"
  return $rc
}

# Hold the single-GPU lock for the whole measured case.
locked() {
  bongo_gpu_lock_acquire "$1" || { echo "gpu-lock: busy, giving up" >&2; return 3; }
  shift
  "$@"
  local rc=$?
  bongo_gpu_lock_release
  return $rc
}

run_named_case() {
  local name="$1" force="$2" cold="$3"
  local line
  line=$(bench_cases | awk -F'|' -v n="$name" '$1==n{print; exit}')
  if [[ -n "$line" ]]; then
    local -a f; IFS='|' read -r -a f <<< "$line"
    if [[ "$force" != "1" && -s "$RAW/$name.json" ]]; then echo "SKIP $name (done)"; return 0; fi
    locked "$name" run_bench_case "${f[@]}" "$cold"
    return $?
  fi
  line=$(mtp_cases | awk -F'|' -v n="$name" '$1==n{print; exit}')
  if [[ -n "$line" ]]; then
    local -a f; IFS='|' read -r -a f <<< "$line"
    if [[ "$force" != "1" && -s "$RAW/$name.meta.json" ]]; then echo "SKIP $name (done)"; return 0; fi
    locked "$name" run_mtp_case "${f[@]}"
    return $?
  fi
  echo "unknown case: $name" >&2
  return 2
}

main() {
  local mode="all" name="" force=0 cold=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --list) list_cases; return 0 ;;
      --all) mode=all; shift ;;
      --case) mode=case; name="${2:?}"; shift 2 ;;
      --force) force=1; shift ;;
      --cold) cold=1; shift ;;
      -h|--help) sed -n '2,26p' "$0"; return 0 ;;
      *) echo "unknown arg: $1" >&2; return 2 ;;
    esac
  done

  local rc_total=0
  if [[ "$mode" == "case" ]]; then
    run_named_case "$name" "$force" "$cold"; rc_total=$?
  else
    local first=1
    while IFS= read -r line; do
      IFS='|' read -r -a f <<< "$line"
      local c=0
      [[ $first -eq 1 && $cold -eq 1 ]] && c=1
      if [[ "$force" != "1" && -s "$RAW/${f[0]}.json" ]]; then echo "SKIP ${f[0]} (done)"; first=0; continue; fi
      locked "${f[0]}" run_bench_case "${f[@]}" "$c" || rc_total=$?
      first=0
    done < <(bench_cases)
    while IFS= read -r line; do
      IFS='|' read -r -a f <<< "$line"
      if [[ "$force" != "1" && -s "$RAW/${f[0]}.meta.json" ]]; then echo "SKIP ${f[0]} (done)"; continue; fi
      locked "${f[0]}" run_mtp_case "${f[@]}" || rc_total=$?
    done < <(mtp_cases)
  fi
  echo "matrix rc=$rc_total"
  return $rc_total
}

main "$@"
