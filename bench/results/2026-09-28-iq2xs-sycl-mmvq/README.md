# 2026-09-28 — IQ2_XS SYCL integer matmul (M3.1, BAS-74)

Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`), SYCL backend, built with
the oneAPI 2025.3 container (`ghcr.io/ggml-org/llama.cpp:server-intel`), `GGML_SYCL=ON`,
`GGML_SYCL_F16=ON`. GPU: Intel Arc Pro B70 (32 GiB), Level Zero.

Patch under test: [`tools/patches/iq2xs-sycl-integer-mmvq.patch`](../../../tools/patches/iq2xs-sycl-integer-mmvq.patch)
(multi-column MMVQ for IQ2_XS; does not touch the prefill MMQ gate).

## Files

| file | what |
| --- | --- |
| `correctness-iq2-xs.txt` | `test-backend-ops test -b SYCL0 -o MUL_MAT -p iq2_xs` — 14/14 pass |
| `mulmat-perf-patched.txt` | `test-backend-ops perf` for the patched build |
| `mulmat-perf-stock.txt` | same, unpatched `b11223` |

## Caveat — read this before quoting a number

The reference box was **GPU-saturated by a parallel M-series run** (a 256K Vulkan server holding
~26.5 GiB of the 32 GiB VRAM), so the two `perf` runs did not see equal host load. The `n=512`
row is the tell: the patch does not touch that path, yet patched vs stock differ ~6x, so the
end-to-end timings are not a clean A/B. Only the small-`n` direction (multi-column > per-column)
is meaningful, and even that must be re-measured in a quiet window.

Reproduce:

```sh
# patched
tools/build-llama-sycl.sh ~/.bongo/engine/llama.cpp-pin "test-backend-ops"
docker run --rm --entrypoint sh --device /dev/dri --user "$(id -u):$(id -g)" \
  -v "$HOME/.bongo/engine:/work" -w /work/llama.cpp-pin \
  ghcr.io/ggml-org/llama.cpp:server-intel -c \
  './build-sycl/bin/test-backend-ops perf -b SYCL0 -o MUL_MAT -p "iq2_xs"'
```

Full-model prompt/decode A/B and the cached-turn metric are still to be run; see
[`docs/research/iq2xs-sycl-integer-mmvq.md`](../../../docs/research/iq2xs-sycl-integer-mmvq.md) §5.
