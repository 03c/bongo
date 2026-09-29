# M3.6 — the warm-prefix (cached-turn) profile and the dominant cost

Work for [BAS-130](/BAS/issues/BAS-130) (M3.6), part of [BAS-62](/BAS/issues/BAS-62).
Engine: llama.cpp `4da6337767f973e2b4d0797e5b323d77d8565e4a` (`b11223`), Vulkan
device `Vulkan1`, tier `iq2_xs`, q8 KV, Stage 0 placement `--n-cpu-moe 16`,
`--flash-attn on`, `--ctx-size 131072`, `--parallel 1`.
Author: Coder (Paperclip). Date: 2026-09-29.

Raw files: [`bench/results/2026-09-28-warm-prefix-profile/`](../../bench/results/2026-09-28-warm-prefix-profile/).
Reproduce with [`bench/run-warm-prefix-profile.sh`](../../bench/run-warm-prefix-profile.sh)
(one command; holds the single-GPU flock, BAS-80). The numbers here are single
measurements per config (no repeats); the ~20/80 GPU/host split is far larger
than the run-to-run noise.

## TL;DR

The 512-token warm-prefix delta turn is **host/CPU-bound, not GPU-kernel-bound**.
At 16K the Vulkan kernels are busy **793 ms of a 3983 ms turn (20%)**; the other
**80% is CPU work and host/GPU synchronisation**. The single largest GPU op is
the flash attention over the cached KV (356 ms, 9% of the turn); the MoE expert
matmul — the target of every planned kernel milestone — is only 135 ms (3.4%).
The `--ubatch-size 128` ablation makes the turn **2.3x slower**, which proves a
large fixed per-batch host/CPU cost. **None of the planned GPU-kernel levers
(kernel / speculation / PLE) moves the turn target; placement is the only
planned lever that does, and it is VRAM-capped at roughly the +18% already
measured.**

## Method

One server restart per ablation, **exactly one flag changed** vs the pinned
Stage 0 baseline. For each config the profiler primes a prefix, then measures
the delta turn against the cached KV:

| case | prompt | cache_prompt | what it is |
| --- | --- | --- | --- |
| `cold` | P | false | the one full prefill that primes the slot |
| `hit` | P | true | full reuse; ~4 new tokens = fixed per-request cost |
| `grow_D` | P+D | true | **the measured delta turn**, D new tokens |
| `decode` | P | true, `max_tokens=32` | decode rate at context P after a hit |

The component attribution has two independent legs:

1. **Ablation (wall clock).** Take one component off the fast GPU onto the slow
   CPU and measure how much the delta turn changes. A large positive change
   names a component that contributes to the critical path; a change near zero
   names one hidden behind other work. `--ubatch-size` probes the per-batch
   fixed cost.
2. **Per-op GPU time.** A `GGML_VK_PERF_LOGGER=1` config records device-side
   timestamps for every Vulkan op; `bench/parse-vk-perf.py` groups them into
   MoE experts (`MUL_MAT_ID`), flash attention (`FLASH_ATTN_EXT`), dense
   (`MUL_MAT`), GatedDeltaNet/SSM, and norms. `GPU busy` (`Total time`) against
   the wall time splits the turn into GPU and non-GPU.

The per-op dump is supporting evidence, not the sole method: the Vulkan logger
rows do not carry tensor names, so the ablations are what attribute a class to a
lever.

## Result 1 — the ranked 16K delta-turn cost table

Shipped `--n-cpu-moe 16` baseline, prefix 16384, delta 510 tokens,
wall `prompt_ms` = **3982.5 ms** (the shipped target is <=3000 ms, so 33% over).
GPU busy is measured with the perf logger (whose own overhead is +1.8% on the
wall); the non-GPU row is wall minus GPU busy.

| rank | component | ms | % of turn | evidence |
| --: | --- | --: | --: | --- |
| 1 | **host/CPU work + GPU sync (non-GPU)** | **3190** | **80.1%** | wall minus Vulkan GPU busy |
| 2 | GPU flash attention over the cached KV | 356 | 8.9% | `FLASH_ATTN_EXT` |
| 3 | GPU dense / non-MoE matmuls | 206 | 5.2% | `MUL_MAT` (attn/DeltaNet projections, hyper-connections) |
| 4 | GPU MoE expert matmuls | 135 | 3.4% | `MUL_MAT_ID` (32 GPU layers only) |
| 5 | GPU GatedDeltaNet (recurrent) ops | 31 | 0.8% | `GATED_DELTA_NET`, `SSM_CONV` |
| 6 | GPU norms / activations | 27 | 0.7% | `RMS_NORM`, `SILU`, … |
| 7 | GPU memory layout / elementwise / other | 38 | 1.0% | `CPY`, `CONT`, `PERMUTE`, … |

Supporting measurements:

- `hit` (4 tokens at 16K) = 231 ms wall, 145 ms GPU busy -> ~114 ms is fixed
  per-request serving/host overhead (**2.9%** of the turn). So serving and
  tokenisation are *not* the dominant cost.
- Cold 16K prefill = 84 633 ms wall, 27 410 ms GPU busy (32% GPU) — the same
  kernels at a large batch are more GPU-efficient; the delta turn is not.
- The per-op breakdown of the 16K cold prefill and the decode request is in
  `baseline_perf/vk-perf-task*.json`.

## Result 2 — the ablation ranking (the proof)

16K, delta 512, wall `prompt_ms`, vs the shipped baseline (3982.5 ms):

| rank | config | change | delta-turn ms | Δ | proves |
| --: | --- | --- | --: | --: | --- |
| 1 | `ub128` | `--ubatch-size 128` (4 batches for 512 tokens) | 9123.7 | **+129.1%** | large fixed per-batch host/CPU cost |
| 2 | `attn_cpu` | attention + DeltaNet QKV/output projections on CPU | 5594.2 | +40.5% | those projections matter, but they run on GPU in the baseline |
| 3 | `hc_cpu` | hyper-connection tensors (`hc_*`) on CPU | 5248.1 | +31.8% | the BF16 hyper-connection matmuls matter |
| 4 | `ncmoe24` | 8 more MoE layers' experts on CPU | 4711.8 | +18.3% | CPU-resident expert compute is on the critical path |
| 5 | `ub1024` | `--ubatch-size 1024` | 4170.5 | +4.7% | 512 is already near the batch sweet spot |
| 6 | `ssm_cpu` | GatedDeltaNet weights on CPU | 4060.4 | +2.0% | recurrent layers are not a bottleneck |
| 7 | `baseline_perf` | + perf logger | 4054.8 | +1.8% | profiler overhead (control) |

`ncmoe8` (more experts on the GPU) and `fa_off` (`--flash-attn off`) both
**failed to load at ctx 131072** (VRAM), which is itself a result: the GPU cannot
hold more experts than the Stage 0 placement already does, so the placement
lever is at its VRAM edge.

**The proving ablation is `ub128`.** If the turn were GPU-kernel-bound, splitting
512 tokens into four batches would change little (the same work). It instead
costs **+5.1 s (+129%)**, so the turn is dominated by work paid *per batch* —
CPU-side expert dispatch, host/CPU segment transitions, and synchronisation.
Fitting `p + 4c = 9123.7`, `p + c = 3982.5` gives a fixed per-batch cost
**c ≈ 1.71 s** and a per-token cost of ~4.4 ms/token: nearly half the turn is a
fixed per-batch cost.

## Result 3 — the ~128K delta turn

Shipped baseline, prefix 127 999, delta 516 tokens, wall **5899.5 ms** (target
<=5000 ms, 18% over). Same engine/flags.

| component | ms | % of turn | basis |
| --- | --: | --: | --- |
| delta compute (prefix-independent) | 3982 | 67.5% | measured at 16K; 80% of it host/CPU |
| prefix-attention increment | 1917 | 32.5% | measured: 128K wall − 16K wall |
| **total** | **5900** | 100% | measured |

The only thing that changed between the two points is the cached-KV length
(16 384 -> 127 999), so the +1917 ms is the extra cost of attending over a 7.8x
longer cache. It is likely almost all GPU flash attention (the 16K attention op
is 356 ms), which would put the GPU share at ~46% and non-GPU at ~54% at 128K —
but the non-GPU delta-compute term (the ~3.2 s CPU/host term) is still the
single largest term. That 128K GPU split is **derived, not separately
profiled**; a `baseline_perf_128k` config is committed to measure it directly
(one 128K prime, ~16 min).

For reference, the full prefix sweep (wall `prompt_ms`, single run each) is:

| prefix | hit | grow 128 | grow 512 | grow 1024 | decode tok/s |
| --: | --: | --: | --: | --: | --: |
| 16384 | 231.3 | 3947.6 | 3982.5 | 6861.5 | 11.3 |
| 127999 | 404.6 | 4381.5 | 5899.5 | 10390.9 | 7.7 |

`grow 128 ≈ grow 512` at both contexts is the same fixed-per-batch cost visible
from the other direction: four times fewer new tokens costs almost nothing.

## Decision note — which planned lever moves the target

| lever | measured effect on the turn | verdict |
| --- | --- | --- |
| **Kernel** (integer MMQ/MMVQ for i-quant experts) | targets GPU MoE experts, 135 ms = 3.4% of the turn; M3.1 measured ~1.05x | **does not move the target** |
| **Placement** (`--n-cpu-moe`, `-ot` byte budget) | `ncmoe24` costs +18.3%; `ncmoe8` cannot load at 128K | moves it **by at most ~18%**, and is VRAM-capped |
| **Speculation** (suffix/n-gram) | decode-side only; the turn metric is a 512-token *prefill* TTFT | **does not move TTFT** |
| **PLE reader** | engine A/B neutral (ADR-0004) | **does not move the target** |
| host/CPU critical path (not in the planned set) | ~80% of the turn; `ub128` shows ~1.7 s of fixed per-batch cost | **the only term large enough to close the gap** |

**Recommendation.** Do not fund more GPU-kernel work against the cached-turn
target: the whole MoE expert matmul is 3.4% of the turn, so even a perfect MMQ
cannot pay for itself. The turn is a host/CPU scheduling problem. The next
milestone should attack the fixed per-batch cost: keep as many experts on the
GPU as VRAM allows (placement), reduce the number of CPU/GPU segment transitions
per batch, and measure why a 512-token batch pays ~1.7 s of fixed cost that a
35 000-token cold prefill (32 such batches) largely amortises. A `--n-cpu-moe 0`
build or a larger-VRAM card is the decisive experiment; neither is in the
current plan. Target check: 16K needs a 33% cut and 128K an 18% cut, so even the
full placement lever (+18%) closes only the 128K gap, not the 16K one.

## Reproduce

```sh
# shipped baseline + all 16K ablations + the per-op config, holding the GPU lock
bench/run-warm-prefix-profile.sh

# analysed tables
bench/analyze-warm-prefix.py --root bench/results/2026-09-28-warm-prefix-profile --prefix 16384

# per-op GPU split for one request (task ids are in the server log's `launch_slot_` lines)
bench/parse-vk-perf.py bench/results/2026-09-28-warm-prefix-profile/baseline_perf/llama-server.log \
    --task 24 --summary

# the 128K per-op split (not run here; ~16 min of GPU)
BONGO_PROFILE_CONFIGS=baseline_perf_128k bench/run-warm-prefix-profile.sh
```

Raw outputs per config: `profile.json`, `server-flags.json`, `llama-server.log`,
plus `vk-perf-task*.json` for the parsed per-op breakdown.

## Caveats

- Single measurement per config (no repeats); load failures for `ncmoe8` and
  `fa_off` are recorded in their `profile.json`.
- The Vulkan `Total time` is device-side GPU-busy time; it excludes host work and
  GPU idle time by construction, which is the point of the split.
- The 128K GPU/non-GPU split is derived from the measured 16K split plus the
  measured 128K prefix increment; the committed `baseline_perf_128k` config
  measures it directly.
