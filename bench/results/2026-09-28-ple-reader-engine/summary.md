# M3.4b PLE reader engine A/B

| config | ctx | prefill? | prompt tok/s off | prompt tok/s on | prefill delta | decode tok/s off | decode tok/s on | TTFT off (ms) | TTFT on (ms) | VRAM peak (GiB) | RSS peak (GiB) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4096 | cache-hit | 1.0 | 11.6 | n/a (cache-hit) | 15.48 | 15.75 | 3137 | 313 | 29.07 | 9.19 |
| baseline | 131072 | yes | 132.4 | 133.1 | +0.5% | 7.69 | 7.76 | 957797 | 952784 | 29.51 | 11.34 |
| m33 | 4096 | cache-hit | 12.5 | 13.2 | n/a (cache-hit) | 16.41 | 16.65 | 295 | 281 | 29.05 | 9.87 |
| m33 | 131072 | yes | 136.5 | 135.6 | -0.7% | 7.57 | 7.64 | 929006 | 935381 | 29.29 | 11.78 |

## Acceptance

- **verdict: `fail-no-gain`** (4 off/on pair(s) compared, tolerance -5.0%)
- engine revision: `4da633776`; PLE table: 26.82 GiB
- >5.0% regressions: 0
- prefill gains: 0
- cache-hit rows excluded from the prefill comparison (their prompt tok/s is a continuation rate, not a prefill):
  - baseline @4096: prompt_n=3 cache_n=4094, prompt tok/s 0.97 -> 11.62
  - m33 @4096: prompt_n=3 cache_n=4094, prompt tok/s 12.55 -> 13.17
- RSS baseline @4096: off 8.95 GiB -> on 9.19 GiB (growth 0.24 GiB, table 26.82 GiB)
- RSS baseline @131072: off 10.95 GiB -> on 11.34 GiB (growth 0.39 GiB, table 26.82 GiB)
- RSS m33 @4096: off 10.54 GiB -> on 9.87 GiB (growth -0.68 GiB, table 26.82 GiB)
- RSS m33 @131072: off 12.56 GiB -> on 11.78 GiB (growth -0.79 GiB, table 26.82 GiB)

## Provenance

- `baseline-off`: tier `iq2_xs`, ctx `131072`, n-cpu-moe `16`, cache `q8_0/q8_0`, flash-attn `on`, device `Vulkan1`, --ple-reader `off`
- `baseline-on`: tier `iq2_xs`, ctx `131072`, n-cpu-moe `16`, cache `q8_0/q8_0`, flash-attn `on`, device `Vulkan1`, --ple-reader `on`
- `m33-off`: tier `iq2_xs`, ctx `131072`, n-cpu-moe `0`, cache `q8_0/q8_0`, flash-attn `on`, device `Vulkan1`, --ple-reader `off`
- `m33-on`: tier `iq2_xs`, ctx `131072`, n-cpu-moe `0`, cache `q8_0/q8_0`, flash-attn `on`, device `Vulkan1`, --ple-reader `on`
