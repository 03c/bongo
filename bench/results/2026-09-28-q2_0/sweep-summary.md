# Q2_0 `--n-cpu-moe` sweep — 128K feasibility on the Arc Pro B70

Tier `q2_0`, llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`), Vulkan (`--device Vulkan1`).
Each config: server restart, discarded 4K warm-up, harness at 4096 and 131072 (1 repeat).
`fit` means the config loaded, the 128K context served a 200, and the needle passed.

| n-cpu-moe | loads | fits 128K | GPU experts GiB | CPU experts GiB | VRAM after load GiB | 4K prompt | 4K output | 128K prompt | 128K output | peak VRAM GiB | needle |
| ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | yes | **no** | 25.05 | 6.59 | 31.34 | 296.89 | 11.19 |  |  | 31.85 | fail |
| 11 | yes | yes | 24.39 | 7.25 | 30.68 | 288.88 | 12.38 | 151.40 | 7.44 | 31.20 | pass |
| 12 | yes | yes | 23.73 | 7.91 | 30.02 | 279.03 | 8.81 | 149.74 | 7.63 | 30.54 | pass |
| 15 | yes | yes | 21.75 | 9.89 | 28.04 | 256.34 | 9.82 | 141.61 | 6.81 | 28.56 | pass |
| 16 | yes | yes | 21.09 | 10.55 | 27.38 | 247.92 | 6.97 | 139.02 | 6.91 | 27.90 | pass |

**Smallest fitting `--n-cpu-moe` = 11.** `n=10` loses the device at 128K
(`vk::Queue::submit: ErrorDeviceLost`); its 4K result is from before the failure.
`n=12`'s harness run was interrupted during the pre-fix unbounded `negative_max_tokens`
error case; its 4K/128K/needle raw results are complete (`sweep/ncmoe-12/raw/`).

See [`RECOMMENDATION.md`](RECOMMENDATION.md) for the full comparison.
