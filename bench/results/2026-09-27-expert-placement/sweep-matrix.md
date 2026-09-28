# Expert placement sweep — Arc Pro B70 / IQ2_XS

`--n-cpu-moe N` keeps the routed experts of the first N layers on the CPU;
layers N..47 keep their experts on the GPU. One repeat per context.

`fit` means the config loaded **and** every measured context served OK.
`fit_by_context` shows which contexts passed. A failed context still records
the peak VRAM reached before the failure.

## Summary

| n-cpu-moe | loaded | fit | GPU experts GiB | CPU experts GiB | VRAM after load GiB | 4096 prompt tok/s | 4096 output tok/s | 4096 VRAM GiB | 4096 RAM GiB | 131072 prompt tok/s | 131072 output tok/s | 131072 VRAM GiB | 131072 RAM GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **no** | **no (load OOM)** | 33.02 | 0.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 12 | yes | **no (131072 failed)** | 25.02 | 8.00 | 31.37 | 260.438 | 17.707 | 31.45 | 6.68 | n/a | n/a | 31.85 | 7.12 |
| 16 | yes | yes | 22.40 | 10.62 | 28.74 | 231.502 | 16.114 | 28.82 | 9.16 | 133.431 | 7.623 | 29.26 | 10.96 |
| 24 | yes | yes | 16.83 | 16.19 | 23.17 | 186.504 | 11.253 | 23.25 | 14.14 | 114.464 | 7.532 | 23.68 | 16.53 |

## CPU-expert share and marginal 128K throughput

Only configs that served 128K are comparable. The marginal column is
`d(output tok/s at 128K) / d(GPU expert GiB)` between adjacent measured
configs, i.e. how many output tok/s each extra GiB of GPU-resident experts buys.

| n-cpu-moe | CPU expert share | GPU experts GiB | 128K output tok/s | marginal tok/s per GPU GiB |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 33.02 | n/a | n/a |
| 12 | 0.242 | 25.02 | n/a | n/a |
| 16 | 0.322 | 22.40 | 7.623 | n/a |
| 24 | 0.490 | 16.83 | 7.532 | -0.0163 |

