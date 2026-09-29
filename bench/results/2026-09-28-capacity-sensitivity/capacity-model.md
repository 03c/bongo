# Capacity model output

Reproduce: `python3 bench/capacity-model.py`

## Measured VRAM axis (from the sweep, unconstrained RAM)

| n-cpu-moe | GPU experts GiB | CPU experts GiB | 4K prompt | 4K output | 128K prompt | 128K output | SSD read B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 0.00 | 33.02 | 52.92 | 2.95 | 72.35 | 2.26 | 74078482432 |
| 24 | 16.83 | 16.19 | 186.50 | 11.25 | 114.46 | 7.53 | 2129502208 |
| 16 | 22.40 | 10.62 | 231.50 | 16.11 | 133.43 | 7.62 | 1642684416 |
| 12 | 25.02 | 8.00 | 260.44 | 17.71 | — | — | — |
| 0 | 33.02 | 0.00 | — | — | — | — | — |

## Fit

- ctx 4096: `t_ms = 51.23 + 0.0131 * B^2.86` (B = CPU expert GiB, SSE=0.3)
- ctx 131072: `t_ms = 131.11 + 0.0000 * B^7.34` (B = CPU expert GiB, SSE=0.0)

## Grid (modelled output tok/s)

### ctx 131072

| VRAM GiB | RAM GiB | GPU exp budget GiB | GPU exp resident GiB | n-cpu-moe | CPU exp GiB | RAM-safe | output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 12 | 16 | 5.14 | 5.04 | 41 | 27.98 | **no** | 4.48 |
| 12 | 32 | 5.14 | 5.04 | 41 | 27.98 | yes | 4.48 |
| 12 | 64 | 5.14 | 5.04 | 41 | 27.98 | yes | 4.48 |
| 24 | 16 | 17.14 | 16.83 | 24 | 16.19 | **no** | 7.53 |
| 24 | 32 | 17.14 | 16.83 | 24 | 16.19 | yes | 7.53 |
| 24 | 64 | 17.14 | 16.83 | 24 | 16.19 | yes | 7.53 |
| 32 | 16 | 22.4 | 22.4 | 16 | 10.62 | yes | 7.62 |
| 32 | 32 | 22.4 | 22.4 | 16 | 10.62 | yes | 7.62 |
| 32 | 64 | 22.4 | 22.4 | 16 | 10.62 | yes | 7.62 |

`RAM-safe` = `CPU expert GiB + 4 GiB overhead <= RAM`.  The modelled output tok/s does not depend on RAM inside the measured range; see the RAM section.

### ctx 4096

| VRAM GiB | RAM GiB | GPU exp budget GiB | GPU exp resident GiB | n-cpu-moe | CPU exp GiB | RAM-safe | output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 12 | 16 | 5.58 | 5.04 | 41 | 27.98 | **no** | 4.34 |
| 12 | 32 | 5.58 | 5.04 | 41 | 27.98 | yes | 4.34 |
| 12 | 64 | 5.58 | 5.04 | 41 | 27.98 | yes | 4.34 |
| 24 | 16 | 17.58 | 17.55 | 23 | 15.47 | **no** | 11.89 |
| 24 | 32 | 17.58 | 17.55 | 23 | 15.47 | yes | 11.89 |
| 24 | 64 | 17.58 | 17.55 | 23 | 15.47 | yes | 11.89 |
| 32 | 16 | 25.58 | 25.02 | 12 | 8.0 | yes | 17.79 |
| 32 | 32 | 25.58 | 25.02 | 12 | 8.0 | yes | 17.79 |
| 32 | 64 | 25.58 | 25.02 | 12 | 8.0 | yes | 17.79 |

`RAM-safe` = `CPU expert GiB + 4 GiB overhead <= RAM`.  The modelled output tok/s does not depend on RAM inside the measured range; see the RAM section.

## Leave-one-out error of the VRAM fit

| ctx | n-cpu-moe | measured | LOO modelled | rel err |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 12 | 17.71 | 17.97 | +1.5% |
| 4096 | 16 | 16.11 | 15.96 | -1.0% |
| 4096 | 24 | 11.25 | 11.55 | +2.6% |
| 4096 | 48 | 2.95 | 2.44 | -17.1% |
| 131072 | 16 | 7.62 | 13.41 | +75.9% |
| 131072 | 24 | 7.53 | 4.84 | -35.8% |
| 131072 | 48 | 2.26 | 7.37 | +225.6% |

## Measured RAM axis

| run | memory.max GiB | n-cpu-moe | CPU exp GiB | 4K output | 128K output | RAM peak GiB | VRAM peak GiB | SSD read_bytes | major faults |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mem16g-ncmoe16 | 16.0 | 16 | 10.62 | 15.61 | 8.00 | 10.89 | 29.26 | 1311301632 | 8096 |
| mem16g-ncmoe24 | 16.0 | 24 | 16.19 | 14.29 | 6.99 | 15.56 | 23.68 | 7340044288 | 62725 |
| uncapped-ncmoe16 | none | 16 | 10.62 | 15.42 | 7.93 | 10.53 | 29.26 | 1234382848 | 7501 |
| uncapped-ncmoe24 | none | 24 | 16.19 | 9.33 | 7.50 | 16.36 | 23.68 | 5734645760 | 56378 |


## Model error at measured anchors

| source | ctx | n-cpu-moe | RAM GiB | measured | modelled | rel err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sweep | 4096 | 12 | — | 17.71 | 17.79 | +0.5% |
| sweep | 4096 | 16 | — | 16.11 | 16.01 | -0.6% |
| sweep | 4096 | 24 | — | 11.25 | 11.27 | +0.2% |
| sweep | 4096 | 48 | — | 2.95 | 2.95 | -0.0% |
| sweep | 131072 | 16 | — | 7.62 | 7.62 | -0.0% |
| sweep | 131072 | 24 | — | 7.53 | 7.53 | +0.0% |
| sweep | 131072 | 48 | — | 2.26 | 2.26 | -0.0% |
| capacity | 4096 | 16 | 16.0 | 15.61 | 16.01 | +2.6% |
| capacity | 131072 | 16 | 16.0 | 8.00 | 7.62 | -4.7% |
| capacity | 4096 | 24 | 16.0 | 14.29 | 11.27 | -21.1% |
| capacity | 131072 | 24 | 16.0 | 6.99 | 7.53 | +7.8% |
| capacity | 4096 | 16 | none | 15.42 | 16.01 | +3.8% |
| capacity | 131072 | 16 | none | 7.93 | 7.62 | -3.8% |
| capacity | 4096 | 24 | none | 9.33 | 11.27 | +20.8% |
| capacity | 131072 | 24 | none | 7.50 | 7.53 | +0.5% |
