# bongo baseline benchmark — iq2_xs

Generated: `2026-09-28T12:10:56+00:00`  
Endpoint: `http://127.0.0.1:8080/v1`  
Model: `bongo-iq2_xs`  
Harness: `1.0.0`  
Command: `bench/run.sh --repo-root . --contexts 262144 --needle-context 262144 --repeats 1 --repeats-deep 1 --deep-threshold 131072 --max-tokens 32 --needle-tokens 32 --timeout 3600 --tier iq2_xs --out-dir bench/results/2026-09-28-ctx256-full --hash-mode sampled --skip-error-cases --server-pid 2301829`

## Verdict

- all measured contexts OK: **True**
- highest context that worked: **262144**
- 128K needle: **pass**
- note: server reported n_ctx=262144

## Run configuration

- contexts: `[262144]`  
- repeats: `1`  
- repeats at >= 131072 tokens: `1`  
- max_tokens: `32`  
- TTFT stream tokens: `8`  
- server n_ctx: `262144`  
- shard hash mode: `sampled`  

## Results

`prompt tok/s` is prefill throughput, `output tok/s` is decode throughput, `TTFT` is time to first streamed token. Values are the median of the repeats; `cv` is the coefficient of variation (stdev/median).

| context | prompt tokens | prompt tok/s | output tok/s | TTFT ms | prefill ms | repeats | cv(ttft) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262144 | 261997 | 84.788 | 4.494 | 3090275.3 | 3090030.1 | 1 | n/a |

## Memory

- peak VRAM: **30.92 GiB** (method: `fdinfo`)
- peak system RAM (process RSS): **10.65 GiB** (method: `proc_status_sum`)
- process RSS high-water mark: 24.15 GiB

## 128K needle

- status: **pass**
- prompt tokens: 261997
- expected token: `VAULT-COORD-7391-QXZ`
- answer: `<think>

</think>

VAULT-COORD-7391-QXZ.
 service.  The mountain road curved past the reservoir, where`

## Machine spec

- uname: `Linux localhost.localdomain 7.0.13-200.fc44.x86_64 #1 SMP PREEMPT_DYNAMIC Fri Jun 19 22:51:30 UTC 2026 x86_64 GNU/Linux`
- CPU: `AMD Ryzen 7 9700X 8-Core Processor`
- memory: `Mem:     32699588608  7783993344   338354176   618270720 25668321280 24915595264`
- display devices:

  ```
  03:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Arc Pro B70] [8086:e223]
  12:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Granite Ridge [Radeon Graphics] [1002:13c0] (rev c5)
  ```
- DRM drivers: `{'card0': 'xe', 'card1': 'amdgpu'}`
- modules: `{'xe': {'version': None, 'refcnt': '4'}, 'amdgpu': {'version': None, 'refcnt': '1'}}`
- DRI nodes: `['/dev/dri/by-path', '/dev/dri/card0', '/dev/dri/card1', '/dev/dri/renderD128', '/dev/dri/renderD129']`

## Server

- pids: `[2301829]`
- llama.cpp build: `b11223-4da633776`
- flags:

  ```
  /home/cchild/.bongo/llama/b11223/vulkan/llama-server --model /home/cchild/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ctx-size 262144 --jinja --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 18 --host 127.0.0.1 --port 8080 --parallel 1 --alias bongo-iq2_xs --metrics --device Vulkan1 --slot-save-path /tmp/paperclip-run-bas-78-e83a5acc-7d7-edJLsZ/slot 
  ```

## Model / shards

- gguf dir: `/home/cchild/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs`

| shard | size | sha256 |
| --- | ---: | --- |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 37.06 GiB | `9117886e27aae30d5993d5966b54da2da3c6c6a3ba2826d1ea04c82a5503ee6e` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` | 26.42 GiB | `0ecf15c38ca003c29e069483b0f3b7065f4542f0d9abc4b1c06f5b35df6e08b5` |

## Error / malformed-prompt cases

| case | HTTP | error status | excerpt |
| --- | ---: | --- | --- |

