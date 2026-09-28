# bongo baseline benchmark — iq2_xs

Generated: `2026-09-28T15:33:53+00:00`  
Endpoint: `http://127.0.0.1:8090/v1`  
Model: `bongo-iq2_xs`  
Harness: `1.0.0`  
Command: `bench/run.sh --repo-root /home/cchild/.paperclip/instances/default/projects/c1ccf5ff-cc7d-4b68-a23c-def2d29bb0a3/15f9f0b7-0021-4164-ae30-902276def736/bongo/.paperclip/worktrees/BAS-62-improve-speed-architecture --tier iq2_xs --contexts 4096,131072 --base-url http://127.0.0.1:8090/v1 --repeats 1 --max-tokens 128 --needle-context 131072 --hash-mode none --server-pid 2652162 --out-dir bench/results/2026-09-28-ple-reader-engine/baseline-on`

## Verdict

- all measured contexts OK: **True**
- highest context that worked: **131072**
- 128K needle: **pass**
- note: server reported n_ctx=131072

## Run configuration

- contexts: `[4096, 131072]`  
- repeats: `1`  
- max_tokens: `128`  
- TTFT stream tokens: `8`  
- server n_ctx: `131072`  
- prompt cache: `cache_prompt=True` (profile `agentic`)  
- shard hash mode: `none`  

## Results

`prompt tok/s` is prefill throughput, `output tok/s` is decode throughput, `TTFT` is time to first streamed token. Values are the median of the repeats; `cv` is the coefficient of variation (stdev/median). `cached tok` is the median number of prompt tokens the server reused from the slot KV (`timings.cache_n` / `usage.prompt_tokens_details.cached_tokens`).

| context | prompt tokens | cached tok | prompt tok/s | output tok/s | TTFT ms | prefill ms | repeats | cv(ttft) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 3 | 4094 | 11.615 | 15.748 | 312.8 | 258.3 | 1 | n/a |
| 131072 | 126767 | 4094 | 133.078 | 7.759 | 952784.2 | 952579.0 | 1 | n/a |

## Memory

- peak VRAM: **29.51 GiB** (method: `fdinfo`)
- peak system RAM (process RSS): **11.34 GiB** (method: `proc_status_sum`)
- process RSS high-water mark: 16.66 GiB

## 128K needle

- status: **pass**
- prompt tokens: 126767
- expected token: `VAULT-COORD-7391-QXZ`
- answer: `<think>

</think>

VAULT-COORD-7391-QXZ

The secret access code for the vault is VAULT-COORD-7391-QXZ.
allow channel for movement. The turbine hall hummed at a steady frequency, a sound felt more th`

## Machine spec

- uname: `Linux localhost.localdomain 7.0.13-200.fc44.x86_64 #1 SMP PREEMPT_DYNAMIC Fri Jun 19 22:51:30 UTC 2026 x86_64 GNU/Linux`
- CPU: `AMD Ryzen 7 9700X 8-Core Processor`
- memory: `Mem:     32699588608  6661079040   346849280    76967936 26287677440 26038509568`
- display devices:

  ```
  03:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Arc Pro B70] [8086:e223]
  12:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Granite Ridge [Radeon Graphics] [1002:13c0] (rev c5)
  ```
- DRM drivers: `{'card0': 'xe', 'card1': 'amdgpu'}`
- modules: `{'xe': {'version': None, 'refcnt': '4'}, 'amdgpu': {'version': None, 'refcnt': '1'}}`
- DRI nodes: `['/dev/dri/by-path', '/dev/dri/card0', '/dev/dri/card1', '/dev/dri/renderD128', '/dev/dri/renderD129']`

## Server

- pids: `[2652162]`
- llama.cpp build: `b1-4da633776`
- flags:

  ```
  /home/cchild/.bongo/engine/llama.cpp-ple/build-vulkan/bin/llama-server --model /home/cchild/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ctx-size 131072 --jinja --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 16 --host 127.0.0.1 --port 8090 --parallel 1 --alias bongo-iq2_xs --metrics --device Vulkan1 --ple-reader on 
  ```

## Model / shards

- gguf dir: `/home/cchild/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs`

| shard | size | sha256 |
| --- | ---: | --- |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 37.06 GiB | `None` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` | 26.42 GiB | `None` |

## Error / malformed-prompt cases

| case | HTTP | error status | excerpt |
| --- | ---: | --- | --- |
| empty_prompt | 200 | False | `{"choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"length"}],"created":1790609600,"model":"bongo-iq2_xs",` |
| missing_prompt_field | 400 | True | `{"error":{"code":400,"message":"[json.exception.out_of_range.403] key 'prompt' not found","type":"invalid_request_error"` |
| unknown_model | 200 | False | `{"choices":[{"text":", please help me","index":0,"logprobs":null,"finish_reason":"length"}],"created":1790609601,"model"` |
| negative_max_tokens | 200 | False | `{"choices":[{"text":" 请帮我把下面内容精简成100个英文单词：The world is in the midst of an energy transition, marked by growing pressure ` |
| prompt_exceeds_context | 400 | True | `{"error":{"code":400,"message":"request (133121 tokens) exceeds the available context size (131072 tokens), try increasi` |
| non_json_body | 500 | True | `{"error":{"code":500,"message":"[json.exception.parse_error.101] parse error at line 1, column 2: syntax error while par` |

