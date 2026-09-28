# bongo baseline benchmark — q2_0

Generated: `2026-09-28T05:12:37+00:00`  
Endpoint: `http://127.0.0.1:8080/v1`  
Model: `bongo-q2_0`  
Harness: `1.0.0`  
Command: `bench/run.sh --repo-root /home/cchild/.paperclip/instances/default/projects/c1ccf5ff-cc7d-4b68-a23c-def2d29bb0a3/15f9f0b7-0021-4164-ae30-902276def736/bongo/.paperclip/worktrees/BAS-48-project-setup --tier q2_0 --contexts 4096,131072 --repeats 1 --max-tokens 128 --needle-context 131072 --hash-mode none --out-dir bench/results/2026-09-28-q2_0/sweep/ncmoe-16`

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
- shard hash mode: `none`  

## Results

`prompt tok/s` is prefill throughput, `output tok/s` is decode throughput, `TTFT` is time to first streamed token. Values are the median of the repeats; `cv` is the coefficient of variation (stdev/median).

| context | prompt tokens | prompt tok/s | output tok/s | TTFT ms | prefill ms | repeats | cv(ttft) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 4097 | 247.921 | 6.975 | 16581.0 | 16525.4 | 1 | n/a |
| 131072 | 130861 | 139.021 | 6.911 | 941485.2 | 941301.7 | 1 | n/a |

## Memory

- peak VRAM: **27.90 GiB** (method: `fdinfo`)
- peak system RAM (process RSS): **11.32 GiB** (method: `proc_status_sum`)
- process RSS high-water mark: 20.73 GiB

## 128K needle

- status: **pass**
- prompt tokens: 130861
- expected token: `VAULT-COORD-7391-QXZ`
- answer: `<think>

</think>

VAULT-COORD-7391-QXZ.
The mountain road curved past the reservoir, where the morning fog had not yet lifted. Engineers logged each pressure reading in a ledger, comparing it again`

## Machine spec

- uname: `Linux localhost.localdomain 7.0.13-200.fc44.x86_64 #1 SMP PREEMPT_DYNAMIC Fri Jun 19 22:51:30 UTC 2026 x86_64 GNU/Linux`
- CPU: `AMD Ryzen 7 9700X 8-Core Processor`
- memory: `Mem:     32699588608  5154271232   370331648    54013952 27702108160 27545317376`
- display devices:

  ```
  03:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Arc Pro B70] [8086:e223]
  12:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Granite Ridge [Radeon Graphics] [1002:13c0] (rev c5)
  ```
- DRM drivers: `{'card0': 'xe', 'card1': 'amdgpu'}`
- modules: `{'xe': {'version': None, 'refcnt': '4'}, 'amdgpu': {'version': None, 'refcnt': '1'}}`
- DRI nodes: `['/dev/dri/by-path', '/dev/dri/card0', '/dev/dri/card1', '/dev/dri/renderD128', '/dev/dri/renderD129']`

## Server

- pids: `[1807625]`
- llama.cpp build: `b11223-4da633776`
- flags:

  ```
  /home/cchild/.bongo/llama/b11223/vulkan/llama-server --model /home/cchild/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ctx-size 131072 --jinja --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --n-cpu-moe 16 --host 127.0.0.1 --port 8080 --parallel 1 --alias bongo-q2_0 --metrics --device Vulkan1 
  ```

## Model / shards

- gguf dir: `/home/cchild/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/q2_0`

| shard | size | sha256 |
| --- | ---: | --- |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37.07 GiB | `None` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | 24.91 GiB | `None` |

## Error / malformed-prompt cases

| case | HTTP | error status | excerpt |
| --- | ---: | --- | --- |
| empty_prompt | 200 | False | `{"choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"length"}],"created":1790572356,"model":"bongo-q2_0","s` |
| missing_prompt_field | 400 | True | `{"error":{"code":400,"message":"[json.exception.out_of_range.403] key 'prompt' not found","type":"invalid_request_error"` |
| unknown_model | 200 | False | `{"choices":[{"text":"  I need some","index":0,"logprobs":null,"finish_reason":"length"}],"created":1790572357,"model":"b` |
| prompt_exceeds_context | 400 | True | `{"error":{"code":400,"message":"request (133121 tokens) exceeds the available context size (131072 tokens), try increasi` |
| non_json_body | 500 | True | `{"error":{"code":500,"message":"[json.exception.parse_error.101] parse error at line 1, column 2: syntax error while par` |
| negative_max_tokens | 200 | False | `{"choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"stop"}],"created":1790572357,"model":"bongo-q2_0","sys` |

