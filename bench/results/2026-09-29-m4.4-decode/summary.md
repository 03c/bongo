# M4.4 4K decode lever screen

4096-token prompt, 128 generated tokens, `cache_prompt=false`, discarded warm-up,
engine llama.cpp `b11223` + the M4.2 patch, Vulkan1, iq2_xs, q8 KV, `--ctx-size 131072`,
`--load-mode none`, `GGML_VK_HOST_BUFT_PER_DEVICE=1 GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1`.

| config | repeats | 4K decode tok/s | median | vs `decode16` |
| --- | --: | --- | --: | --: |
| `decode_nc12` | 3 | 18.98 / 19.59 / 19.61 | **19.59** | +15.50% |
| `decode_offload1_nc12` | 3 | 17.87 / 17.89 / 17.89 | **17.89** | +5.48% |
| `decode_forcemmvq` | 3 | 16.45 / 17.52 / 17.76 | **17.52** | +3.31% |
| `decode16` | 3 | 16.93 / 16.96 / 17.81 | **16.96** | +0.00% |
| `decode_offload1` | 3 | 16.68 / 16.69 / 16.62 | **16.68** | -1.62% |
| `decode_t16` | 3 | 6.02 / 5.63 / 6.27 | **6.02** | -64.53% |

| config | note |
| --- | --- |
| `decode_nc12` | placement: 4 more expert layers on the GPU (less per-token host upload) |
| `decode_offload1_nc12` | lever + placement: offload at batch 1, 4 more layers on the GPU |
| `decode_forcemmvq` | force the explicit mul_mat_vec path for every dense matmul |
| `decode16` | shipped decode config + M4.2 upload levers (the M4.4 reference) |
| `decode_offload1` | lever: offload the CPU-resident expert MUL_MAT_ID to Vulkan at batch 1 |
| `decode_t16` | CPU-side lever: 16 CPU threads for the host-resident expert matmuls |
