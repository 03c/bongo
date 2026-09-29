Title: [Intel BMG G31 / Arc Pro B70, Linux] host DMA (dedicated transfer queue) loses the device on first prefill

## Summary

On an Intel Arc Pro B70 (BMG G31, `8086:e223`) under Mesa ANV, INFR `0.9.0`
(`ed62393068679573afe94a1472454efe7eae0f15`) loses the logical Vulkan device on
the **first prefill** when the default host-DMA path (dedicated transfer queue
family) is enabled. Setting `INFR_NO_HOST_DMA=1` makes the same command run to
completion, so the host-DMA upload path is the differentiator.

## Environment

- GPU: Intel Corporation Battlemage G31 [Arc Pro B70] `8086:e223`, 31.9 GiB
  device-local, IntelXe2, `INTEL_OPEN_SOURCE_MESA`
- Kernel: `7.0.13-200.fc44.x86_64` (Fedora 44)
- Mesa: `mesa-vulkan-drivers-26.1.8-1.fc44.x86_64` (ANV)
- INFR: `release-0.9.0`, commit `ed62393068679573afe94a1472454efe7eae0f15`,
  built from source with `cargo build --release -p infr-cli --locked`
- Model: `Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`
  (arch `qwen4exp`, 48 blocks, 512 experts)

## Reproduction

```sh
# clear the pipeline cache first so the run is genuinely cold
rm -f ~/.cache/infr/vk-pipeline-cache-*
INFR_DEV=Vulkan1 infr bench \
  Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  -p 512 -n 0 -r 1 --ctx 4096 --dev Vulkan1 -u 256 \
  --set paging.cache=6GiB
```

`INFR_NO_HOST_DMA` is deliberately **unset** so the default host-DMA path runs.

## Observed

The run logs the host-DMA setup, imports the full host arena, and then dies on
the first prefill:

```
INFO infr_vulkan: [infr] host DMA: dedicated transfer queue family 2 enabled
INFO infr_vulkan: [infr] host DMA import total: 20.96/20.96 GiB across 4/4 arena(s)
INFO infr_vulkan::unified: [infr] unified VRAM physical residency verified arena_bytes=7625510400 shards=2
INFO infr_llama::seam::weights: expanded dynamic KV cache requested_tokens=7 committed_tokens=528 segments=1
INFO infr_vulkan::pager: [moe-prefill] target_lanes=4 actual_lanes=4 resident_layers=0/48 streamed_layer_max=773324800 ring_bytes=3093299200 retired_expert_slots=6692 per_shard_ring_bytes=[0, 3093299200]
ERROR infr_vulkan: [infr] pipelined queue_submit could not be submitted after recovery (The logical device has been lost. See <https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#devsandqueues-lost-device>); refusing later GPU submissions because pager residency may describe copies that never executed
Error: backend: backend: queue_submit: The logical device has been lost. See <https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#devsandqueues-lost-device>
```

Exit code `1` (the process returns rather than hanging). Reproduced with
`paging.cache=6GiB` / `-u 256` on a cold pipeline cache, and previously at the
auto `paging.cache≈22GiB` / adaptive ubatch. With `INFR_NO_HOST_DMA=1` all of
these complete.

## Expected

Either the host-DMA path submits successfully on Intel ANV, or INFR detects the
dedicated-queue path is unusable on this device/driver and falls back to the
non-DMA upload path automatically, instead of losing the device.

## Notes

- The Intel device advertises `external_memory`, `external_memory_fd`,
  `external_memory_dma_buf`; the dedicated transfer queue family is family 2.
- No other Vulkan application loses this device on this box; only the INFR
  host-DMA path does.
