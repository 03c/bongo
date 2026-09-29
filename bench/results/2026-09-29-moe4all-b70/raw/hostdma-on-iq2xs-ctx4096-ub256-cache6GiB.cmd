# Host-DMA device-lost reproduction (BAS-181 scope 3). INFR_NO_HOST_DMA is
# deliberately UNSET. Pipeline cache cleared first.
$ rm -f ~/.cache/infr/vk-pipeline-cache-*
$ INFR_DEV=Vulkan1 infr bench \
    ~/.bongo/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/iq2_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
    -p 512 -n 0 -r 1 --ctx 4096 --dev Vulkan1 -u 256 --set paging.cache=6GiB
