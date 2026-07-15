ARG BASE_IMAGE=ghcr.io/anemll/dspark-vllm-gx10@sha256:d0a0c050252dd0b64a3213c728d1c15db9eb37602593ba37534bc708dd223ae7
FROM ${BASE_IMAGE}

# vLLM PR #47356 (upstream, merged 2026-07-07): exclude kv_cache_memory_bytes from
# CacheConfig.compute_hash. A runtime-measured KV-sizing knob, not a compiled-graph-shape
# factor — pre-patch it perturbs the torch.compile cache key and needlessly invalidates the
# warm FlashInfer/Triton autotune caches on every boot. Pure-Python source swap: no JIT or
# wheel rebuild, just a file replace + container restart.
#
# cache.py in the build context is produced at build time by build-patched-image.sh
# (extracted from THIS base image + git apply), not shipped statically. The in-image path
# for this base differs from the prior 0.21.x lane's stage-c base:
#   this base:      /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py
#   prior 0.21.x:   /opt/env/lib/python3.12/site-packages/vllm/config/cache.py
COPY cache.py /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py
LABEL com.dgx-spark-2x-deepseek-v4-flash.vllm-patch="pr47356-kv-cache-memory-bytes-hash-exclude-vgx10"
