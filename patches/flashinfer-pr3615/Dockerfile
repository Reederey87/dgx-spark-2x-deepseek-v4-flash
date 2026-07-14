ARG BASE_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
FROM ${BASE_IMAGE}

# FlashInfer PR #3615 / commit 49f2abf: last-CTA reset for the multi-CTA
# radix top-k sampler on SM120/SM121. Sampling is JIT-compiled from this source.
COPY include/flashinfer/topk.cuh /opt/env/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/topk.cuh
LABEL com.dgx-spark-2x-deepseek-v4-flash.flashinfer-patch="pr3615-49f2abf"
