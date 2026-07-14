# FlashInfer PR #3615 safety layer

The pinned stage-c image contains FlashInfer 0.6.12 and enables the FlashInfer
sampler. On SM120/SM121, the multi-CTA radix top-k kernel can reset its software
barrier while a peer CTA is still waiting, permanently wedging the CUDA stream.

This directory vendors the source-only reset-race fix from FlashInfer
[PR #3615](https://github.com/flashinfer-ai/flashinfer/pull/3615), commit
`49f2abfbdb517e04b14402389213237aa71658e5`. The sampler is JIT-compiled from
the installed `topk.cuh`, so the thin image layer replaces that header and the
bring-up scripts remove every version/architecture variant of the stale sampling
JIT object before serving.

`bringup/05-build-image.sh` builds both tags:

- `DSPARK_VLLM_BASE_IMAGE`: unpatched stage-c rollback image.
- `DSPARK_VLLM_IMAGE`: final `-fi3615` serving image.

The builder checks that the patch applies cleanly and that the final header has
exactly four `RadixGroupResetStateLastCTA` occurrences (one definition and three
call sites). After a change, run the normal smoke serve and full evaluator.

`06-distribute-image.sh` deliberately archives and verifies **both** tags so the
rollback image is present on both nodes.

Rollback: set `DSPARK_VLLM_IMAGE` to the value of `DSPARK_VLLM_BASE_IMAGE` in
`cluster.env`, sync/render both nodes, and run `clear-sampling-cache.sh` on each
node with `HF_CACHE` and the selected image exported before the normal ordered
pair restart. The next sampler request will JIT-compile from the selected image's
header; never switch these source-overlay tags while reusing a sampling cache.
