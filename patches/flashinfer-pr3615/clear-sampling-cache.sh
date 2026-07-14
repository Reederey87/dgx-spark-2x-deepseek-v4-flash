#!/usr/bin/env bash
# Remove every cached FlashInfer sampling JIT object so the selected image's
# installed header is compiled on next use. Run locally on a cluster node.
set -euo pipefail

CACHE="${HF_CACHE:?set HF_CACHE to the host Hugging Face cache directory}"
IMAGE="${DSPARK_VLLM_IMAGE:?set DSPARK_VLLM_IMAGE to an image present on this node}"

[ -d "$CACHE" ] \
  || { echo "FAIL: $CACHE does not exist; refusing a root container bind that would create it" >&2; exit 1; }
[ -w "$CACHE" ] \
  || { echo "FAIL: $CACHE is not writable by $(id -un)" >&2; exit 1; }

docker image inspect "$IMAGE" >/dev/null
docker run --rm --user 0 \
  -v "$CACHE:/cache/huggingface" \
  --entrypoint bash "$IMAGE" -c '
    set -euo pipefail
    root=/cache/huggingface/flashinfer/.cache/flashinfer
    if [ -d "$root" ]; then
      find "$root" -type d -path "*/cached_ops/sampling" -prune -exec rm -rf {} +
      remaining="$(find "$root" -type d -path "*/cached_ops/sampling" -print -quit)"
      [ -z "$remaining" ] || { echo "FAIL: sampling cache remains at $remaining" >&2; exit 1; }
    fi
  '

echo "ok: cleared all FlashInfer sampling JIT caches for $IMAGE"
