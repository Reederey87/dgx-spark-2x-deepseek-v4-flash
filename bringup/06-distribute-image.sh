#!/usr/bin/env bash
# Save the DSpark image on head, copy it over QSFP, and load it on worker.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

if [ "${SKIP_SAVE:-0}" != "1" ]; then
  ssh "$CLUSTER_USER@$HEAD_HOST" "DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' DSPARK_VLLM_BASE_IMAGE='$DSPARK_VLLM_BASE_IMAGE' bash -s" <<'REMOTE' \
    || fail "docker save failed on $HEAD_HOST" "verify image exists and zstd is installed"
set -euo pipefail
docker save "$DSPARK_VLLM_IMAGE" "$DSPARK_VLLM_BASE_IMAGE" | zstd -T0 -3 > ~/vllm-dsv4-image.tar.zst
REMOTE
  echo "ok: serving + rollback image tags archived on head"
else
  echo "ok: SKIP_SAVE=1, using existing head archive"
fi

ssh "$CLUSTER_USER@$HEAD_HOST" "rsync -a --partial --info=progress2 ~/vllm-dsv4-image.tar.zst '$CLUSTER_USER@$WORKER_R1:~/'" \
  || fail "image archive rsync failed" "verify head-to-worker QSFP SSH"
echo "ok: image archive copied to worker"

ssh "$CLUSTER_USER@$WORKER_HOST" 'zstd -dc ~/vllm-dsv4-image.tar.zst | docker load' \
  || fail "docker load failed on $WORKER_HOST" "verify archive and docker daemon"
echo "ok: image loaded on worker"

# Force the worker's first sampler use to compile from the patched topk.cuh.
rsync -a "$KIT/../patches/flashinfer-pr3615/" \
  "$CLUSTER_USER@$WORKER_HOST:~/flashinfer-pr3615/" \
  || fail "FlashInfer helper sync failed" "check control-host SSH access"
ssh "$CLUSTER_USER@$WORKER_HOST" \
  "HF_CACHE='$HF_CACHE' DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' bash ~/flashinfer-pr3615/clear-sampling-cache.sh" \
  || fail "worker sampler JIT-cache clear failed" "verify HF_CACHE permissions and the patched image"

head_id="$(ssh "$CLUSTER_USER@$HEAD_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' --format '{{.Id}}'")" \
  || fail "could not inspect head image" "verify image tag on head"
worker_id="$(ssh "$CLUSTER_USER@$WORKER_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' --format '{{.Id}}'")" \
  || fail "could not inspect worker image" "verify docker load on worker"
head_base_id="$(ssh "$CLUSTER_USER@$HEAD_HOST" "docker image inspect '$DSPARK_VLLM_BASE_IMAGE' --format '{{.Id}}'")" \
  || fail "could not inspect head rollback image" "verify base image tag on head"
worker_base_id="$(ssh "$CLUSTER_USER@$WORKER_HOST" "docker image inspect '$DSPARK_VLLM_BASE_IMAGE' --format '{{.Id}}'")" \
  || fail "could not inspect worker rollback image" "rebuild the archive with SKIP_SAVE=0"

echo "head image:   $head_id"
echo "worker image: $worker_id"
[ "$head_id" = "$worker_id" ] || fail "image IDs differ" "rerun distribution or rebuild from the same tag"
echo "ok: image IDs match"
echo "head rollback image:   $head_base_id"
echo "worker rollback image: $worker_base_id"
[ "$head_base_id" = "$worker_base_id" ] \
  || fail "rollback image IDs differ" "rerun distribution with SKIP_SAVE=0"
echo "ok: rollback image IDs match"
