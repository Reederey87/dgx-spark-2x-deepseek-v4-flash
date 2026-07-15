#!/usr/bin/env bash
# ============================================================================
# PRIOR 0.21.x LANE — build script for the previously-shipped default.
#
# This is the pre-2026-07-15 build path, kept intact for rollback: the pinned
# tonyd2wild recipe (stage-c) base + the FlashInfer PR #3615 sampler-hang fix
# as a thin final layer. It is NOT the current default — bringup/05-build-image.sh
# builds the dspark-vllm-gx10 (vLLM 0.25.1) lane. See docs/08.
#
# To use this lane, first swap cluster.env section 4 to the ROLLBACK block
# (uncomment the prior 0.21.x pins there), then run this script instead of 05.
# ============================================================================
# Build the DSpark vLLM image on the head node.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

ssh "$CLUSTER_USER@$HEAD_HOST" \
  "RECIPE_REPO='$RECIPE_REPO' RECIPE_SHA='$RECIPE_SHA' BASE_IMAGE_REF='$BASE_IMAGE_REF' DSPARK_VLLM_IMAGE='$DSPARK_VLLM_BASE_IMAGE' bash -s" <<'REMOTE' \
  || fail "image build failed on $HEAD_HOST" "inspect ~/dspark-recipe build logs on the head node"
set -euo pipefail

git clone "$RECIPE_REPO" ~/dspark-recipe 2>/dev/null || true
cd ~/dspark-recipe
git fetch
git checkout "$RECIPE_SHA"
echo "ok: recipe checked out"

docker pull "$BASE_IMAGE_REF"
digest="$(docker image inspect "$BASE_IMAGE_REF" --format '{{index .RepoDigests 0}}')"
echo "PIN THIS in cluster.env BASE_IMAGE_DIGEST=$digest"
echo "ok: base image pulled"

WORKER_BUILD=0 ./build-dspark-vllm-runtime.sh
echo "ok: build script completed"

docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_VLLM_IMAGE" -c "import vllm; print(vllm.__version__)"
help="$(docker run --rm --entrypoint /opt/env/bin/vllm "$DSPARK_VLLM_IMAGE" serve --help)"
printf '%s\n' "$help" | grep -q -- '--nnodes' || { echo "FAIL: vllm serve missing --nnodes — image is not cluster-capable" >&2; exit 1; }
printf '%s\n' "$help" | grep -q -- '--node-rank' || { echo "FAIL: vllm serve missing --node-rank — image is not cluster-capable" >&2; exit 1; }
printf '%s\n' "$help" | grep -q -- '--headless' || { echo "FAIL: vllm serve missing --headless — image is not cluster-capable" >&2; exit 1; }
printf '%s\n' "$help" | grep -E -- '--nnodes|--node-rank|--headless' | head -5
echo "ok: vLLM cluster flags present"

docker image inspect "$DSPARK_VLLM_IMAGE" --format 'image id: {{.Id}}'
REMOTE

echo "== applying the pinned FlashInfer PR #3615 thin layer"
rsync -a "$KIT/../patches/flashinfer-pr3615/" \
  "$CLUSTER_USER@$HEAD_HOST:~/flashinfer-pr3615/" \
  || fail "FlashInfer patch sync failed" "check control-host SSH access"
ssh "$CLUSTER_USER@$HEAD_HOST" \
  "BASE_IMAGE='$DSPARK_VLLM_BASE_IMAGE' PATCHED_TAG='$DSPARK_VLLM_IMAGE' bash ~/flashinfer-pr3615/build-patched-image.sh" \
  || fail "FlashInfer patched image build failed" "inspect the patch apply/build output"

# A prior unpatched sampling JIT object must not survive the source overlay.
ssh "$CLUSTER_USER@$HEAD_HOST" \
  "HF_CACHE='$HF_CACHE' DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' bash ~/flashinfer-pr3615/clear-sampling-cache.sh" \
  || fail "head sampler JIT-cache clear failed" "verify HF_CACHE permissions and the patched image"

ssh "$CLUSTER_USER@$HEAD_HOST" \
  "docker image inspect '$DSPARK_VLLM_IMAGE' --format 'final image id: {{.Id}}'" \
  || fail "could not inspect final image" "verify the FlashInfer thin layer was tagged"
