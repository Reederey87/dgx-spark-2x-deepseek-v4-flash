#!/usr/bin/env bash
# Build the DSpark vLLM serving image on the head node — dspark-vllm-gx10 (vLLM 0.25.1) lane.
#
# Pulls the digest-pinned anemll base, then applies the vLLM PR #47356 fix as a thin final
# layer (patches/vllm-pr47356-vgx10/). 06 distributes the final image head -> worker; both the
# base (unpatched rollback) and the patched tag stay present on both nodes. The prior 0.21.x
# lane's build path is preserved as bringup/05-build-image.prior-0.21.sh (see docs/08).
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

BASE_PINNED="${BASE_IMAGE_REF}@${BASE_IMAGE_DIGEST}"

ssh "$CLUSTER_USER@$HEAD_HOST" \
  "BASE_PINNED='$BASE_PINNED' bash -s" <<'REMOTE' \
  || fail "base image pull/verify failed on $HEAD_HOST" "check control-host SSH + ghcr reachability"
set -euo pipefail
docker pull "$BASE_PINNED"
echo "ok: base image pulled at pinned digest"
digest="$(docker image inspect "$BASE_PINNED" --format '{{index .RepoDigests 0}}')"
echo "verified base digest: $digest"
REMOTE

echo "== applying the vLLM PR #47356 thin layer (kv_cache_memory_bytes hash exclude)"
rsync -a "$KIT/../patches/vllm-pr47356-vgx10/" \
  "$CLUSTER_USER@$HEAD_HOST:~/vllm-pr47356-vgx10/" \
  || fail "PR #47356 patch sync failed" "check control-host SSH access"
ssh "$CLUSTER_USER@$HEAD_HOST" \
  "BASE_IMAGE='$DSPARK_VLLM_BASE_IMAGE' PATCHED_TAG='$DSPARK_VLLM_IMAGE' bash ~/vllm-pr47356-vgx10/build-patched-image.sh" \
  || fail "PR #47356 patched image build failed" "inspect the patch apply/build output"

echo "== verifying the patched image is cluster-capable"
ssh "$CLUSTER_USER@$HEAD_HOST" \
  "DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' bash -s" <<'REMOTE' \
  || fail "patched image verification failed on $HEAD_HOST" "inspect the image / vllm serve --help"
set -euo pipefail
docker run --rm --entrypoint python3 "$DSPARK_VLLM_IMAGE" -c "import vllm; print('vllm', vllm.__version__)"
help="$(docker run --rm --entrypoint vllm "$DSPARK_VLLM_IMAGE" serve --help)"
for flag in --nnodes --node-rank --headless; do
  printf '%s\n' "$help" | grep -q -- "$flag" \
    || { echo "FAIL: vllm serve missing $flag — image is not cluster-capable" >&2; exit 1; }
done
printf '%s\n' "$help" | grep -E -- '--nnodes|--node-rank|--headless' | head -5
echo "ok: vLLM cluster flags present"
docker image inspect "$DSPARK_VLLM_IMAGE" --format 'final image id: {{.Id}}'
REMOTE
