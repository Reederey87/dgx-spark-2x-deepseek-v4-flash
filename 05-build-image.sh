#!/usr/bin/env bash
# Build the DSpark vLLM image on the head node.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

ssh "$CLUSTER_USER@$HEAD_HOST" \
  "RECIPE_REPO='$RECIPE_REPO' RECIPE_SHA='$RECIPE_SHA' BASE_IMAGE_REF='$BASE_IMAGE_REF' DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' bash -s" <<'REMOTE' \
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
