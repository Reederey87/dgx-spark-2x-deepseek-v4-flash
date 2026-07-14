#!/usr/bin/env bash
# Build the pinned FlashInfer PR #3615 sampler safety layer on a cluster node.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE_IMAGE:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
TAG="${PATCHED_TAG:-vllm-dspark-runtime:dspark-nvfp4-stage-c-fi3615}"
CUH_IN_IMAGE=/opt/env/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/topk.cuh
WORK="$(mktemp -d)"
CID=
cleanup() {
  [ -z "$CID" ] || docker rm -f "$CID" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$WORK/include/flashinfer"
CID="$(docker create "$BASE")"
docker cp "$CID:$CUH_IN_IMAGE" "$WORK/include/flashinfer/topk.cuh"
docker rm -f "$CID" >/dev/null
CID=

(
  cd "$WORK"
  git init -q .
  git add -A
  git -c user.email=patch@example.invalid -c user.name=patch commit -qm base
  git apply --check "$HERE/49f2abf.patch"
  git apply "$HERE/49f2abf.patch"
)

[ "$(grep -c RadixGroupResetStateLastCTA "$WORK/include/flashinfer/topk.cuh")" = 4 ] \
  || { echo "FAIL: patched topk.cuh does not contain the expected definition + 3 call sites" >&2; exit 1; }

cp "$HERE/Dockerfile" "$WORK/Dockerfile"
docker build --build-arg "BASE_IMAGE=$BASE" -f "$WORK/Dockerfile" -t "$TAG" "$WORK"
count="$(docker run --rm --entrypoint bash "$TAG" -c "grep -c RadixGroupResetStateLastCTA '$CUH_IN_IMAGE'")"
[ "$count" = 4 ] || { echo "FAIL: final image patch verification returned $count (want 4)" >&2; exit 1; }
echo "ok: built $TAG from $BASE with FlashInfer PR #3615"
