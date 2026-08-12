#!/usr/bin/env bash
# PRIOR LANE — vLLM 0.26.0 cand7 (thin layer on official v0.26.0).
#
# Superseded 2026-08-11 by the full-source c8r lane (bringup/05-build-image.sh →
# patches/vllm-0261-main-c8r/). Keep this script for the first image rollback rung:
# set DSPARK_VLLM_IMAGE / DSPARK_VLLM_BASE_IMAGE / the three -cand7 cache roots as
# documented in runtime/cluster.env.example, then run this and 06-distribute-image.sh.
#
# Pulls the digest-pinned official vllm/vllm-openai:v0.26.0 (linux/arm64) base, then
# applies the gx10 overlay + backports as a thin final layer (patches/vllm-026-rebase/).
# See docs/13 (cand7) and docs/14 (c8r promotion + rollback).
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

BASE_PINNED="${BASE_IMAGE_REF}@${BASE_IMAGE_DIGEST}"

ssh "$CLUSTER_USER@$HEAD_HOST" \
  "BASE_PINNED='$BASE_PINNED' bash -s" <<'REMOTE' \
  || fail "base image pull/verify failed on $HEAD_HOST" "check control-host SSH + registry reachability"
set -euo pipefail
docker pull "$BASE_PINNED"
echo "ok: base image pulled at pinned digest"
digest="$(docker image inspect "$BASE_PINNED" --format '{{index .RepoDigests 0}}')"
echo "verified base digest: $digest"
REMOTE

echo "== applying the gx10 0.26 overlay layer (patches/vllm-026-rebase)"
rsync -a "$KIT/../patches/vllm-026-rebase/" \
  "$CLUSTER_USER@$HEAD_HOST:~/vllm-026-rebase/" \
  || fail "0.26 overlay kit sync failed" "check control-host SSH access"
ssh "$CLUSTER_USER@$HEAD_HOST" \
  "BASE_IMAGE='$DSPARK_VLLM_BASE_IMAGE' PATCHED_TAG='$DSPARK_VLLM_IMAGE' bash ~/vllm-026-rebase/build-patched-image.sh" \
  || fail "0.26 overlay image build failed" "inspect the patch apply/build output"

echo "== verifying the patched image is cluster-capable"
ssh "$CLUSTER_USER@$HEAD_HOST" \
  "DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' bash -s" <<'REMOTE' \
  || fail "patched image verification failed on $HEAD_HOST" "inspect the image / vllm serve --help=all"
set -euo pipefail
docker run --rm --entrypoint python3 "$DSPARK_VLLM_IMAGE" -c "import vllm; print('vllm', vllm.__version__)"
help="$(docker run --gpus=all --rm --entrypoint vllm "$DSPARK_VLLM_IMAGE" serve --help=all)"
for flag in --nnodes --node-rank --headless; do
  printf '%s\n' "$help" | grep -- "$flag" > /dev/null \
    || { echo "FAIL: vllm serve missing $flag — image is not cluster-capable" >&2; exit 1; }
done
printf '%s\n' "$help" | grep -E -- '--nnodes|--node-rank|--headless' | head -5
echo "ok: vLLM cluster flags present"
docker image inspect "$DSPARK_VLLM_IMAGE" --format 'final image id: {{.Id}}'
REMOTE
