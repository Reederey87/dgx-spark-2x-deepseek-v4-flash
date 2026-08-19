#!/usr/bin/env bash
# Build the c8r-tbfix #52492 (indexer-scoring-in-breakable-graphs) derivative;
# optionally distribute it. Mirrors patches/vllm-0261-main-tbfix/build-and-distribute.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../../runtime/cluster.env"

HEAD_SSH="${HEAD_SSH:-$CLUSTER_USER@$HEAD_HOST}"
WORKER_SSH="${WORKER_SSH:-$CLUSTER_USER@$WORKER_HOST}"
BASE="${BASE:-vllm-dspark-runtime:v0261-main-c8r-tbfix}"
TAG="${TAG:-vllm-dspark-runtime:v0261-main-c8r-tbfix-ixfix}"
REMOTE_HOME="$(ssh "$HEAD_SSH" 'printf %s "$HOME"')"
REMOTE="${REMOTE:-$REMOTE_HOME/build/vllm-0261-main-ixfix}"
TARBALL="vllm-c8r-tbfix-ixfix.tar.zst"
DISTRIBUTE=0

for arg in "$@"; do
  case "$arg" in
    --distribute) DISTRIBUTE=1 ;;
    *) echo "FAIL: unknown flag '$arg' (only --distribute)" >&2; exit 1 ;;
  esac
done

ssh "$HEAD_SSH" "mkdir -p '$REMOTE'"
rsync -a --delete "$HERE/" "$HEAD_SSH:$REMOTE/"
pkg_dir="$(ssh "$HEAD_SSH" "docker run --rm --entrypoint python3 '$BASE' -c 'import os, vllm; print(os.path.dirname(vllm.__file__))'")"
ssh "$HEAD_SSH" "docker buildx build --provenance=false --sbom=false --platform linux/arm64 \
  --build-arg 'VLLM_BASE=$BASE' --build-arg 'VLLM_PKG_DIR=$pkg_dir' \
  --output 'type=docker,name=$TAG' -f '$REMOTE/Dockerfile.runtime' '$REMOTE'"
ssh "$HEAD_SSH" "docker run --rm --entrypoint grep '$TAG' -q \
  is_current_stream_capturing '$pkg_dir/models/deepseek_v4/attention.py' && echo 'ixfix marker OK'"

if [ "$DISTRIBUTE" = 0 ]; then
  echo "OK: built $TAG on $HEAD_SSH"
  echo "next: bash bringup/06-distribute-image.sh"
  exit 0
fi

ssh "$HEAD_SSH" "docker save '$TAG' | zstd -T0 -3 -f -o '$REMOTE_HOME/$TARBALL' && \
  rsync -a --partial --info=progress2 '$REMOTE_HOME/$TARBALL' '$CLUSTER_USER@$WORKER_R1:~/$TARBALL'"
ssh "$WORKER_SSH" "zstd -dc '$TARBALL' | docker load"
head_id="$(ssh "$HEAD_SSH" "docker image inspect '$TAG' --format '{{.Id}}'")"
worker_id="$(ssh "$WORKER_SSH" "docker image inspect '$TAG' --format '{{.Id}}'")"
[ "$head_id" = "$worker_id" ] || { echo "FAIL: image ID mismatch head=$head_id worker=$worker_id" >&2; exit 1; }
ssh "$HEAD_SSH" "rm -f '$REMOTE_HOME/$TARBALL'"
ssh "$WORKER_SSH" "rm -f '$TARBALL'"
echo "OK: $TAG on both nodes: $head_id"
