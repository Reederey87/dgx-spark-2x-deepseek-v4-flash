#!/usr/bin/env bash
# Build the production DSpark vLLM serving image — full-source c8r lane.
#
# There is no official arm64 Docker image for this base. Stage-1 is a multi-hour
# nvcc source build of upstream vLLM main @48bada6ea4 on the head node (cluster
# must be stopped), then a thin runtime layer applies the gx10 overlay0261 + the
# #49731 revert. The real builder lives at patches/vllm-0261-main-c8r/ — this
# script is the numbered bringup entry point that sources cluster.env and
# forwards to it.
#
#   bash bringup/05-build-image.sh                 # stage-1 + runtime + smoke
#   bash bringup/05-build-image.sh --distribute    # also push to the worker
#   bash bringup/05-build-image.sh --runtime-only  # overlay-only rebuild
#
# After a build without --distribute, run bringup/06-distribute-image.sh so both
# nodes share the same image IDs. Prior lanes:
#   bringup/05-build-image.prior-0.26-cand7.sh   # thin layer on official v0.26.0
#   patches/vllm-pr47356-vgx10/                  # 0.25.1
#   bringup/05-build-image.prior-0.21.sh         # 0.21.x
#
# Full receipt: docs/14-vllm-027-c8r.md and patches/vllm-0261-main-c8r/README.md.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$KIT/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

echo "== production image: full-source c8r (patches/vllm-0261-main-c8r/)"
echo "   DSPARK_VLLM_IMAGE=$DSPARK_VLLM_IMAGE"
echo "   DSPARK_VLLM_BASE_IMAGE=$DSPARK_VLLM_BASE_IMAGE"
echo "   NOTE: stage-1 is multi-hour nvcc; stop the cluster first (runtime/stop-cluster.sh)."
echo "   Prepare the pinned checkout once: git -C ~/vllm worktree add ~/vllm-0261-main-wt 48bada6ea4"

bash "$ROOT/patches/vllm-0261-main-c8r/build-0261-image.sh" "$@" \
  || fail "c8r image build failed" "inspect patches/vllm-0261-main-c8r/build-0261-image.sh output"

echo "ok: c8r image build finished"
if [[ " $* " != *" --distribute "* ]]; then
  echo "next: bash bringup/06-distribute-image.sh   # head → worker + image-ID parity"
fi
