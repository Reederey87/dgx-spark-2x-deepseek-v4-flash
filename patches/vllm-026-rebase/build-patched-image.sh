#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the vLLM 0.26.0 gx10-overlay image (the cand4 production lane) on a cluster node.
#
# WHAT: official linux/arm64 vllm/vllm-openai:v0.26.0 + gx10-overlay-026.patch
#       (the dspark-vllm-gx10 GB10 overlay cherry-picked onto v0.26.0 with zero
#       conflicts, plus the warn-only dspark_block_size guard, the zero-token-prefill-
#       chunk guard, and backports #50004/#49486) + git-pinned flashinfer/b12x.
# HOW:  extract-then-patch — the 12 upstream files the patch modifies are pulled out of
#       the base image, patched in a temp git repo (which also creates the 1 new
#       overlay-only file), and COPYed back in. Upstream source is never
#       vendored in this repo (see NOTICE); the patch alone carries our delta.
#
# RUN ON ONE NODE, then distribute:
#   TAG=vllm-dspark-runtime:v026-gx10-cand4-backports
#   docker save "$TAG" | zstd -T0 -3 > ~/v026c4-image.tar.zst
#   rsync -a --partial ~/v026c4-image.tar.zst <worker-host>:~/
#   ssh <worker-host> 'zstd -dc ~/v026c4-image.tar.zst | docker load'
# pip-install layers embed timestamps, so per-node builds are NOT byte-identical —
# build once and distribute; assert parity with docker image inspect '{{.Id}}'.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE_IMAGE:-vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52}"
TAG="${PATCHED_TAG:-vllm-dspark-runtime:v026-gx10-cand4-backports}"
PATCH="$HERE/gx10-overlay-026.patch"
CONTEXT_DIR="$HERE/.overlay-build-vllm"

# Extract only files the patch MODIFIES (its `--- a/` side). Files the patch CREATES
# (`--- /dev/null`, e.g. the overlay-only experts/b12x_mxfp4_moe.py) do not exist in the
# base image — docker cp fails on them (issue #7); git apply creates them in the temp repo.
FILES="$(grep -E '^--- a/' "$PATCH" | sed 's|^--- a/||')"
[ -n "$FILES" ] || { echo "FAIL: no file list parsed from $PATCH" >&2; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base not pulled: $BASE" >&2; exit 1; }

echo "== locating vllm package dir in $BASE"
PKG="$(docker run --rm --entrypoint python3 "$BASE" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
echo "   $PKG"

echo "== extracting $(echo "$FILES" | wc -l | tr -d ' ') upstream files from the base"
WORK="$(mktemp -d)"
cid="$(docker create "$BASE")"
for f in $FILES; do
  mkdir -p "$WORK/$(dirname "$f")"
  docker cp "$cid:$PKG/${f#vllm/}" "$WORK/$f"
done
docker rm -f "$cid" >/dev/null

echo "== applying gx10-overlay-026.patch (git apply --check first)"
( cd "$WORK" && git init -q . && git add -A && git -c user.email=x@x -c user.name=x commit -qm base \
  && git apply --check "$PATCH" && git apply "$PATCH" )

echo "== building $TAG"
rm -rf "$CONTEXT_DIR"
cp -r "$WORK/vllm" "$CONTEXT_DIR"
trap 'rm -rf "$CONTEXT_DIR" "$WORK"' EXIT
docker buildx build --provenance=false --sbom=false \
  --build-arg VLLM_BASE="$BASE" \
  --build-arg VLLM_PKG_DIR="$PKG" \
  --output "type=docker,name=$TAG" \
  -f "$HERE/Dockerfile.runtime-026" "$HERE"

echo "== smoke: version + guards + pins"
docker run --rm -i --entrypoint python3 "$TAG" - "$PKG" <<'PY'
import sys
pkg = sys.argv[1]
import vllm
print("vllm version:", vllm.__version__)
assert vllm.__version__.startswith("0.26.0")
spec = open(f"{pkg}/config/speculative.py").read()
assert "upstream guard downgraded" in spec, "warn-only dspark_block_size guard MISSING"
assert "DSpark requires num_speculative_tokens" not in spec, "stock hard-raise present"
assert '"nvfp4_ds_mla",' in open(f"{pkg}/config/cache.py").read(), "CacheDType missing nvfp4"
from vllm.v1.kv_cache_interface import KVQuantMode
assert hasattr(KVQuantMode, "NVFP4"), "KVQuantMode.NVFP4 missing"
fs = open(f"{pkg}/models/deepseek_v4/nvidia/flashinfer_sparse.py").read()
assert "Skipping zero-token sparse prefill chunk" in fs, "zero-chunk guard MISSING"
assert "active_topk_width" in open(f"{pkg}/models/deepseek_v4/sparse_mla.py").read(), "#50004 MISSING"
assert "skip_k_cache_insert" in open(f"{pkg}/models/deepseek_v4/attention.py").read(), "#49486 MISSING"
import b12x, flashinfer
print("overlay smoke PASSED")
PY

echo "== done: $TAG"
docker image inspect --format '{{.Id}}' "$TAG"
