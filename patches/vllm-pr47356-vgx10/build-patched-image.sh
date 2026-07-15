#!/usr/bin/env bash
# Build the vLLM-PR#47356-patched dspark-vllm-gx10 serving image on a cluster node.
#
# WHAT: excludes kv_cache_memory_bytes from CacheConfig.compute_hash() so setting the
#       documented fast-boot KV-sizing knob doesn't perturb the torch.compile cache key
#       and needlessly invalidate the warm FlashInfer/Triton autotune caches on each boot.
# WHY:  the pinned dspark-vllm-gx10 base (vLLM v0.25.1 @ 752a3a50, dated 2026-07-12) just
#       misses this fix — PR #47356 merged upstream 2026-07-07 but landed after the base's
#       vLLM pin. kv_cache_memory_bytes is a runtime-measured sizing field, not a
#       compiled-graph-shape factor, yet pre-patch it leaks into the hash.
#
# PATTERN: extract-then-patch, identical in shape to patches/flashinfer-pr3615/. The patched
#       file is extracted live from THIS build's own base image, patched, verified, and
#       layered back on top — nothing is vendored statically.
#
# RUN ON THE HEAD NODE (bringup/05-build-image.sh drives this). Requires the gx10 base image
# already pulled locally at the pinned digest.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10@sha256:d0a0c050252dd0b64a3213c728d1c15db9eb37602593ba37534bc708dd223ae7}"
TAG="${PATCHED_TAG:-vllm-dspark-runtime:vgx10-cand1-pr47356}"
CACHE_PY_IN_IMG=/usr/local/lib/python3.12/dist-packages/vllm/config/cache.py

echo "== extracting cache.py from $BASE"
WORK="$(mktemp -d)"
mkdir -p "$WORK/vllm/config"
cid="$(docker create "$BASE")"; docker cp "$cid:$CACHE_PY_IN_IMG" "$WORK/vllm/config/cache.py"; docker rm -f "$cid" >/dev/null

echo "== applying cache-hash-exclude-kv-bytes.patch (git apply --check first)"
( cd "$WORK" && git init -q . && git add -A && git -c user.email=patch@example.invalid -c user.name=patch commit -qm base \
  && git apply --check "$HERE/cache-hash-exclude-kv-bytes.patch" && git apply "$HERE/cache-hash-exclude-kv-bytes.patch" )
grep -q '"kv_cache_memory_bytes",' "$WORK/vllm/config/cache.py" \
  || { echo "FAIL: patched cache.py does not contain the expected kv_cache_memory_bytes entry" >&2; exit 1; }

echo "== building $TAG (byte-identical image ID across nodes)"
# Getting `docker buildx build` to produce a byte-identical image ID on both nodes for this
# same base + patch took FOUR things together — each was verified insufficient alone:
#   1. host-side mtime pin on the COPY source file (touch -d "@0") — buildx's COPY otherwise
#      carries the source file's real (node- and time-dependent) mtime into the layer.
#   2. `--output type=docker,rewrite-timestamp=true` with SOURCE_DATE_EPOCH as a shell env var
#      (NOT a --build-arg, which this flag does not read) — normalizes the image config's
#      "created" timestamp and other export-time metadata.
#   3. `--no-cache` — a cached COPY layer from a prior, non-normalized build exports with its
#      stale diff-id even under rewrite-timestamp=true on a later invocation.
#   4. a post-build fixup pass (fixup-layer-mtime.py) — cache.py already exists in the gx10
#      base, so this COPY *overwrites* it. Overlayfs bumps the *parent directory's* mtime to
#      build-wall-clock time as a side effect of the copy-up, and rewrite-timestamp does NOT
#      normalize that directory entry (only file content entries + the image config's own
#      timestamps). This alone left the final layer's diff-id, and therefore the image ID,
#      node-dependent even with everything above in place. Confirmed via a minimal busybox
#      repro: COPY creating a new file at a fresh path was reproducible; COPY overwriting an
#      existing /etc/passwd was not, regardless of rewrite-timestamp.
# Plain `docker build`/`RUN` steps are avoided in the Dockerfile entirely for the same reason:
# any RUN stamps the container's /etc, /proc, /sys mountpoint dirs with build-wall-clock time
# regardless of SOURCE_DATE_EPOCH — the same class of bug as #4.
cp "$WORK/vllm/config/cache.py" "$HERE/cache.py"
touch -d "@0" "$HERE/cache.py"
SOURCE_DATE_EPOCH=0 docker buildx build --no-cache --provenance=false --sbom=false \
  --output "type=docker,name=$TAG,rewrite-timestamp=true" -f "$HERE/Dockerfile" "$HERE"
rm -f "$HERE/cache.py"

echo "== fixup pass: normalizing the parent-directory mtime baked in by the COPY overwrite"
SAVE_DIR="$(mktemp -d)"
docker save "$TAG" -o "$SAVE_DIR/img.tar"
mkdir -p "$SAVE_DIR/extracted"
tar -xf "$SAVE_DIR/img.tar" -C "$SAVE_DIR/extracted"
python3 "$HERE/fixup-layer-mtime.py" "$SAVE_DIR/extracted"
( cd "$SAVE_DIR/extracted" && tar -cf "$SAVE_DIR/fixed.tar" . )
docker load -i "$SAVE_DIR/fixed.tar"
rm -rf "$SAVE_DIR"

echo "== smoke test: hash must be unaffected by kv_cache_memory_bytes"
docker run --rm --entrypoint python3 "$TAG" -c \
  "from vllm.config.cache import CacheConfig; a=CacheConfig().compute_hash(); b=CacheConfig(kv_cache_memory_bytes=1<<30).compute_hash(); assert a==b, 'hash still perturbed by kv_cache_memory_bytes'; print('OK: hash unaffected by kv_cache_memory_bytes')"
rm -rf "$WORK"
echo "== done: $TAG"
docker image inspect --format 'image id: {{.Id}}' "$TAG"
