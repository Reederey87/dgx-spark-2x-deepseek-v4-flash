#!/usr/bin/env bash
# Build the v0261-main-c8r image — FULL-SOURCE vLLM main @ 48bada6ea4 + the gx10
# overlay0261 (14 files) PLUS the #49731 revert (DSpark Markov head back to TP-sharded;
# overlay0261/ carries the 14 base files + the 3 reverted files: qwen3_dspark.py,
# vocab_parallel_embedding.py, logits_processor.py). The current production lane
# (promoted 2026-08-11 — the gate story and the revert rationale are docs/14).
#
# CONTROL-HOST DRIVER: there is no official arm64 image for this base, so stage-1 is a
# multi-HOUR nvcc source build that must not hold an interactive node shell. This script
# orchestrates from the control host over ssh; every docker step runs on the head node
# as the cluster user, and --distribute ships the bits to the worker over rail 1
# (docker save | zstd | rsync | docker load), making cross-node image-ID parity true
# BY CONSTRUCTION.
#
# Usage:
#   bash build-0261-image.sh                 # build stage-1 + runtime + smoke on the head
#   bash build-0261-image.sh --distribute    # the above, then push to the worker + parity assert
#   bash build-0261-image.sh --runtime-only [--distribute]   # overlay-only iteration:
#                                             # skip stage-1, rebuild just the runtime
#                                             # layer over the EXISTING $SRC_TAG
#
# Hosts come from runtime/cluster.env like every other script in this kit (sourced
# below: CLUSTER_USER, HEAD_HOST, WORKER_HOST, WORKER_R1). Override via HEAD_SSH /
# WORKER_SSH if your ssh aliases differ.
#
# Env knobs:
#   VLLM_REPO   a local clone of github.com/vllm-project/vllm (default ~/vllm)
#   SRC_TREE    prepared local checkout at the pinned SHA (default ~/vllm-0261-main-wt).
#               This script NEVER creates git refs — prepare it by hand first:
#                 git -C ~/vllm worktree add ~/vllm-0261-main-wt 48bada6ea4
#               or reuse any existing worktree at the pin (`git -C ~/vllm worktree list`)
#               via SRC_TREE=<path>.
#   SRC_TAG     stage-1 tag     (default vllm-dspark-src:v0261-main-c8)
#   FINAL_TAG   runtime tag     (default vllm-dspark-runtime:v0261-main-c8r)
#   MAX_JOBS    if set, passed to stage-1 as --build-arg max_jobs (upstream default is 2;
#               GB10 UMA — do NOT raise alongside the serving pool, see below)
#
# PREREQUISITES:
#   - overlay0261/vllm/ present next to this script (fixed contract — same whole-file
#     layout as patches/vllm-026-rebase/overlay026/vllm/).
#   - SRC_TREE at the pinned SHA (verified via rev-parse; dirty-tree = WARN only, since
#     re-runs re-apply the overlay onto the same files).
#   - CLUSTER STOPPED on the head node: a full nvcc build does NOT fit alongside the
#     serving pool on the 121 GiB UMA (stop with runtime/stop-cluster.sh, or run with
#     MAX_JOBS=2 + swap watch). The script refuses to build while container vllm-dsv4
#     is running on the head.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../../runtime/cluster.env"

VLLM_REPO="${VLLM_REPO:-$HOME/vllm}"
PIN_SHA="48bada6ea49ad7f3ecbe03128aa76562089c8b00"   # v0.26.1rc0-590-g48bada6ea4 (main)
SRC_TREE="${SRC_TREE:-$HOME/vllm-0261-main-wt}"

HEAD_SSH="${HEAD_SSH:-$CLUSTER_USER@$HEAD_HOST}"
WORKER_SSH="${WORKER_SSH:-$CLUSTER_USER@$WORKER_HOST}"
# worker rail-1 IP ($WORKER_R1 from cluster.env) is used for the head->worker push.
REMOTE_HOME="$(ssh "$HEAD_SSH" 'printf %s "$HOME"')"
REMOTE_SRC="$REMOTE_HOME/build/vllm-0261-main-c8"          # stage-1 build context on the head (shared with the c8 lane — REUSED by --runtime-only)
REMOTE_RUNTIME="$REMOTE_HOME/build/vllm-0261-main-c8r-runtime"  # runtime-layer context (Dockerfile + overlay)

SRC_TAG="${SRC_TAG:-vllm-dspark-src:v0261-main-c8}"       # c8 stage-1 base (already on the head node)
FINAL_TAG="${FINAL_TAG:-vllm-dspark-runtime:v0261-main-c8r}"

DISTRIBUTE=0
for arg in "$@"; do
  case "$arg" in
    --distribute) DISTRIBUTE=1 ;;
    --runtime-only) RUNTIME_ONLY=1 ;;
    *) echo "FAIL: unknown flag '$arg' (only --distribute / --runtime-only)" >&2; exit 1 ;;
  esac
done

# --runtime-only (2026-08-11, DG SF-layout overlay hotfix): skip the SRC_TREE checks,
# the tree sync, and the multi-hour stage-1 — rebuild ONLY the runtime layer from the
# EXISTING $SRC_TAG on the head node (overlay copy + b12x pin), then smoke + optional
# distribute. Use for overlay-only iterations (overlay0261/ content changes); the
# runtime context rsync below already ships the fresh overlay to the build dir.
RUNTIME_ONLY="${RUNTIME_ONLY:-0}"

# ---- local preconditions (cheap, fail before any GB move) ----
[ -d "$HERE/overlay0261/vllm" ] || { echo "FAIL: overlay0261/vllm missing next to this script — the overlay export is a fixed contract produced by the separate overlay-port workstream (same layout as patches/vllm-026-rebase/overlay026/vllm/)" >&2; exit 1; }
if [ "$RUNTIME_ONLY" = 0 ]; then
if [ ! -d "$SRC_TREE" ]; then
  echo "FAIL: SRC_TREE not found: $SRC_TREE" >&2
  echo "  This script never creates git refs. Prepare the pinned checkout by hand:" >&2
  echo "    git -C $VLLM_REPO worktree add $SRC_TREE 48bada6ea4" >&2
  echo "  or reuse an existing worktree at the pin — current worktrees:" >&2
  git -C "$VLLM_REPO" worktree list >&2 || true
  echo "  then re-run with SRC_TREE=<path>." >&2
  exit 1
fi
HEAD_SHA="$(git -C "$SRC_TREE" rev-parse HEAD 2>/dev/null || true)"
if [ "$HEAD_SHA" != "$PIN_SHA" ]; then
  echo "FAIL: SRC_TREE is at ${HEAD_SHA:-<not a git tree>}, need $PIN_SHA (v0.26.1rc0-590-g48bada6ea4)" >&2
  echo "  Re-prepare: git -C $VLLM_REPO worktree add $SRC_TREE 48bada6ea4" >&2
  exit 1
fi
if [ -n "$(git -C "$SRC_TREE" status --porcelain 2>/dev/null || true)" ]; then
  echo "WARN: SRC_TREE is dirty before overlay application (expected on re-runs: the overlay re-applies onto the same files; unexpected otherwise)" >&2
fi
fi  # end RUNTIME_ONLY-skipped local preconditions

# ---- remote preflight: cluster must be stopped on the build node ----
# nvcc does not fit alongside the serving pool; refuse while vllm-dsv4 is up on the head node.
echo "== preflight: cluster stopped on the head node (nvcc build vs serving pool)"
# Capture first: an ssh failure must abort via set -e, NEVER read as "cluster stopped".
PS_NAMES="$(ssh "$HEAD_SSH" "docker ps --format '{{.Names}}'")"
if printf '%s\n' "$PS_NAMES" | grep -qx vllm-dsv4; then
  echo "FAIL: container vllm-dsv4 is RUNNING on the head node — stop the cluster first (runtime/stop-cluster.sh)." >&2
  echo "  A full nvcc source build does not fit alongside the serving pool (121 GiB UMA)." >&2
  exit 1
fi
echo "   OK: vllm-dsv4 not running on the head node"

# ---- a. apply the overlay onto the pinned source tree (whole-file replace) ----
# Per-file list computed BEFORE the sync from the overlay set itself (not from rsync
# --out-format — some control hosts ship openrsync; keep flag use minimal (kit convention).
OVERLAY_FILES="$(cd "$HERE/overlay0261/vllm" && find . -type f | sed 's|^\./||' | sort)"
OVERLAY_PY_FILES="$(printf '%s\n' $OVERLAY_FILES | grep '\.py$' || true)"
[ -n "$OVERLAY_PY_FILES" ] || { echo "FAIL: overlay0261/vllm contains no .py files — the overlay export is incomplete (fixed contract, produced separately)" >&2; exit 1; }
if [ "$RUNTIME_ONLY" = 0 ]; then
echo "== applying overlay0261 -> $SRC_TREE/vllm/ (rsync -a, whole-file replace)"
rsync -a "$HERE/overlay0261/vllm/" "$SRC_TREE/vllm/"
printf '   overlaid: %s\n' $OVERLAY_FILES
echo "   $(printf '%s\n' $OVERLAY_FILES | wc -l | tr -d ' ') files ($(printf '%s\n' $OVERLAY_PY_FILES | wc -l | tr -d ' ') .py)"
else
echo "== runtime-only: skipping overlay->SRC_TREE apply (overlay ships in the runtime context)"
fi
# NOTE: plain rsync -a is CORRECT here (apply onto a pristine pinned checkout). The
# 026 README's rsync -a --delete warning governs syncing the overlay EXPORT dir between
# variants, not application onto the source tree — --delete here would gut the tree.

# ---- b. sync the tree (minus .git) + the runtime-layer context to the head node ----
# --delete on BOTH syncs: these remote dirs are dedicated to this build, and a stale
# file lingering from a previous sync would be baked into the image — the exact
# wrong-content trap the 026 README documents for the overlay export (burned twice
# 2026-07-28). (Kit's spark-sync.sh stays additive-by-default for ~/Developer; build
# contexts are the opposite case — they must MIRROR the local tree.)
echo "== rsync runtime-layer context -> $HEAD_SSH:$REMOTE_RUNTIME/"
ssh "$HEAD_SSH" "mkdir -p '$REMOTE_SRC' '$REMOTE_RUNTIME'"
rsync -a --delete "$HERE/" "$HEAD_SSH:$REMOTE_RUNTIME/"

if [ "$RUNTIME_ONLY" = 0 ]; then
echo "== rsync source tree -> $HEAD_SSH:$REMOTE_SRC/ (excluding .git)"
rsync -a --delete --exclude='.git' "$SRC_TREE/" "$HEAD_SSH:$REMOTE_SRC/"

# The upstream Dockerfile bind-mounts the context's .git into the wheel build
# (docker/Dockerfile:497,514) and setup.py derives the version with setuptools_scm —
# both hard-fail without it ("failed to calculate checksum ... /.git: not found", then
# "unable to detect version"). A worktree's .git is a dangling pointer off-box, and we
# exclude it above, so synthesize a minimal truthful repo in the remote context:
# one commit + the rc0 tag → describe yields 0.26.1rc0.dev1+g<hash> (honestly NOT the
# official rc0). Idempotent: rsync --delete does not touch the excluded remote .git,
# and re-runs hit "nothing to commit" (|| true) + tag -f.
echo "== synthesizing .git in $HEAD_SSH:$REMOTE_SRC (docker .git bind-mount + setuptools_scm)"
ssh "$HEAD_SSH" "cd '$REMOTE_SRC' && git init -q && \
  git -c user.email=cand8@local -c user.name=cand8 add -A && \
  { git -c user.email=cand8@local -c user.name=cand8 commit -q -m 'cand8: main 48bada6ea4 + overlay0261' || true; } && \
  git tag -f v0.26.1rc0 >/dev/null && git describe --tags --long"

# ---- c. stage-1: upstream full-source build (HOURS on GB10) ----
MAX_JOBS_ARG=()
if [ -n "${MAX_JOBS:-}" ]; then
  MAX_JOBS_ARG=(--build-arg "max_jobs=$MAX_JOBS")
  echo "== stage-1: $SRC_TAG (upstream docker/Dockerfile --target vllm-openai, torch_cuda_arch_list=12.1a, max_jobs=$MAX_JOBS)"
else
  # upstream default max_jobs=2 — the safe UMA value; only override deliberately.
  echo "== stage-1: $SRC_TAG (upstream docker/Dockerfile --target vllm-openai, torch_cuda_arch_list=12.1a, max_jobs=<upstream default 2>)"
fi
# arm64 base: the Dockerfile default BUILD_BASE_IMAGE (pytorch/manylinux2_28-builder) is
# x86_64-only — "no match for platform in manifest". Upstream's own arm64 lane
# (.buildkite/image_build/image_build_arm64.sh) swaps in the aarch64 builder; mirror it.
ssh "$HEAD_SSH" "docker buildx build --provenance=false --sbom=false \
  --platform linux/arm64 \
  --target vllm-openai \
  --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0 \
  --build-arg torch_cuda_arch_list=12.1a \
  ${MAX_JOBS_ARG[*]:-} \
  --output 'type=docker,name=$SRC_TAG' \
  -f '$REMOTE_SRC/docker/Dockerfile' '$REMOTE_SRC'"
fi  # end RUNTIME_ONLY-skipped stage-1

# ---- d. runtime layer: overlay copy + b12x pin -> FINAL_TAG ----
echo "== locating vllm package dir in $SRC_TAG"
VLLM_PKG_DIR="$(ssh "$HEAD_SSH" "docker run --rm --entrypoint python3 '$SRC_TAG' -c 'import vllm, os; print(os.path.dirname(vllm.__file__))'")"
echo "   $VLLM_PKG_DIR"

echo "== building $FINAL_TAG (Dockerfile.runtime-0261, VLLM_BASE=$SRC_TAG)"
ssh "$HEAD_SSH" "docker buildx build --provenance=false --sbom=false \
  --build-arg 'VLLM_BASE=$SRC_TAG' \
  --build-arg 'VLLM_PKG_DIR=$VLLM_PKG_DIR' \
  --output 'type=docker,name=$FINAL_TAG' \
  -f '$REMOTE_RUNTIME/Dockerfile.runtime-0261' '$REMOTE_RUNTIME'"

# ---- f. post-build smoke on the head node (BEFORE any distribute: fail fast, not after a
#         ~20 GB rail-1 push). GPU-free python smoke, then --gpus=all CLI check. ----
echo "== smoke: version + overlay/native markers + flashinfer pin + py_compile"
# Compile EVERY overlaid .py file (026-script lesson: a hardcoded subset silently skips
# new files when the overlay set swings between candidates).
ssh "$HEAD_SSH" "docker run --rm -i --entrypoint python3 '$FINAL_TAG' - '$VLLM_PKG_DIR' '$OVERLAY_PY_FILES'" <<'PY'
import importlib.metadata, os, py_compile, sys
pkg = sys.argv[1]
overlay_files = sys.argv[2].split()
import vllm
print("vllm version:", vllm.__version__)

# --- overlay-carried markers (overlay0261 — gx10 port onto main; boot depends on these) ---
spec = open(f"{pkg}/config/speculative.py").read()
assert "upstream guard downgraded" in spec, "guard warn-only overlay MISSING (CRITICAL: boot depends on it)"
assert "DSpark requires num_speculative_tokens" not in spec, "stock hard-raise still present (main's stock wording verified present @48bada6ea4 — overlay did not apply)"
cache_src = open(f"{pkg}/config/cache.py").read()
assert '"nvfp4_ds_mla",' in cache_src, "nvfp4_ds_mla missing from CacheDType"
from vllm.v1.kv_cache_interface import KVQuantMode
assert hasattr(KVQuantMode, "NVFP4"), "KVQuantMode.NVFP4 missing"
assert "VLLM_USE_B12X_MOE" in open(f"{pkg}/envs.py").read(), "VLLM_USE_B12X_MOE env missing"
assert "B12X_MXFP4" in open(f"{pkg}/model_executor/layers/fused_moe/oracle/mxfp4.py").read(), "B12X_MXFP4 mxfp4 oracle missing"
assert os.path.exists(f"{pkg}/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py"), "b12x_mxfp4_moe.py missing"
fi_sparse = open(f"{pkg}/models/deepseek_v4/nvidia/flashinfer_sparse.py").read()
assert "Packed SM120 DSv4 KV page size must be a multiple of 64" in fi_sparse, "SM120 DSv4 KV page guard missing"
assert "_pad_decode_sparse_indices" in fi_sparse, "_pad_decode_sparse_indices missing"
dg_src = open(f"{pkg}/utils/deep_gemm.py").read()
assert "_dg_sm12x_packed_ue8m0_layout" in dg_src, "SM120 SF-layout torch port MISSING (CRITICAL: boot dies on DeepGEMM e21c821 'Unknown SF transformation' without it)"

# --- DeepGEMM pin revert (2026-08-11, runtime-layer swap to deepseek-ai@a6b593d2):
#     stage-1's e21c821 fork pin dropped the whole sm120 backend (boot dies at
#     hyperconnection.hpp:56 'Unsupported architecture' on the first forward). Assert the
#     swap landed: cp312 binding present + sm120 markers in the JIT include tree (e21c821
#     has ZERO sm120 files there; a6b593d2 has 11). ---
dg_pkg = f"{pkg}/third_party/deep_gemm"
import glob as _glob
assert _glob.glob(f"{dg_pkg}/_C.cpython-312-*.so"), "vendored DeepGEMM _C.so for cp312 missing (pin-swap RUN did not apply)"
_sm120_hits = [p for p in _glob.glob(f"{dg_pkg}/include/**/*.h*", recursive=True) if "sm120" in open(p, errors="ignore").read()]
assert _sm120_hits, "no sm120 refs in vendored DeepGEMM include/ — still the e21c821 pin (drops sm120; GB10 unsupported)"
print(f"DeepGEMM vendored pin OK (a6b593d2 swap; {len(_sm120_hits)} sm120 include files)")

# --- #49731 revert markers (c8r lane): DSpark Markov head back to TP-sharded ---
_q3 = open(f"{pkg}/model_executor/models/qwen3_dspark.py").read()
assert "disable_tp" not in _q3, "#49731 revert MISSING in qwen3_dspark.py (disable_tp still present)"
_lp = open(f"{pkg}/model_executor/layers/logits_processor.py").read()
assert "lm_head.tp_size" not in _lp and "get_tensor_model_parallel_world_size" in _lp, "#49731 revert MISSING in logits_processor.py"
assert "disable_tp" not in open(f"{pkg}/model_executor/layers/vocab_parallel_embedding.py").read(), "#49731 revert MISSING in vocab_parallel_embedding.py"
print("#49731 revert markers OK (Markov head TP-sharded, quant plumbing kept)")

# --- native on the main @48bada6ea4 base (were cand4/cand7 backports on 0.26.0; the
#     overlay port must carry main's versions of these files, not re-add the patches) ---
assert "active_topk_width" in open(f"{pkg}/models/deepseek_v4/sparse_mla.py").read(), "native adaptive topk width missing"
assert "_get_c128_boundary" in open(f"{pkg}/models/deepseek_v4/compressor.py").read(), "native c128 empty-launch skip missing"
assert "_SPARSE_MLA_SUPPORTED_Q_HEADS" in fi_sparse, "native q-head helper missing"
assert "get_draft_quant_config" in open(f"{pkg}/v1/worker/gpu/spec_decode/dspark/utils.py").read(), "native draft quant override missing"
assert "_fill_short_context_topk_indices" in open(f"{pkg}/models/deepseek_v4/attention.py").read(), "native short-context topk fill missing"
# NB: deliberately NO #47574 negative assert on this base — #47574 and its companion
# #49903 are both natively present and correct on main (the 026 assert was 0.26.0-specific).

# --- flashinfer: main's native pin, installed by stage-1 from requirements/cuda.txt
#     (https://flashinfer.ai/whl/). The runtime layer deliberately does NOT reinstall —
#     assert the exact version here instead (decision 2026-08-11: native pin, not the
#     026 layer's @0472b9b3 source rebuild). ---
fi_ver = importlib.metadata.version("flashinfer-python")
cubin_ver = importlib.metadata.version("flashinfer-cubin")
assert fi_ver == "0.6.16.post3", f"flashinfer-python {fi_ver} != 0.6.16.post3 — stage-1 did not install main's pin"
assert cubin_ver == "0.6.16.post3", f"flashinfer-cubin {cubin_ver} != 0.6.16.post3 — stage-1 did not install main's pin"
print(f"flashinfer-python {fi_ver} + cubin {cubin_ver} OK (native pin, no reinstall)")

for f in overlay_files:
    py_compile.compile(f"{pkg}/{f}", doraise=True)
print(f"py_compile ok ({len(overlay_files)} files)")
import b12x, flashinfer
print("b12x + flashinfer import OK; overlay smoke PASSED")
PY

echo "== verify cluster-capable (serve --help=all)"
# vLLM CLI gotchas (from the 026 script, public-kit PR #15): constructing the serve arg
# parser instantiates DeviceConfig → device-type inference needs a GPU visible in the
# container (default-runtime is runc on the Sparks, so pass --gpus=all); a bare --help
# prints only Config Group titles (individual flags need --help=all); and grep -q on the
# ~100 KB help text SIGPIPEs printf under pipefail (false "missing flag"), so let grep
# read the full stream.
help="$(ssh "$HEAD_SSH" "docker run --gpus=all --rm --entrypoint vllm '$FINAL_TAG' serve --help=all")"
for flag in --nnodes --node-rank --headless; do
  printf '%s\n' "$help" | grep -- "$flag" > /dev/null \
    || { echo "FAIL: vllm serve missing $flag — image is not cluster-capable" >&2; exit 1; }
done
echo "ok: --nnodes/--node-rank/--headless present"

echo "== GPU probe: vendored DeepGEMM binding imports and exposes the HC entry point"
ssh "$HEAD_SSH" "docker run --gpus=all --rm --entrypoint python3 '$FINAL_TAG' -c \"
from vllm.third_party import deep_gemm as dg
assert hasattr(dg, 'tf32_hc_prenorm_gemm'), 'tf32_hc_prenorm_gemm missing — DG pin revert did not land'
assert hasattr(dg, 'transform_sf_into_required_layout'), 'transform_sf_into_required_layout missing'
print('DeepGEMM binding import + HC/SF entry points OK (a6b593d2)')
\""

# ---- e. --distribute: ship the bits to the worker over rail 1, assert parity ----
if [ "$DISTRIBUTE" = 1 ]; then
  TARBALL="v0261-main-c8r-image.tar.zst"
  echo "== distribute: docker save | zstd -> rsync rail-1 -> docker load on the worker"
  ssh "$HEAD_SSH" "docker save '$FINAL_TAG' | zstd -T0 -3 > ~/$TARBALL"
  ssh "$HEAD_SSH" "rsync -a --partial --info=progress2 ~/$TARBALL '$CLUSTER_USER@$WORKER_R1:~/'"
  ssh "$WORKER_SSH" "zstd -dc ~/$TARBALL | docker load"
  echo "== parity assert: image ID must match on both nodes (true by construction via save/load; asserted anyway)"
  id_head="$(ssh "$HEAD_SSH" "docker image inspect --format '{{.Id}}' '$FINAL_TAG'")"
  id_worker="$(ssh "$WORKER_SSH" "docker image inspect --format '{{.Id}}' '$FINAL_TAG'")"
  if [ "$id_head" != "$id_worker" ]; then
    echo "FAIL: image ID mismatch — the head node $id_head != the worker $id_worker" >&2
    exit 1
  fi
  echo "   OK: $id_head on both nodes"
  echo "   NOTE: ~/$TARBALL left on BOTH nodes per kit convention (manual cleanup:"
  echo "         ssh $HEAD_SSH rm ~/$TARBALL; ssh $WORKER_SSH rm ~/$TARBALL)"
fi

echo "== done: $FINAL_TAG"
ssh "$HEAD_SSH" "docker image inspect --format '{{.Id}}' '$FINAL_TAG'"
