#!/usr/bin/env bash
# Build NCCL and nccl-tests on both nodes.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

build_node() {
  local host="$1"
  echo "== building on $host"
  ssh "$CLUSTER_USER@$host" 'bash -s' <<'REMOTE' \
    || fail "NCCL build failed on $host" "verify git, CUDA, OpenMPI, and build-essential are installed"
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

if [ ! -f ~/nccl/build/lib/libnccl.so ]; then
  [ -d ~/nccl ] || git clone -b v2.30u1 --depth 1 https://github.com/NVIDIA/nccl.git ~/nccl
  cd ~/nccl
  make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
fi
echo "ok: NCCL built"

if [ ! -x ~/nccl-tests/build/all_reduce_perf ]; then
  [ -d ~/nccl-tests ] || git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests
  cd ~/nccl-tests
  make -j MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi NCCL_HOME="$HOME/nccl/build" CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
fi
~/nccl-tests/build/all_reduce_perf --help >/dev/null
echo "ok: nccl-tests built"
cd ~/nccl
printf 'NCCL commit: '
git rev-parse HEAD
REMOTE
}

build_node "$HEAD_HOST"
build_node "$WORKER_HOST"
echo "ok: NCCL tests build complete"
