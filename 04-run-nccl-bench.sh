#!/usr/bin/env bash
# Run NCCL benchmark arms from the head node.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }
# NOTE: named GID_IDX, not GID — GID is a readonly bash builtin (and bash
# overrides an env-passed GID with the real group id, silently).
GID_IDX="${GID_IDX:-3}"

ssh "$CLUSTER_USER@$HEAD_HOST" \
  "HEAD_R1='$HEAD_R1' WORKER_R1='$WORKER_R1' QSFP_IF='$QSFP_IF' NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME' GID_IDX='$GID_IDX' bash -s" <<'REMOTE' \
  || fail "NCCL benchmark failed" "verify 02-setup-cluster-ssh.sh and 03-build-nccl-tests.sh completed"
set -euo pipefail

run_arm() {
  local name="$1" hca="$2" gid_index="$3" ib_disable="$4"
  local out
  out="$HOME/nccl-bench-$name.log"
  local -a envs=(
    -x NCCL_DEBUG=INFO
    -x NCCL_IB_DISABLE="$ib_disable"
    -x NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
    -x UCX_NET_DEVICES="$QSFP_IF"
    -x OMPI_MCA_btl_tcp_if_include="$QSFP_IF"
    -x GLOO_SOCKET_IFNAME="$QSFP_IF"
    -x LD_LIBRARY_PATH="$HOME/nccl/build/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/openmpi/lib"
  )
  if [ -n "$hca" ]; then
    envs+=(-x NCCL_IB_HCA="$hca")
  fi
  if [ -n "$gid_index" ]; then
    envs+=(-x NCCL_IB_GID_INDEX="$gid_index")
  fi
  mpirun -np 2 -H "$HEAD_R1:1,$WORKER_R1:1" \
    --mca plm_rsh_agent "ssh -o BatchMode=yes" \
    "${envs[@]}" \
    "$HOME/nccl-tests/build/all_reduce_perf" -b 256M -e 4G -f 2 -g 1 -n 20 -w 5 \
    >"$out" 2>&1
  local busbw transport
  busbw="$(grep 'Avg bus bandwidth' "$out" | awk '{print $NF}' | tail -1)"
  transport="$(grep -E 'NET/IB|NET/Socket|NCCL_IB_HCA|NCCL_IB_GID_INDEX|Selected|Using network' "$out" | tail -8 | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g')"
  printf '%s\t%s\t%s\n' "$name" "${busbw:-0}" "${transport:-no transport line found}"
  echo "full log: $out" >&2
}

summary="$(mktemp)"
{
  run_arm "A-dual-twin" "rocep1s0f1,roceP2p1s0f1" "" "0"
  run_arm "B-single-gid" "rocep1s0f1" "$GID_IDX" "0"
  run_arm "C-socket" "" "" "1"
} > "$summary"

echo "arm | busbw GB/s | transport line"
sed 's/\t/ | /g' "$summary"

best="$(awk -F '\t' '$1 ~ /^[AB]-/ && $2+0 > b {b=$2+0; n=$1; t=$3} END {printf "%s\t%.3f\t%s\n", n, b, t}' "$summary")"
best_arm="$(printf '%s\n' "$best" | awk -F '\t' '{print $1}')"
best_bw="$(printf '%s\n' "$best" | awk -F '\t' '{print $2}')"
best_transport="$(printf '%s\n' "$best" | cut -f3-)"
socket_bw="$(awk -F '\t' '$1 == "C-socket" {print $2+0}' "$summary")"

awk -v bw="$best_bw" 'BEGIN {exit !(bw >= 15.0)}' \
  || { echo "FAIL: best RDMA bus bandwidth $best_bw GB/s below 15.0 — inspect NCCL debug output" >&2; exit 1; }
printf '%s\n' "$best_transport" | grep -q 'NET/IB' \
  || { echo "FAIL: best arm did not show NET/IB — check NCCL_IB_HCA/GID settings" >&2; exit 1; }
awk -v sock="$socket_bw" -v best="$best_bw" 'BEGIN {exit !(sock < best)}' \
  || { echo "FAIL: socket control was not lower than RDMA best — inspect fabric and NCCL settings" >&2; exit 1; }

echo "ok: best RDMA arm $best_arm at $best_bw GB/s"
echo "Recommended cluster.env lines:"
case "$best_arm" in
  A-dual-twin)
    echo "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1"
    echo "NCCL_IB_GID_INDEX="
    ;;
  B-single-gid)
    echo "NCCL_IB_HCA=rocep1s0f1"
    echo "NCCL_IB_GID_INDEX=$GID_IDX"
    ;;
esac
rm -f "$summary"
REMOTE
