#!/usr/bin/env bash
# Verify the QSFP/RDMA fabric from your control host.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

check_node() {
  local host="$1" my_r1="$2" my_r2="$3"
  echo "== $host"
  ssh "$CLUSTER_USER@$host" \
    "QSFP_IF='$QSFP_IF' QSFP_IF2='$QSFP_IF2' MY_R1='$my_r1' MY_R2='$my_r2' MTU='$MTU' bash -s" <<'REMOTE' \
    || exit 1
set -euo pipefail
fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

ip -br addr show "$QSFP_IF" | grep -q "$MY_R1" \
  || fail "$QSFP_IF missing $MY_R1" "run 00-node-prep.sh on this node"
echo "ok: $QSFP_IF has $MY_R1"

for iface in "$QSFP_IF" "$QSFP_IF2"; do
  mtu="$(ip link show "$iface" | grep -o 'mtu [0-9]*' | awk '{print $2}')"
  [ "$mtu" = "$MTU" ] || fail "$iface mtu is $mtu" "expected $MTU"
done
echo "ok: QSFP MTU $MTU"

# Capture once — piping into grep -q under pipefail SIGPIPEs the producer.
IBDEV_OUT="$(ibdev2netdev)"
printf '%s\n' "$IBDEV_OUT" | grep -q 'rocep1s0f1.*(Up)' \
  || fail "rocep1s0f1 not Up" "check RoCE device state"
printf '%s\n' "$IBDEV_OUT" | grep -q 'roceP2p1s0f1.*(Up)' \
  || fail "roceP2p1s0f1 not Up" "check RoCE device state"
echo "ok: RoCE devices Up"

docker info >/dev/null || fail "docker info failed" "ensure the cluster user is in the docker group and docker is running"
echo "ok: docker info"

if ip route show default | grep -Eq "$QSFP_IF|$QSFP_IF2"; then
  fail "default route uses QSFP iface" "remove default routing from QSFP rail interfaces"
fi
echo "ok: no default route via QSFP"

echo "RoCEv2 IPv4 GID candidates:"
for dev in rocep1s0f1 roceP2p1s0f1; do
  for gid_file in /sys/class/infiniband/"$dev"/ports/1/gids/*; do
    [ -e "$gid_file" ] || continue
    idx="${gid_file##*/}"
    type_file="/sys/class/infiniband/$dev/ports/1/gid_attrs/types/$idx"
    type="$(cat "$type_file" 2>/dev/null || true)"
    gid="$(cat "$gid_file" 2>/dev/null || true)"
    # sysfs prints GIDs uncompressed — match ':ffff:' not '::ffff:'
    if printf '%s\n' "$type" | grep -qi 'RoCE v2' && printf '%s\n' "$gid" | grep -qi ':ffff:'; then
      echo "  $dev index=$idx type=$type gid=$gid"
    fi
  done
done
REMOTE
}

check_ping() {
  local from_host="$1" target="$2"
  ssh "$CLUSTER_USER@$from_host" "ping -c2 -W2 -M do -s 8972 '$target' >/dev/null" \
    || fail "jumbo ping from $from_host to $target failed" "verify QSFP addressing, MTU, and link state"
  echo "ok: jumbo ping $from_host -> $target"
}

check_node "$HEAD_HOST" "$HEAD_R1" "$HEAD_R2"
check_node "$WORKER_HOST" "$WORKER_R1" "$WORKER_R2"
check_ping "$HEAD_HOST" "$WORKER_R1"
check_ping "$HEAD_HOST" "$WORKER_R2"
check_ping "$WORKER_HOST" "$HEAD_R1"
check_ping "$WORKER_HOST" "$HEAD_R2"

echo "ok: fabric verification complete"
