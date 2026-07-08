#!/usr/bin/env bash
# 00-node-prep.sh — the ONE batched-sudo step. Run interactively ON each node,
# once per node, with its role. Copy this kit to the node first (as your normal
# login user — the 'nvidia' user does not exist yet), then:
#
#   ssh -t "$HEAD_HOST"   'cd ~/dgx-cluster && bash bringup/00-node-prep.sh head'
#   ssh -t "$WORKER_HOST" 'cd ~/dgx-cluster && bash bringup/00-node-prep.sh worker'
#
# Idempotent. Prints the change list, asks one confirmation, needs sudo once.
# Creates the shared unprivileged CLUSTER_USER (docker group only — NEVER sudo),
# enables linger, authorizes your control host's SSH key, writes the QSFP netplan
# (static rail IPs, jumbo MTU), and installs the build/RDMA apt deps.
#
# Optional firmware pass (do this FIRST if your two nodes' firmware differ —
# mismatched NIC/SoC firmware is a documented cause of collapsed NCCL bandwidth):
#   bash bringup/00-node-prep.sh <head|worker> --firmware   # then reboot, then re-run
#                                                    # without --firmware.
set -euo pipefail

ROLE="${1:?usage: 00-node-prep.sh <head|worker> [--firmware]}"
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

APT_PKGS="libopenmpi-dev openmpi-bin build-essential git iperf3 zstd ibverbs-utils infiniband-diags"

case "$ROLE" in
  head)   R1_CIDR="${HEAD_R1}/24";   R2_CIDR="${HEAD_R2}/24" ;;
  worker) R1_CIDR="${WORKER_R1}/24"; R2_CIDR="${WORKER_R2}/24" ;;
  *) echo "ERROR: role must be 'head' or 'worker' (got '$ROLE')" >&2; exit 1 ;;
esac

if [ "${2:-}" = "--firmware" ]; then
  echo ">> Firmware pass on $(hostname) (requires sudo; a reboot will likely follow)."
  sudo fwupdmgr refresh --force || true
  sudo fwupdmgr update
  echo ">> If an update was applied: sudo reboot, then re-run this script without --firmware."
  exit 0
fi

# Control-host public key that will be authorized for passwordless SSH to the
# cluster user. Set CONTROL_HOST_PUBKEY in cluster.env or export it before running:
#   export CONTROL_HOST_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"
PUBKEY="${CONTROL_HOST_PUBKEY:?set CONTROL_HOST_PUBKEY in cluster.env or the environment (the SSH public key of your control host)}"

# The human login user on THIS node to grant docker access as a convenience.
# Defaults to whoever is running the script; set QOL_USER= to skip, or override.
QOL_USER="${QOL_USER-$(id -un)}"
case "$QOL_USER" in root|"$CLUSTER_USER"|"") QOL_USER="" ;; esac

NETPLAN=$(cat <<EOF
network:
  version: 2
  ethernets:
    ${QSFP_IF}:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: ${MTU}
      addresses: [${R1_CIDR}]
    ${QSFP_IF2}:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: ${MTU}
      addresses: [${R2_CIDR}]
EOF
)

CLUSTER_HOME="/home/${CLUSTER_USER}"
if [ -n "$QOL_USER" ]; then
  QOL_LINE="Add '${QOL_USER}' to 'docker' group (convenience)"
  QOL_ROLLBACK="  sudo gpasswd -d ${QOL_USER} docker   # only if you want the convenience grant undone too"
else
  QOL_LINE="(no convenience docker grant)"
  QOL_ROLLBACK=""
fi
cat <<SUMMARY
== 00-node-prep ($ROLE) on $(hostname) will:
  1. Create user '${CLUSTER_USER}' (no password login, NO sudo group), add to 'docker' group
  2. ${QOL_LINE}
  3. loginctl enable-linger ${CLUSTER_USER}
  4. Authorize your control-host key -> ${CLUSTER_HOME}/.ssh/authorized_keys
  5. Write /etc/netplan/40-cx7.yaml (below), validate with 'netplan generate', apply:
${NETPLAN}
  6. apt-get install: ${APT_PKGS}
Rollback (full): sudo rm /etc/netplan/40-cx7.yaml && sudo netplan apply;
  sudo loginctl disable-linger ${CLUSTER_USER}; sudo userdel -r ${CLUSTER_USER}   # destroys ${CLUSTER_HOME} (kit, caches)!
${QOL_ROLLBACK}
SUMMARY
read -r -p "Proceed? [y/N] " ans
[ "${ans}" = "y" ] || { echo "aborted"; exit 1; }

sudo CLUSTER_USER="$CLUSTER_USER" QOL_USER="$QOL_USER" PUBKEY="$PUBKEY" NETPLAN="$NETPLAN" APT_PKGS="$APT_PKGS" \
  bash -euo pipefail -c '
  id -u "$CLUSTER_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$CLUSTER_USER"
  usermod -aG docker "$CLUSTER_USER"
  [ -n "$QOL_USER" ] && usermod -aG docker "$QOL_USER" || true
  loginctl enable-linger "$CLUSTER_USER"

  CLUSTER_HOME="/home/$CLUSTER_USER"
  install -d -m 700 -o "$CLUSTER_USER" -g "$CLUSTER_USER" "$CLUSTER_HOME/.ssh"
  touch "$CLUSTER_HOME/.ssh/authorized_keys"
  grep -qF "$PUBKEY" "$CLUSTER_HOME/.ssh/authorized_keys" 2>/dev/null \
    || echo "$PUBKEY" >> "$CLUSTER_HOME/.ssh/authorized_keys"
  chown "$CLUSTER_USER:$CLUSTER_USER" "$CLUSTER_HOME/.ssh/authorized_keys"
  chmod 600 "$CLUSTER_HOME/.ssh/authorized_keys"

  printf "%s\n" "$NETPLAN" > /etc/netplan/40-cx7.yaml
  chmod 600 /etc/netplan/40-cx7.yaml
  netplan generate
  netplan apply

  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $APT_PKGS
'

echo "== done. Quick self-check:"
ip -br addr show dev "$QSFP_IF"
ip -br addr show dev "$QSFP_IF2"
id "$CLUSTER_USER"
echo "== Now run 01-verify-fabric.sh from your control host."
