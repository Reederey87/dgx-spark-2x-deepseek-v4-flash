#!/usr/bin/env bash
# Install the kit and systemd user units on both nodes.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

install_node() {
  local host="$1" service="$2"
  # Sync only the runtime kit to the node — leave docs, license, git, and the
  # example config out of the on-node dir.
  rsync -a \
    --exclude='.env.dspark' --exclude='.git' --exclude='.gitignore' \
    --exclude='docs' --exclude='*.md' --exclude='LICENSE' --exclude='NOTICE' \
    --exclude='cluster.env.example' \
    "$KIT/" "$CLUSTER_USER@$host:$KIT_DIR/" \
    || fail "rsync to $host failed" "check control-host SSH access and $KIT_DIR permissions"
  ssh "$CLUSTER_USER@$host" "chmod +x '$KIT_DIR'/*.sh"
  echo "ok: kit synced to $host"

  ssh "$CLUSTER_USER@$host" "mkdir -p ~/.config/systemd/user && cp '$KIT_DIR/$service' ~/.config/systemd/user/ && systemctl --user daemon-reload" \
    || fail "service install failed on $host" "check the systemd user manager for $CLUSTER_USER"
  echo "ok: $service installed on $host"

  if [ "$service" = "vllm-dsv4-head.service" ]; then
    ssh "$CLUSTER_USER@$host" "cp '$KIT_DIR/vllm-dsv4-watchdog.service' '$KIT_DIR/vllm-dsv4-watchdog.timer' ~/.config/systemd/user/ && systemctl --user daemon-reload" \
      || fail "watchdog install failed on $host" "check systemd user manager"
    echo "ok: watchdog units installed on $host"
  fi

  linger="$(ssh "$CLUSTER_USER@$host" "loginctl show-user '$CLUSTER_USER' --property=Linger")" \
    || fail "could not inspect linger on $host" "run 00-node-prep.sh on the node"
  [ "$linger" = "Linger=yes" ] || fail "linger is not enabled on $host" "re-run 00-node-prep.sh"
  echo "ok: linger enabled on $host"
}

install_node "$HEAD_HOST" "vllm-dsv4-head.service"
install_node "$WORKER_HOST" "vllm-dsv4-worker.service"
echo "ok: services installed; units were not enabled or started"
