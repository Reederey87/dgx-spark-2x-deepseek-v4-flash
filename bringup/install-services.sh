#!/usr/bin/env bash
# Install the kit and systemd user units on both nodes.
# Run from the control host. The repo's bringup/ + runtime/ layout is preserved on
# the node under $KIT_DIR (units reference %h/dgx-cluster/runtime/...).
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$KIT/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

install_node() {
  local host="$1" service="$2"
  # Sync the kit to the node (bringup/ + runtime/), preserving structure — leave
  # docs, license, git, the example configs, and any local secrets out of the node.
  rsync -a \
    --exclude='.env.dspark' --exclude='.git' --exclude='.gitignore' \
    --exclude='docs' --exclude='*.md' --exclude='LICENSE' --exclude='NOTICE' \
    --exclude='cluster.env.example' --exclude='notify.env.example' --exclude='notify.env' \
    "$REPO/" "$CLUSTER_USER@$host:$KIT_DIR/" \
    || fail "rsync to $host failed" "check control-host SSH access and $KIT_DIR permissions"
  ssh "$CLUSTER_USER@$host" "find '$KIT_DIR' -name '*.sh' -exec chmod +x {} +"
  echo "ok: kit synced to $host"

  ssh "$CLUSTER_USER@$host" "mkdir -p ~/.config/systemd/user && cp '$KIT_DIR/runtime/$service' ~/.config/systemd/user/ && systemctl --user daemon-reload" \
    || fail "service install failed on $host" "check the systemd user manager for $CLUSTER_USER"
  echo "ok: $service installed on $host"

  if [ "$service" = "vllm-dsv4-head.service" ]; then
    # Head also runs the inference watchdog + the loopback metrics watcher.
    ssh "$CLUSTER_USER@$host" "cp '$KIT_DIR/runtime/vllm-dsv4-watchdog.service' '$KIT_DIR/runtime/vllm-dsv4-watchdog.timer' '$KIT_DIR/runtime/vllm-metrics-watch.service' '$KIT_DIR/runtime/vllm-metrics-watch.timer' ~/.config/systemd/user/ && systemctl --user daemon-reload" \
      || fail "head auxiliary units install failed on $host" "check systemd user manager"
    echo "ok: watchdog + metrics-watch units installed on $host"
  fi

  linger="$(ssh "$CLUSTER_USER@$host" "loginctl show-user '$CLUSTER_USER' --property=Linger")" \
    || fail "could not inspect linger on $host" "run 00-node-prep.sh on the node"
  [ "$linger" = "Linger=yes" ] || fail "linger is not enabled on $host" "re-run 00-node-prep.sh"
  echo "ok: linger enabled on $host"
}

install_node "$HEAD_HOST" "vllm-dsv4-head.service"
install_node "$WORKER_HOST" "vllm-dsv4-worker.service"
echo "ok: services installed; units were not enabled or started"
echo "note: enable the metrics watcher on the head with:  systemctl --user enable --now vllm-metrics-watch.timer"
