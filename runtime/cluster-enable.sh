#!/usr/bin/env bash
# Make the 2-node cluster the primary local model: enable its systemd user units
# for boot and start it. If CONFLICTING_SERVICE is set in cluster.env, this first
# stops+disables that single-node service on the head (they share the unified
# memory pool — see docs/04). Run from your control host.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

if [ -n "${CONFLICTING_SERVICE:-}" ]; then
  echo "== stopping + disabling conflicting service '$CONFLICTING_SERVICE' on $HEAD_HOST"
  ssh "$CLUSTER_USER@$HEAD_HOST" \
    "systemctl --user stop '$CONFLICTING_SERVICE' 2>/dev/null; systemctl --user disable '$CONFLICTING_SERVICE' 2>/dev/null" || true
  state=$(ssh "$CLUSTER_USER@$HEAD_HOST" "systemctl --user is-active '$CONFLICTING_SERVICE'" 2>/dev/null || true)
  [ "$state" != "active" ] || { echo "FAIL: '$CONFLICTING_SERVICE' still active — cannot cut over (memory-pool contract)" >&2; exit 1; }
fi

echo "== enabling cluster units for boot"
ssh "$CLUSTER_USER@$HEAD_HOST"   'systemctl --user enable vllm-dsv4-head.service && systemctl --user enable --now vllm-dsv4-watchdog.timer'
ssh "$CLUSTER_USER@$WORKER_HOST" 'systemctl --user enable vllm-dsv4-worker.service'

bash "$KIT/start-cluster.sh"

cat <<EOF
== cluster is primary.
  eval:     bash $KIT/eval-cluster.sh
  rollback: bash $KIT/cluster-disable.sh
EOF
