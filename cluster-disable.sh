#!/usr/bin/env bash
# Stop the cluster and disable its systemd user units. If CONFLICTING_SERVICE is
# set in cluster.env, re-enable + start that single-node service on the head
# afterward (the inverse of cluster-enable.sh). Run from your control host.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

bash "$KIT/stop-cluster.sh"

echo "== disabling cluster units"
ssh "$CLUSTER_USER@$HEAD_HOST"   'systemctl --user disable --now vllm-dsv4-watchdog.timer; systemctl --user disable vllm-dsv4-head.service' || true
ssh "$CLUSTER_USER@$WORKER_HOST" 'systemctl --user disable vllm-dsv4-worker.service' || true

if [ -n "${CONFLICTING_SERVICE:-}" ]; then
  echo "== re-enabling + starting '$CONFLICTING_SERVICE' on $HEAD_HOST"
  ssh "$CLUSTER_USER@$HEAD_HOST" "systemctl --user enable '$CONFLICTING_SERVICE' && systemctl --user start '$CONFLICTING_SERVICE'"
  sleep 5
  ssh "$CLUSTER_USER@$HEAD_HOST" "systemctl --user is-active '$CONFLICTING_SERVICE'" \
    || { echo "ERROR: '$CONFLICTING_SERVICE' did not come back — check journalctl --user -u $CONFLICTING_SERVICE" >&2; exit 1; }
fi

echo "== cluster disabled."
