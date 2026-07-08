#!/usr/bin/env bash
# Start the 2-node cluster: worker unit first, then head, then poll /health.
# Run from your control host (uses ssh names; nodes talk QSFP among themselves).
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

HEALTH_WAIT_SECS="${HEALTH_WAIT_SECS:-1500}"   # engine + weights load can take 10-20 min

echo "== starting worker unit on $WORKER_HOST"
ssh "$CLUSTER_USER@$WORKER_HOST" 'systemctl --user start vllm-dsv4-worker.service'

echo "== starting head unit on $HEAD_HOST (its preflight waits for the worker)"
ssh "$CLUSTER_USER@$HEAD_HOST" 'systemctl --user start vllm-dsv4-head.service'

echo "== waiting for API health on $HEAD_HOST:$API_PORT (up to ${HEALTH_WAIT_SECS}s)"
deadline=$(( $(date +%s) + HEALTH_WAIT_SECS ))
while :; do
  if ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    echo "== cluster is serving. Models:"
    ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS http://127.0.0.1:$API_PORT/v1/models"
    echo
    exit 0
  fi
  for h in "$HEAD_HOST" "$WORKER_HOST"; do
    unit=vllm-dsv4-head.service; [ "$h" = "$WORKER_HOST" ] && unit=vllm-dsv4-worker.service
    state=$(ssh "$CLUSTER_USER@$h" "systemctl --user is-active $unit" 2>/dev/null || true)
    if [ "$state" = "failed" ]; then
      echo "ERROR: $unit on $h entered failed state. Recent logs:" >&2
      ssh "$CLUSTER_USER@$h" "journalctl --user -u $unit -n 40 --no-pager" >&2 || true
      exit 1
    fi
  done
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERROR: timed out waiting for /health. Head logs:" >&2
    ssh "$CLUSTER_USER@$HEAD_HOST" "docker logs --tail 60 vllm-dsv4" >&2 || true
    exit 1
  fi
  sleep 15
done
