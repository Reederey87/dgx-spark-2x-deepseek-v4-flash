#!/usr/bin/env bash
# Stop the cluster deterministically: head first, then worker.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

echo "== stopping head unit on $HEAD_HOST"
ssh "$CLUSTER_USER@$HEAD_HOST" 'systemctl --user stop vllm-dsv4-head.service' \
  || echo "WARN: head unit stop returned non-zero" >&2
echo "== stopping worker unit on $WORKER_HOST"
ssh "$CLUSTER_USER@$WORKER_HOST" 'systemctl --user stop vllm-dsv4-worker.service' \
  || echo "WARN: worker unit stop returned non-zero" >&2

for h in "$HEAD_HOST" "$WORKER_HOST"; do
  unit=vllm-dsv4-head.service; [ "$h" = "$WORKER_HOST" ] && unit=vllm-dsv4-worker.service
  state=$(ssh "$CLUSTER_USER@$h" "systemctl --user is-active $unit" || true)
  [ "$state" != "active" ] && [ "$state" != "activating" ] \
    || { echo "FAIL: $unit on $h is still $state — not removing its container" >&2; exit 1; }
  ssh "$CLUSTER_USER@$h" 'docker rm -f vllm-dsv4 >/dev/null 2>&1 || true'
done
echo "== confirming API port released on $HEAD_HOST"
if ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 3 http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
  echo "WARN: something still answers on :$API_PORT" >&2; exit 1
fi
echo "== cluster stopped"
