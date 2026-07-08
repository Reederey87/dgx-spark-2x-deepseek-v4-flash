#!/usr/bin/env bash
# Print lightweight cluster metrics from both nodes.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

for host in "$HEAD_HOST" "$WORKER_HOST"; do
  echo "== $host nvidia-smi"
  ssh "$CLUSTER_USER@$host" 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader' || true
  echo "== $host free -g"
  ssh "$CLUSTER_USER@$host" 'free -g | head -2' || true
done

echo "== $HEAD_HOST vLLM metrics"
ssh "$CLUSTER_USER@$HEAD_HOST" "curl -s http://127.0.0.1:$API_PORT/metrics | grep -E 'vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc)' | head -10" || true
