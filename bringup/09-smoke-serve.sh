#!/usr/bin/env bash
# Foreground/tmux-free smoke bring-up using docker compose directly.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

# Optional memory-pool guard: refuse to bring up the cluster while a configured
# conflicting single-node model service is active on the head (shared unified
# pool). Empty CONFLICTING_SERVICE (the default) skips this.
if [ -n "${CONFLICTING_SERVICE:-}" ]; then
  if ssh "$CLUSTER_USER@$HEAD_HOST" "systemctl --user is-active --quiet '$CONFLICTING_SERVICE'" 2>/dev/null; then
    echo "FAIL: conflicting service '$CONFLICTING_SERVICE' is active on the head — stop it first (or run cluster-enable.sh)" >&2
    exit 1
  fi
  echo "ok: conflicting service '$CONFLICTING_SERVICE' not active"
fi

ssh "$CLUSTER_USER@$WORKER_HOST" "cd '$KIT_DIR/runtime' && bash render-env.sh worker && docker compose --env-file .env.dspark -f docker-compose.dspark.yml up -d" \
  || fail "worker compose start failed" "inspect docker compose logs on worker"
echo "ok: worker compose started"

ssh "$CLUSTER_USER@$HEAD_HOST" "cd '$KIT_DIR/runtime' && bash render-env.sh head && docker compose --env-file .env.dspark -f docker-compose.dspark.yml up -d" \
  || fail "head compose start failed" "inspect docker compose logs on head"
echo "ok: head compose started"

deadline=$(( $(date +%s) + 1800 ))
while :; do
  if ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    echo "ok: API health"
    break
  fi
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    running="$(ssh "$CLUSTER_USER@$host" "docker inspect -f '{{.State.Running}}' vllm-dsv4 2>/dev/null" || true)"
    if [ "$running" != "true" ]; then
      echo "FAIL: vllm-dsv4 exited on $host — recent logs:" >&2
      ssh "$CLUSTER_USER@$host" "docker logs --tail 60 vllm-dsv4" >&2 || true
      exit 1
    fi
  done
  [ "$(date +%s)" -lt "$deadline" ] || fail "timed out waiting for /health" "inspect docker logs on both nodes"
  sleep 15
done

payload="$(mktemp)"
response="$(mktemp)"
cat > "$payload" <<JSON
{"model":"$SERVED_MODEL_NAME","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":8,"temperature":0}
JSON
scp "$payload" "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-smoke-chat.json" >/dev/null
ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS -o /tmp/dspark-smoke-response.json -w '%{http_code}' -H 'Content-Type: application/json' --data @/tmp/dspark-smoke-chat.json http://127.0.0.1:$API_PORT/v1/chat/completions" > "$response" \
  || fail "chat completion request failed" "inspect vLLM logs on head"
status="$(cat "$response")"
[ "$status" = "200" ] || fail "chat completion returned HTTP $status" "inspect /tmp/dspark-smoke-response.json on head"
rm -f "$payload" "$response"
echo "ok: chat completion"

echo "KV cache log lines:"
ssh "$CLUSTER_USER@$HEAD_HOST" "docker logs vllm-dsv4 2>&1 | grep -iE 'kv cache|GPU KV cache size|token' | grep -iE 'cache' | tail -5" || true

for host in "$HEAD_HOST" "$WORKER_HOST"; do
  echo "listeners NOT on loopback/QSFP (review!) on $host:"
  ssh "$CLUSTER_USER@$host" "ss -tlnp 2>/dev/null | grep -vE '127.0.0.1|::1|192.168.17[78]' || true"
  echo "cluster listeners on $host:"
  ssh "$CLUSTER_USER@$host" "ss -tlnp | grep -E ':$API_PORT|:$MASTER_PORT' || true"
done

cat <<EOF
ok: smoke serve complete
teardown:
  ssh $CLUSTER_USER@$HEAD_HOST 'cd $KIT_DIR/runtime && docker compose --env-file .env.dspark -f docker-compose.dspark.yml down'
  ssh $CLUSTER_USER@$WORKER_HOST 'cd $KIT_DIR/runtime && docker compose --env-file .env.dspark -f docker-compose.dspark.yml down'
EOF
