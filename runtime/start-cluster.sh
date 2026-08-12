#!/usr/bin/env bash
# Start the 2-node cluster: worker unit first, then head, then poll /health.
# Run from your control host (uses ssh names; nodes talk QSFP among themselves).
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

HEALTH_WAIT_SECS="${HEALTH_WAIT_SECS:-1500}"   # engine + weights load can take 10-20 min

echo "== starting worker unit on $WORKER_HOST"
# reset-failed first: a unit in start-limit-hit (the flapping case preflight repairs)
# would otherwise make systemctl start fail and set -e abort bring-up before the head
# preflight's recovery path can ever run.
ssh "$CLUSTER_USER@$WORKER_HOST" 'systemctl --user reset-failed vllm-dsv4-worker.service 2>/dev/null || true; systemctl --user start vllm-dsv4-worker.service'

echo "== starting head unit on $HEAD_HOST (its preflight waits for the worker)"
ssh "$CLUSTER_USER@$HEAD_HOST" 'systemctl --user reset-failed vllm-dsv4-head.service 2>/dev/null || true; systemctl --user start vllm-dsv4-head.service'

echo "== waiting for API health on $HEAD_HOST:$API_PORT (up to ${HEALTH_WAIT_SECS}s)"
deadline=$(( $(date +%s) + HEALTH_WAIT_SECS ))
while :; do
  if ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    echo "== cluster is serving. Models:"
    ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS http://127.0.0.1:$API_PORT/v1/models"
    echo
    # Bounded inference smoke: /health proves the API answers, not that the engine can
    # still generate. One short non-thinking completion, hard-capped; on failure dump
    # both nodes' recent unit journals (same pattern as the unit-failure path below).
    SMOKE_WAIT_SECS="${SMOKE_WAIT_SECS:-120}"
    echo "== inference smoke probe (thinking off, max_tokens 32, ${SMOKE_WAIT_SECS}s cap)"
    probe_payload="$(SERVED_MODEL_NAME="$SERVED_MODEL_NAME" python3 - <<'PY'
import json, os
print(json.dumps({"model": os.environ["SERVED_MODEL_NAME"],
                  "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                  "temperature": 0, "max_tokens": 32,
                  "chat_template_kwargs": {"thinking": False}}))
PY
)"
    probe_ok=0
    probe_resp="$(ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time $SMOKE_WAIT_SECS -H 'Content-Type: application/json' --data @- http://127.0.0.1:$API_PORT/v1/chat/completions" <<<"$probe_payload" 2>/dev/null)" || probe_ok=1
    if [ "$probe_ok" = "0" ]; then
      PROBE_RESP="$probe_resp" python3 - <<'PY' || probe_ok=1
import json, os
data = json.loads(os.environ["PROBE_RESP"])
choice = (data.get("choices") or [{}])[0]
content = (choice.get("message") or {}).get("content") or ""
if not content.strip() and not choice.get("finish_reason"):
    raise SystemExit(1)
PY
    fi
    if [ "$probe_ok" != "0" ]; then
      echo "ERROR: /health is up but the inference smoke probe failed. Recent unit logs:" >&2
      for h in "$HEAD_HOST" "$WORKER_HOST"; do
        unit=vllm-dsv4-head.service; [ "$h" = "$WORKER_HOST" ] && unit=vllm-dsv4-worker.service
        echo "--- $unit on $h:" >&2
        ssh "$CLUSTER_USER@$h" "journalctl --user -u $unit -n 40 --no-pager" >&2 || true
      done
      exit 1
    fi
    echo "ok: inference smoke probe"
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
