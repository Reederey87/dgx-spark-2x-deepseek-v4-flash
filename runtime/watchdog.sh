#!/usr/bin/env bash
# Inference-level watchdog for the head node (drill-proven need: the API
# process survives a lost TP peer, so /health lies while inference hangs).
# Logic: if /health answers (engine fully initialized) but a real 1-token
# completion times out, restart the head unit. During startup /health is
# down, so this never fires mid-load.
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

curl -fsS --max-time 5 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 || exit 0  # not up yet — not our problem

if curl -fsS --max-time 90 -H 'Content-Type: application/json' \
     -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" \
     "http://127.0.0.1:$API_PORT/v1/chat/completions" >/dev/null 2>&1; then
  exit 0
fi

# Bounce the PAIR in strict order (drill-proven twice):
#   1. STOP the head first — its stale rendezvous store on :25000 must be gone
#      before the worker restarts, or the fresh worker joins the dead group
#      and zombifies (never exits, Restart= can't help).
#   2. Restart the worker — headless vLLM waits for a master to appear.
#   3. Start the head — preflight re-checks the worker, then re-rendezvous.
# reset-failed throughout: flap windows exhaust StartLimitBurst, and this
# watchdog's 5-min period is the real rate limiter.
echo "watchdog: /health OK but inference timed out — bouncing pair (head stop, worker restart, head start)" >&2
systemctl --user stop vllm-dsv4-head.service
for _ in 1 2 3 4 5 6; do
  docker inspect vllm-dsv4 >/dev/null 2>&1 || break
  sleep 5
done
docker rm -f vllm-dsv4 >/dev/null 2>&1 || true
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  "$CLUSTER_USER@$WORKER_R1" 'systemctl --user reset-failed vllm-dsv4-worker.service 2>/dev/null; systemctl --user restart vllm-dsv4-worker.service' \
  || echo "watchdog: worker restart over QSFP failed — starting head anyway (preflight will revive it)" >&2
systemctl --user reset-failed vllm-dsv4-head.service 2>/dev/null || true
systemctl --user start vllm-dsv4-head.service
