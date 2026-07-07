#!/usr/bin/env bash
# Role-aware pre-start guards for the vllm-dsv4 units. Runs as the cluster user
# on the node itself (ExecStartPre). Usage: preflight.sh <head|worker>
# Every wait is bounded; systemd TimeoutStartSec must exceed the worst-case sum.
set -euo pipefail

ROLE="${1:?usage: preflight.sh <head|worker>}"
# Boot-critical: don't rely on the systemd user manager's PATH.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

wait_for() { # wait_for <secs> <label> <cmd...>
  local deadline=$(( $(date +%s) + $1 )) label="$2"; shift 2
  until "$@" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "preflight FAIL: timed out waiting for: $label" >&2; exit 1
    fi
    sleep 5
  done
  echo "preflight ok: $label"
}

case "$ROLE" in
  head)   MY_R1="$HEAD_R1";   PEER_R1="$WORKER_R1" ;;
  worker) MY_R1="$WORKER_R1"; PEER_R1="$HEAD_R1" ;;
  *) echo "preflight FAIL: unknown role $ROLE" >&2; exit 1 ;;
esac

# Boot dependencies, in order (linger starts us early in boot).
wait_for "$BOOT_DEP_WAIT_SECS" "QSFP rail-1 IP $MY_R1 assigned" \
  sh -c "ip -4 -o addr show dev $QSFP_IF | grep -Fq '$MY_R1/'"
wait_for "$BOOT_DEP_WAIT_SECS" "docker daemon answering" docker info
[ -e /dev/infiniband ] || { echo "preflight FAIL: /dev/infiniband missing" >&2; exit 1; }

docker image inspect "$DSPARK_VLLM_IMAGE" >/dev/null 2>&1 \
  || { echo "preflight FAIL: image $DSPARK_VLLM_IMAGE not present" >&2; exit 1; }

# Weights present in the token-free cache (served by HF id, offline mode).
MODEL_HUB_DIR="$HF_CACHE/hub/models--${DSPARK_MODEL//\//--}"
[ -d "$MODEL_HUB_DIR" ] && find "$MODEL_HUB_DIR" -name config.json -print -quit | grep -q . \
  || { echo "preflight FAIL: weights not found under $MODEL_HUB_DIR" >&2; exit 1; }

# A stale container from a previous run must not block compose.
docker rm -f vllm-dsv4 >/dev/null 2>&1 || true

if [ "$ROLE" = "head" ]; then
  # Optional memory-pool guard: if a conflicting single-node model service is
  # configured (CONFLICTING_SERVICE in cluster.env) and is active under the
  # cluster user, refuse to start the head — they would fight over the shared
  # unified-memory pool. Empty CONFLICTING_SERVICE disables the check.
  if [ -n "${CONFLICTING_SERVICE:-}" ] && [ "${FORCE:-0}" != "1" ] \
     && systemctl --user is-active --quiet "$CONFLICTING_SERVICE" 2>/dev/null; then
    echo "preflight FAIL: conflicting service '$CONFLICTING_SERVICE' is active on this node." >&2
    echo "Run cluster-enable.sh (stops+disables it), or FORCE=1 to override." >&2
    exit 1
  fi
  # accept-new is safe on this isolated point-to-point fabric and covers a
  # first boot where known_hosts wasn't seeded yet (02-setup-cluster-ssh.sh
  # normally seeds it).
  wait_for "$BOOT_DEP_WAIT_SECS" "peer $PEER_R1 reachable over QSFP" \
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$CLUSTER_USER@$PEER_R1" true
  # Wait for the worker unit — and revive it if it is start-limited/failed
  # (drill-proven: a flapping window can exhaust the worker's StartLimit,
  # leaving head-waits-forever unless someone resets it).
  deadline=$(( $(date +%s) + WORKER_WAIT_SECS ))
  while :; do
    wstate=$(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
      "$CLUSTER_USER@$PEER_R1" systemctl --user is-active vllm-dsv4-worker.service 2>/dev/null || true)
    [ "$wstate" = "active" ] && { echo "preflight ok: worker unit active on $PEER_R1"; break; }
    if [ "$wstate" = "failed" ] || [ "$wstate" = "inactive" ]; then
      echo "preflight: worker unit is '$wstate' — reset+start over QSFP"
      ssh -o BatchMode=yes -o ConnectTimeout=5 "$CLUSTER_USER@$PEER_R1" \
        'systemctl --user reset-failed vllm-dsv4-worker.service 2>/dev/null; systemctl --user start vllm-dsv4-worker.service' || true
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "preflight FAIL: timed out waiting for worker unit on $PEER_R1 (last state: $wstate)" >&2
      exit 1
    fi
    sleep 10
  done
fi

echo "preflight passed ($ROLE)"
