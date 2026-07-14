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

# Non-fatal plumbing-integrity guard for the prefill head-of-line (HoL) fix.
# The HoL fix flows through ${LONG_PREFILL_TOKEN_THRESHOLD:-0}; any config drift or a
# re-render/rebuild that drops it silently reverts to 0 (off), and NO correctness /
# needle / throughput gate catches it (they are all TTFT-blind). This statically asserts
# the fix survives the whole plumbing chain and WARNs — it never fails the boot. See
# docs/07-observability-and-warmup.md.
check_hol_threshold() {
  local warned=0
  # Require a positive integer (0/unset/negative/non-numeric all mean the fix is inert or
  # would break `vllm serve` at container start — catch them all, not just the literal "0").
  if ! printf '%s' "${LONG_PREFILL_TOKEN_THRESHOLD:-}" | grep -qE '^[1-9][0-9]*$'; then
    echo "preflight WARN: prefill HoL fix inert — LONG_PREFILL_TOKEN_THRESHOLD is '${LONG_PREFILL_TOKEN_THRESHOLD:-<unset>}' in cluster.env (want a positive integer, e.g. 4096). Short-request TTFT will regress under a long prefill." >&2
    warned=1
  fi
  if ! grep -q '^LONG_PREFILL_TOKEN_THRESHOLD=' "$KIT/render-env.sh" 2>/dev/null; then
    echo "preflight WARN: render-env.sh no longer emits LONG_PREFILL_TOKEN_THRESHOLD — the rendered .env.dspark will omit it and compose falls back to 0 (off)." >&2
    warned=1
  fi
  # Comment-aware: ignore commented-out compose lines so a stray `# --long-prefill-token-threshold`
  # can't masquerade as the fix being wired.
  if ! grep -vE '^[[:space:]]*#' "$KIT/docker-compose.dspark.yml" 2>/dev/null | grep -q -- '--long-prefill-token-threshold'; then
    echo "preflight WARN: docker-compose.dspark.yml no longer passes --long-prefill-token-threshold — the HoL fix is not wired into the serve command." >&2
    warned=1
  fi
  if [ "$warned" = "0" ]; then
    echo "preflight ok: prefill HoL threshold wired (LONG_PREFILL_TOKEN_THRESHOLD=$LONG_PREFILL_TOKEN_THRESHOLD)"
  fi
  return 0
}

# Fail early on distributed/config invariants whose violations otherwise surface
# as opaque multi-node rendezvous hangs or silent eager decode.
check_serve_invariants() {
  [ "${MASTER_ADDR:-}" = "${HEAD_R1:-}" ] \
    || { echo "preflight FAIL: MASTER_ADDR must equal HEAD_R1 (got ${MASTER_ADDR:-<unset>} vs ${HEAD_R1:-<unset>})" >&2; exit 1; }
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] \
    || { echo "preflight FAIL: GLOO_SOCKET_IFNAME is empty; pin Gloo to the QSFP control rail" >&2; exit 1; }
  local required_capture=$(( MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1) ))
  if [ "$MAX_CUDAGRAPH_CAPTURE_SIZE" -lt "$required_capture" ]; then
    echo "preflight WARN: MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_CUDAGRAPH_CAPTURE_SIZE is below MAX_NUM_SEQS*(MTP_NUM_TOKENS+1)=$required_capture; high-concurrency spec decode may fall back to eager." >&2
  fi
  echo "preflight ok: distributed invariants (Gloo=$GLOO_SOCKET_IFNAME, capture=$MAX_CUDAGRAPH_CAPTURE_SIZE)"
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

# Non-fatal: warn (never fail) if the prefill HoL fix has come unwired.
check_hol_threshold
check_serve_invariants

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
