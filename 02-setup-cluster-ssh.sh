#!/usr/bin/env bash
# Set up node-to-node SSH over QSFP rail IPs.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

ensure_key() {
  local host="$1"
  ssh "$CLUSTER_USER@$host" \
    '[ -f ~/.ssh/id_ed25519 ] || (mkdir -p ~/.ssh && chmod 700 ~/.ssh && ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519)' \
    || fail "could not ensure SSH key on $host" "check Mac SSH access to $host as $CLUSTER_USER"
  echo "ok: key exists on $host"
}

fetch_pubkey() {
  local host="$1"
  ssh "$CLUSTER_USER@$host" 'cat ~/.ssh/id_ed25519.pub' \
    || fail "could not read pubkey on $host" "run key generation again"
}

append_peer_key() {
  local host="$1" pubkey="$2"
  local opts='from="192.168.177.*,192.168.178.*",no-port-forwarding,no-agent-forwarding,no-X11-forwarding'
  ssh "$CLUSTER_USER@$host" "PUBKEY='$pubkey' OPTS='$opts' bash -s" <<'REMOTE' \
    || fail "could not update authorized_keys on $host" "check remote ~/.ssh permissions"
set -euo pipefail
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
if ! grep -qF "$PUBKEY" ~/.ssh/authorized_keys; then
  printf '%s %s\n' "$OPTS" "$PUBKEY" >> ~/.ssh/authorized_keys
fi
REMOTE
  echo "ok: peer key authorized on $host"
}

scan_peers() {
  local host="$1" peer_r1="$2" peer_r2="$3"
  ssh "$CLUSTER_USER@$host" "PEER_R1='$peer_r1' PEER_R2='$peer_r2' bash -s" <<'REMOTE' \
    || fail "could not pre-populate known_hosts on $host" "check QSFP reachability and ssh-keyscan"
set -euo pipefail
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/known_hosts
ssh-keyscan -T 5 "$PEER_R1" "$PEER_R2" >> ~/.ssh/known_hosts
tmp="$(mktemp)"
sort -u ~/.ssh/known_hosts > "$tmp"
mv "$tmp" ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
REMOTE
  echo "ok: known_hosts on $host"
}

ensure_key "$HEAD_HOST"
ensure_key "$WORKER_HOST"

HEAD_PUBKEY="$(fetch_pubkey "$HEAD_HOST")"
WORKER_PUBKEY="$(fetch_pubkey "$WORKER_HOST")"

append_peer_key "$WORKER_HOST" "$HEAD_PUBKEY"
append_peer_key "$HEAD_HOST" "$WORKER_PUBKEY"

scan_peers "$HEAD_HOST" "$WORKER_R1" "$WORKER_R2"
scan_peers "$WORKER_HOST" "$HEAD_R1" "$HEAD_R2"

head_to_worker="$(ssh "$CLUSTER_USER@$HEAD_HOST" "ssh -o BatchMode=yes '$CLUSTER_USER@$WORKER_R1' hostname")" \
  || fail "head cannot SSH to worker over $WORKER_R1" "rerun this script and verify QSFP rail-1"
[ -n "$head_to_worker" ] || fail "empty hostname from worker over $WORKER_R1" "verify QSFP rail-1 SSH"
echo "ok: head -> worker SSH over QSFP"

worker_to_head="$(ssh "$CLUSTER_USER@$WORKER_HOST" "ssh -o BatchMode=yes '$CLUSTER_USER@$HEAD_R1' hostname")" \
  || fail "worker cannot SSH to head over $HEAD_R1" "rerun this script and verify QSFP rail-1"
[ -n "$worker_to_head" ] || fail "empty hostname from head over $HEAD_R1" "verify QSFP rail-1 SSH"
echo "ok: worker -> head SSH over QSFP"
