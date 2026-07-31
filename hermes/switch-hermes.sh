#!/usr/bin/env bash
# switch-hermes.sh — install a Hermes profile on the head node.
#
# Usage (from your control host):
#   bash hermes/switch-hermes.sh                   # install the bundled DeepSeek profile
#   bash hermes/switch-hermes.sh /path/to/other.yaml   # install any other profile
#
# What it does: preflights that the DeepSeek server is healthy on the head
# (127.0.0.1:$API_PORT), backs up the live ~/.hermes/config.yaml (timestamped),
# installs the profile, restarts hermes-gateway, and verifies with a one-shot.
# Unlike the Inkling kit, no de-streaming proxy is needed — the deepseek_v4
# tool parser handles Hermes' streaming requests correctly.
#
# Config: sources runtime/cluster.env when present (HEAD_HOST, API_PORT).
# Env overrides: HEAD_HOST — SSH alias of the head node;
#                HERMES_SSH — SSH target that lands on the account Hermes runs
#                  under (default = HEAD_HOST; set user@host if it differs);
#                API_PORT (default 8000).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/../runtime/cluster.env" ]; then
  # shellcheck disable=SC1091
  source "$HERE/../runtime/cluster.env"
fi
HEAD_HOST="${HEAD_HOST:-head.example.local}"
HERMES_SSH="${HERMES_SSH:-$HEAD_HOST}"
API_PORT="${API_PORT:-8000}"

SRC="${1:-$HERE/config.yaml}"
[ -f "$SRC" ] || { echo "FAIL: profile not found: $SRC" >&2; exit 1; }

echo "== preflight: DeepSeek serving on $HEAD_HOST :$API_PORT"
ssh "$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/health >/dev/null" \
  || { echo "FAIL: nothing healthy on 127.0.0.1:$API_PORT — start the cluster first" >&2; exit 1; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
echo "== backing up live config on $HERMES_SSH -> ~/.hermes/config.yaml.bak-switch-$TS"
ssh "$HERMES_SSH" "cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-switch-$TS 2>/dev/null || true"

echo "== installing profile ($SRC)"
scp -q "$SRC" "$HERMES_SSH:~/.hermes/config.yaml"
ssh "$HERMES_SSH" 'chmod 600 ~/.hermes/config.yaml'

echo "== restarting hermes-gateway"
ssh "$HERMES_SSH" 'systemctl --user reset-failed hermes-gateway 2>/dev/null; systemctl --user restart hermes-gateway; sleep 6; systemctl --user is-active hermes-gateway'

echo "== verify (one-shot)"
ssh "$HERMES_SSH" '~/.local/bin/hermes -z "Reply with exactly: pong"' | tail -2

echo "ok: hermes switched using $SRC (backup of previous: ~/.hermes/config.yaml.bak-switch-$TS)"
