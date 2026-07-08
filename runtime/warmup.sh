#!/usr/bin/env bash
# Recipe #10 (repurposed) — readiness warm-up after the head (re)starts.
# Fired NON-FATALLY + BACKGROUNDED from vllm-dsv4-head.service ExecStartPost, so
# it never delays the unit's active state nor blocks TimeoutStartSec. Waits for
# /health 200, then primes what a typical downstream client hits after a self-heal:
# the decode / first-token path and the deepseek_v4 tool-PARSER path (tool_choice
# =auto). NOTE: a tool_choice=auto client does NOT compile an xgrammar FSM (the
# parser extracts calls from free text), so this warms the parser + decode path —
# not a schema-constrained FSM (see docs/07-observability-and-warmup.md).
# Purely additive; a failure here can never affect serving.
set -uo pipefail

KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

LOG_DIR="$KIT/logs"; mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG="$LOG_DIR/warmup.log"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" >> "$LOG"; }

api="http://127.0.0.1:$API_PORT/v1/chat/completions"
health="http://127.0.0.1:$API_PORT/health"

# Wait for health, bounded (boot-to-serving is ~6 min on this cluster).
deadline=$(( $(date +%s) + 600 ))
until curl -fsS --max-time 5 "$health" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "warmup abort: /health not ready within 600 s"
    exit 0
  fi
  sleep 10
done
log "warmup: /health OK — priming decode + tool-parser paths"

# 1) plain chat — warm the decode / first-token path.
plain="$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"warmup ping"}],"max_tokens":8,"temperature":0}))' "$SERVED_MODEL_NAME")"
if curl -fsS --max-time 90 -H 'Content-Type: application/json' "$api" -d "$plain" >/dev/null 2>&1; then
  log "warmup: plain chat ok"
else
  log "warmup: plain chat failed (non-fatal)"
fi

# 2) tool_choice=auto — warm the deepseek_v4 tool-parser path (the tool_choice=auto path).
toolreq="$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":64,"temperature":0}))' "$SERVED_MODEL_NAME")"
if curl -fsS --max-time 90 -H 'Content-Type: application/json' "$api" -d "$toolreq" >/dev/null 2>&1; then
  log "warmup: tool_choice=auto ok"
else
  log "warmup: tool call failed (non-fatal)"
fi

log "warmup: done"
exit 0
