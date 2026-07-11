#!/usr/bin/env bash
# Recipe #10 (repurposed) — readiness warm-up after the head (re)starts.
# Fired NON-FATALLY + BACKGROUNDED from vllm-dsv4-head.service ExecStartPost, so
# it never delays the unit's active state nor blocks TimeoutStartSec. Waits for
# /health 200, then primes what the downstream client ACTUALLY hits after a self-heal: the
# decode / first-token path and the deepseek_v4 tool-PARSER path (tool_choice
# =auto). NOTE: with tool_choice=auto vLLM does NOT compile an xgrammar FSM
# (the parser extracts calls from free text), so this warms the parser +
# decode path — not a schema-constrained FSM (see docs/07).
# Steps 3+4 (added 2026-07-11) warm two more jit_monitor-flagged Triton kernel
# families that otherwise first-compile during real traffic 10-30 min post-boot:
# the long-prefill chunk-metadata kernels (_pack_topk_routes_*, _build_prefill_
# chunk_metadata_kernel, _compute_prefill_metadata_kernel) and the spec-decode
# rejection-sampling kernels (sample_recovered_tokens_kernel,
# rejection_random_sample_kernel, only reachable with temperature>0). The Triton
# cache is persistent (TRITON_CACHE_DIR on the hf-cache bind), so these compiles
# are per-new-shape, not per-boot — warmup just ensures the common shapes are
# already compiled before real traffic hits them.
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
plain="$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"warmup ping"}],"max_tokens":8,"temperature":0,"user":"vllm-warmup"}))' "$SERVED_MODEL_NAME")"
if curl -fsS --max-time 90 -H 'Content-Type: application/json' "$api" -d "$plain" >/dev/null 2>&1; then
  log "warmup: plain chat ok"
else
  log "warmup: plain chat failed (non-fatal)"
fi

# 2) tool_choice=auto — warm the deepseek_v4 tool-parser path (what the downstream client uses).
toolreq="$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":64,"temperature":0,"user":"vllm-warmup"}))' "$SERVED_MODEL_NAME")"
if curl -fsS --max-time 90 -H 'Content-Type: application/json' "$api" -d "$toolreq" >/dev/null 2>&1; then
  log "warmup: tool_choice=auto ok"
else
  log "warmup: tool call failed (non-fatal)"
fi

# 3) long-prefill warm — >4096-token user message so the chunked long-prefill
#    path compiles (_pack_topk_routes_*, _build_prefill_chunk_metadata_kernel,
#    _compute_prefill_metadata_kernel). Built in python so the filler text never
#    touches the shell.
longprompt="$(python3 - "$SERVED_MODEL_NAME" <<'PY'
import json, sys

segments = " ".join("lorem ipsum filler segment %d." % i for i in range(1200))
prompt = "Reply with the single word ok. " + segments
payload = {
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "temperature": 0,
    "user": "vllm-warmup",
}
print(json.dumps(payload))
PY
)"
if curl -fsS --max-time 240 -H 'Content-Type: application/json' "$api" -d "$longprompt" >/dev/null 2>&1; then
  log "warmup: long-prefill warm ok"
else
  log "warmup: long-prefill warm failed (non-fatal)"
fi

# 4) sampling/rejection warm — temperature>0 routes through the spec-decode
#    rejection-sampling kernels (sample_recovered_tokens_kernel,
#    rejection_random_sample_kernel), which the temperature:0 greedy warmups
#    above never touch.
samplereq="$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"warmup sampling ping"}],"max_tokens":64,"temperature":1.0,"top_p":1.0,"user":"vllm-warmup"}))' "$SERVED_MODEL_NAME")"
if curl -fsS --max-time 90 -H 'Content-Type: application/json' "$api" -d "$samplereq" >/dev/null 2>&1; then
  log "warmup: sampling/rejection warm ok"
else
  log "warmup: sampling/rejection warm failed (non-fatal)"
fi

log "warmup: done"
exit 0
