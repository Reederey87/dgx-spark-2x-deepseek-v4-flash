#!/usr/bin/env bash
# Require a continuous quiet host interval before exposing any A/B/A arm.
# This catches delayed swap writeback from opening evals, boots, or prior arms.
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT="$(cd "$BENCH_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$KIT/runtime/cluster.env"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

OUTPUT="${1:-}"
LABEL="${2:-pre-arm}"
[ -n "$OUTPUT" ] || { echo "usage: wait-host-quiet.sh <output.jsonl> [label]" >&2; exit 2; }

QUIET_SECONDS="${QUIET_SECONDS:-60}"
QUIET_DEADLINE_SECONDS="${QUIET_DEADLINE_SECONDS:-300}"
QUIET_SAMPLE_SECONDS="${QUIET_SAMPLE_SECONDS:-5}"
QUIET_MIN_KIB="${QUIET_MIN_KIB:-2359296}"       # 2.25 GiB; W6 admission floor
QUIET_HARD_MIN_KIB="${QUIET_HARD_MIN_KIB:-1048576}"
SWAP_IO_WARN_PAGES="${SWAP_IO_WARN_PAGES:-256}"
SWAP_IO_WARN_TICKS="${SWAP_IO_WARN_TICKS:-3}"
SWAP_IO_CRIT_PAGES="${SWAP_IO_CRIT_PAGES:-4096}"
for value in QUIET_SECONDS QUIET_DEADLINE_SECONDS QUIET_SAMPLE_SECONDS QUIET_MIN_KIB \
  QUIET_HARD_MIN_KIB SWAP_IO_WARN_PAGES SWAP_IO_WARN_TICKS SWAP_IO_CRIT_PAGES; do
  current="${!value}"
  case "$current" in (*[!0-9]*|'') echo "FAIL: $value must be a non-negative integer" >&2; exit 2;; esac
done
[ "$QUIET_SECONDS" -gt 0 ] && [ "$QUIET_SAMPLE_SECONDS" -gt 0 ] \
  || { echo "FAIL: quiet/sample duration must be positive" >&2; exit 2; }
[ "$QUIET_DEADLINE_SECONDS" -ge "$QUIET_SECONDS" ] \
  || { echo "FAIL: quiet deadline must be >= quiet duration" >&2; exit 2; }

mkdir -p "$(dirname "$OUTPUT")"
: > "$OUTPUT"
last_uint() { sed -nE 's/^[[:space:]]*([0-9]+)[[:space:]]*$/\1/p' | tail -1; }
service_state() {
  local host="$1" role="$2" state attempt
  for attempt in 1 2; do
    state="$(ssh -T "$CLUSTER_USER@$host" \
      "systemctl --user is-active vllm-dsv4-$role 2>/dev/null || true" 2>/dev/null \
      | tr -d '\r' | sed -nE '/^(active|inactive|failed|activating|deactivating)$/p' | head -1)"
    [ -z "$state" ] || { printf '%s\n' "$state"; return 0; }
    [ "$attempt" = 2 ] || sleep 1
  done
  return 1
}
append_sample() {
  uv run python - "$OUTPUT" "$LABEL" "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys, time
path, label, host, mem, swap_delta, service, fatal = sys.argv[1:]
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label, "host": host, "mem_available_kib": int(mem),
        "swap_delta_pages": int(swap_delta), "service": service,
        "fatal_log_matches": int(fatal)}) + "\n")
PY
}

prev_head="$(ssh -T "$CLUSTER_USER@$HEAD_HOST" \
  "awk '/^(pswpin|pswpout) /{s+=\$2} END{print s+0}' /proc/vmstat" | last_uint)"
prev_worker="$(ssh -T "$CLUSTER_USER@$WORKER_HOST" \
  "awk '/^(pswpin|pswpout) /{s+=\$2} END{print s+0}' /proc/vmstat" | last_uint)"
[ -n "$prev_head" ] && [ -n "$prev_worker" ] \
  || { echo "FAIL: $LABEL initial swap telemetry unreadable" >&2; exit 1; }

started="$SECONDS"
quiet_started="$SECONDS"
swap_streak=0
while :; do
  clean_sample=1
  max_swap=0
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    role=$([ "$host" = "$HEAD_HOST" ] && echo head || echo worker)
    mem="$(ssh -T "$CLUSTER_USER@$host" "awk '/^MemAvailable:/{print \$2}' /proc/meminfo" | last_uint)"
    swap_now="$(ssh -T "$CLUSTER_USER@$host" \
      "awk '/^(pswpin|pswpout) /{s+=\$2} END{print s+0}' /proc/vmstat" | last_uint)"
    state="$(service_state "$host" "$role" || true)"
    fatal="$(ssh -T "$CLUSTER_USER@$host" \
      "docker logs --since 15s vllm-dsv4 2>&1 | grep -Eic 'Xid|illegal memory access|EngineCore failed|CUDA error|Traceback' || true" \
      | last_uint)"
    case "$mem" in (*[!0-9]*|'') echo "FAIL: $LABEL $host memory telemetry unreadable" >&2; exit 1;; esac
    case "$swap_now" in (*[!0-9]*|'') echo "FAIL: $LABEL $host swap telemetry unreadable" >&2; exit 1;; esac
    case "$fatal" in (*[!0-9]*|'') echo "FAIL: $LABEL $host fatal-log telemetry unreadable" >&2; exit 1;; esac
    [ "$state" = active ] || { echo "FAIL: $LABEL $host service is '$state'" >&2; exit 1; }
    [ "$fatal" -eq 0 ] || { echo "FAIL: $LABEL $host emitted fatal GPU/engine log" >&2; exit 1; }
    [ "$mem" -ge "$QUIET_HARD_MIN_KIB" ] \
      || { echo "FAIL: $LABEL $host MemAvailable fell below hard floor" >&2; exit 1; }
    if [ "$host" = "$HEAD_HOST" ]; then
      previous="$prev_head"; prev_head="$swap_now"
    else
      previous="$prev_worker"; prev_worker="$swap_now"
    fi
    swap_delta=$(( swap_now - previous )); [ "$swap_delta" -ge 0 ] || swap_delta=0
    append_sample "$host" "$mem" "$swap_delta" "$state" "$fatal"
    [ "$swap_delta" -le "$max_swap" ] || max_swap="$swap_delta"
    [ "$mem" -ge "$QUIET_MIN_KIB" ] || clean_sample=0
    [ "$swap_delta" -le "$SWAP_IO_WARN_PAGES" ] || clean_sample=0
    [ "$swap_delta" -le "$SWAP_IO_CRIT_PAGES" ] \
      || { echo "FAIL: $LABEL $host swap delta exceeded $SWAP_IO_CRIT_PAGES pages" >&2; exit 1; }
  done
  if [ "$max_swap" -gt "$SWAP_IO_WARN_PAGES" ]; then
    swap_streak=$(( swap_streak + 1 ))
  else
    swap_streak=0
  fi
  [ "$swap_streak" -lt "$SWAP_IO_WARN_TICKS" ] \
    || { echo "FAIL: $LABEL swap exceeded $SWAP_IO_WARN_PAGES pages for $swap_streak samples" >&2; exit 1; }
  if [ "$clean_sample" = 1 ]; then
    if [ $(( SECONDS - quiet_started )) -ge "$QUIET_SECONDS" ]; then
      echo "OK: $LABEL hosts quiet for ${QUIET_SECONDS}s; evidence=$OUTPUT"
      exit 0
    fi
  else
    quiet_started="$SECONDS"
  fi
  [ $(( SECONDS - started )) -lt "$QUIET_DEADLINE_SECONDS" ] \
    || { echo "FAIL: $LABEL did not reach a ${QUIET_SECONDS}s quiet interval before deadline" >&2; exit 1; }
  sleep "$QUIET_SAMPLE_SECONDS"
done
