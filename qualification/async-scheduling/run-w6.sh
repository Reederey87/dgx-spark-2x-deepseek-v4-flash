#!/usr/bin/env bash
# Run one deterministic W6 asynchronous workload from the Mac through SSH.
# Never switches profiles or promotes. Requires explicit test-window acknowledgement.
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT="$(cd "$BENCH_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$KIT/runtime/cluster.env"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
W6_START_MIN_KIB="${W6_START_MIN_KIB:-2359296}" # 2.25 GiB
case "$W6_START_MIN_KIB" in (*[!0-9]*|'') echo "FAIL: invalid W6_START_MIN_KIB" >&2; exit 1;; esac
[ "$W6_START_MIN_KIB" -ge 1048576 ] || { echo "FAIL: W6_START_MIN_KIB must be >=1 GiB" >&2; exit 1; }
W6_SAFETY_TAIL_SECONDS="${W6_SAFETY_TAIL_SECONDS:-60}"
case "$W6_SAFETY_TAIL_SECONDS" in (*[!0-9]*|'') echo "FAIL: invalid W6_SAFETY_TAIL_SECONDS" >&2; exit 1;; esac

TEST_ID="${1:-}"
PROFILE="${2:-}"
REP="${3:-1}"
[ -n "$TEST_ID" ] || { echo "usage: run-w6.sh <test-id> <decode-heavy|production-mix> [rep]" >&2; exit 1; }
case "$PROFILE" in decode-heavy|production-mix) ;; *) echo "FAIL: invalid W6 profile '$PROFILE'" >&2; exit 1;; esac
[ "${W6_TEST_WINDOW:-0}" = 1 ] || {
  echo "FAIL: set W6_TEST_WINDOW=1 only after stopping normal traffic and opening a rollback-ready test window" >&2
  exit 1
}

LOCK_DIR="$BENCH_DIR/.ab-run.lock"
mkdir "$LOCK_DIR" 2>/dev/null || { echo "FAIL: another benchmark owns $LOCK_DIR" >&2; exit 1; }
printf 'pid=%s test=%s workload=W6-%s rep=%s started=%s\n' "$$" "$TEST_ID" "$PROFILE" "$REP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner"
WORKDIR="$(mktemp -d)"
POWER_ACTIVE=0
POWER_ID=""
CLIENT_PID=""
TUNNEL_PID=""
NS="$(printf '%s' "$TEST_ID-W6-$PROFILE-rep$REP" | tr -c 'A-Za-z0-9_.-' '_')"
cleanup() {
  if [ -n "$CLIENT_PID" ]; then kill "$CLIENT_PID" 2>/dev/null || true; wait "$CLIENT_PID" 2>/dev/null || true; fi
  if [ -n "$TUNNEL_PID" ]; then kill "$TUNNEL_PID" 2>/dev/null || true; wait "$TUNNEL_PID" 2>/dev/null || true; fi
  if [ "$POWER_ACTIVE" = 1 ]; then
    uv run python "$BENCH_DIR/power-sample.py" cleanup --run-id "$POWER_ID" \
      --user "$CLUSTER_USER" --hosts "$HEAD_HOST" "$WORKER_HOST" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORKDIR"
  rm -f "$LOCK_DIR/owner"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

last_uint() { sed -nE 's/^[[:space:]]*([0-9]+)[[:space:]]*$/\1/p' | tail -1; }
service_state() {
  local host="$1" role="$2" state attempt
  for attempt in 1 2; do
    state="$(ssh -T "$CLUSTER_USER@$host" "systemctl --user is-active vllm-dsv4-$role 2>/dev/null || true" 2>/dev/null | tr -d '\r' | sed -nE '/^(active|inactive|failed|activating|deactivating)$/p' | head -1)"
    [ -z "$state" ] || { printf '%s\n' "$state"; return 0; }
    [ "$attempt" = 2 ] || sleep 1
  done
  return 1
}

for timer in vllm-dsv4-watchdog.timer vllm-metrics-watch.timer; do
  state="$(ssh -T "$CLUSTER_USER@$HEAD_HOST" "systemctl --user is-active $timer" 2>/dev/null | tr -d '\r' | sed -nE '/^(active|inactive|failed|activating|deactivating)$/p' | head -1 || true)"
  [ "$state" = inactive ] || { echo "FAIL: $timer is '$state'" >&2; exit 1; }
done
# An opening eval, boot, or prior arm can release swap writeback after its
# requests finish. Require a full quiet interval before attributing pressure to
# W6; a critical spike aborts here, before any workload request is sent.
QUIET_EVIDENCE="${W6_QUIET_EVIDENCE:-$BENCH_DIR/results/${NS}__prestart-quiet.jsonl}"
bash "$BENCH_DIR/wait-host-quiet.sh" "$QUIET_EVIDENCE" "$TEST_ID-W6-$PROFILE-prestart"
# The opening eval ends with a long-context request and MemAvailable can take a
# few samples to settle.  The 2.25 GiB default is below the measured healthy
# prod idle band (2.39--2.48 GiB) but leaves 1.25 GiB before the independent
# in-run abort.  Swap/fatal/service aborts below remain stricter live signals.
admission_deadline=$((SECONDS + 180))
while :; do
  admission_ready=1
  admission_detail=""
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    mem="$(ssh -T "$CLUSTER_USER@$host" "awk '/^MemAvailable:/{print \$2}' /proc/meminfo" | last_uint)"
    case "$mem" in
      (*[!0-9]*|'') echo "FAIL: $host MemAvailable unreadable" >&2; exit 1;;
    esac
    admission_detail="${admission_detail}${admission_detail:+, }$host=$((mem/1024))MiB"
    [ "$mem" -ge "$W6_START_MIN_KIB" ] || admission_ready=0
  done
  [ "$admission_ready" = 0 ] || break
  if [ "$SECONDS" -ge "$admission_deadline" ]; then
    echo "FAIL: W6 start floor not reached after settle wait ($admission_detail; need >=$((W6_START_MIN_KIB/1024))MiB each)" >&2
    exit 1
  fi
  echo "== waiting for post-eval memory to settle ($admission_detail; need >=$((W6_START_MIN_KIB/1024))MiB each)" >&2
  sleep 10
done
echo "== W6 memory admission passed ($admission_detail)" >&2

MANIFEST="${W6_MANIFEST:-$BENCH_DIR/manifests/w6-${PROFILE}.json}"
mkdir -p "$(dirname "$MANIFEST")"

TUNNEL_PORT="${W6_TUNNEL_PORT:-18081}"
case "$TUNNEL_PORT" in (*[!0-9]*|'') echo "FAIL: invalid W6_TUNNEL_PORT" >&2; exit 1;; esac
[ "$TUNNEL_PORT" -ge 1024 ] && [ "$TUNNEL_PORT" -le 65535 ] || { echo "FAIL: W6_TUNNEL_PORT must be 1024..65535" >&2; exit 1; }
if lsof -nP -iTCP:"$TUNNEL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then echo "FAIL: local W6 tunnel port $TUNNEL_PORT is in use" >&2; exit 1; fi
ssh -T -N -L "${TUNNEL_PORT}:127.0.0.1:${API_PORT}" -o BatchMode=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  "$CLUSTER_USER@$HEAD_HOST" &
TUNNEL_PID=$!
tunnel_ready=0
for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${TUNNEL_PORT}/health" >/dev/null; then tunnel_ready=1; break; fi
  kill -0 "$TUNNEL_PID" 2>/dev/null || break
  sleep 1
done
[ "$tunnel_ready" = 1 ] || { echo "FAIL: W6 SSH tunnel did not become healthy" >&2; exit 1; }

W6_RATE=""
if [ "$PROFILE" = production-mix ]; then
  W6_RATE="${W6_REQUEST_RATE:-0.15}"
  [[ "$W6_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "FAIL: W6_REQUEST_RATE must be numeric" >&2; exit 1; }
fi
W6_THINKING="${W6_THINKING:-0}"
case "$W6_THINKING" in 0|1) ;; *) echo "FAIL: W6_THINKING must be 0 or 1" >&2; exit 1;; esac

# Build optional argv with positional parameters instead of empty arrays: the
# Mac runner uses Bash 3.2, where "${empty[@]}" trips set -u.
w6_client() {
  local mode="$1" output="$2" run_id="$3"
  set -- uv run --with aiohttp python "$BENCH_DIR/w6-async.py" --profile "$PROFILE" \
    --manifest "$MANIFEST" --output "$output" --run-id "$run_id" \
    --model "$SERVED_MODEL_NAME" --base-url "http://127.0.0.1:${TUNNEL_PORT}"
  [ -z "$W6_RATE" ] || set -- "$@" --request-rate "$W6_RATE"
  if [ "$mode" = prepare ]; then
    set -- "$@" --prepare-only
  else
    [ "$W6_THINKING" = 0 ] || set -- "$@" --thinking
    set -- "$@" --skip-tokenize-validation
  fi
  "$@"
}

# Build or verify the exact-token manifest before power timing. The workload
# invocation rechecks hashes/identity locally but skips duplicate /tokenize
# calls, so energy includes request execution rather than calibration traffic.
w6_client prepare "$WORKDIR/prepare.json" "$NS-prepare" >/dev/null

POWER_ID="${NS}-$(date -u +%Y%m%dT%H%M%SZ)"
uv run python "$BENCH_DIR/power-sample.py" start --run-id "$POWER_ID" \
  --user "$CLUSTER_USER" --hosts "$HEAD_HOST" "$WORKER_HOST" >/dev/null
POWER_ACTIVE=1

prev_head="$(ssh -T "$CLUSTER_USER@$HEAD_HOST" "awk '/^(pswpin|pswpout) /{s+=\$2} END{print s+0}' /proc/vmstat" | last_uint)"
prev_worker="$(ssh -T "$CLUSTER_USER@$WORKER_HOST" "awk '/^(pswpin|pswpout) /{s+=\$2} END{print s+0}' /proc/vmstat" | last_uint)"
[ -n "$prev_head" ] && [ -n "$prev_worker" ] || { echo "FAIL: initial swap telemetry unreadable" >&2; exit 1; }

(
  child=""
  terminate() { [ -z "$child" ] || kill "$child" 2>/dev/null || true; exit 143; }
  trap terminate HUP INT TERM
  set +e
  w6_client run "$WORKDIR/result.json" "$NS" \
    >"$WORKDIR/client.out" 2>"$WORKDIR/client.err" &
  child=$!
  wait "$child"; rc=$?
  printf '%s\n' "$rc" > "$WORKDIR/client.rc"
  : > "$WORKDIR/client.done"
) &
CLIENT_PID=$!

ABORT_REASON=""; swap_streak=0; HOST_EVENTS="$WORKDIR/host-events.jsonl"; : > "$HOST_EVENTS"
sample_safety_once() {
  local host role mem swap_now state fatal previous swap_delta
  max_swap=0
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    role=$([ "$host" = "$HEAD_HOST" ] && echo head || echo worker)
    mem="$(ssh -T "$CLUSTER_USER@$host" "awk '/^MemAvailable:/{print \$2}' /proc/meminfo" | last_uint)"
    swap_now="$(ssh -T "$CLUSTER_USER@$host" "awk '/^(pswpin|pswpout) /{s+=\$2} END{print s+0}' /proc/vmstat" | last_uint)"
    state="$(service_state "$host" "$role" || true)"
    fatal="$(ssh -T "$CLUSTER_USER@$host" "docker logs --since 15s vllm-dsv4 2>&1 | grep -Eic 'Xid|illegal memory access|EngineCore failed|CUDA error|Traceback' || true" | last_uint)"; fatal="${fatal:-1}"
    case "$mem" in (*[!0-9]*|'') ABORT_REASON="${ABORT_REASON:-$host memory telemetry unreadable}"; mem=0;; esac
    case "$swap_now" in (*[!0-9]*|'') ABORT_REASON="${ABORT_REASON:-$host swap telemetry unreadable}"; swap_now="$([ "$host" = "$HEAD_HOST" ] && echo "$prev_head" || echo "$prev_worker")";; esac
    if [ "$host" = "$HEAD_HOST" ]; then previous="$prev_head"; prev_head="$swap_now"; else previous="$prev_worker"; prev_worker="$swap_now"; fi
    swap_delta=$((swap_now-previous)); [ "$swap_delta" -ge 0 ] || swap_delta=0
    [ "$swap_delta" -le "$max_swap" ] || max_swap="$swap_delta"
    uv run python - "$HOST_EVENTS" "$host" "$mem" "$swap_delta" "$state" "$fatal" <<'PY'
import json, sys, time
with open(sys.argv[1], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": sys.argv[2], "mem_available_kib": int(sys.argv[3]),
        "swap_delta_pages": int(sys.argv[4]), "service": sys.argv[5],
        "fatal_log_matches": int(sys.argv[6])}) + "\n")
PY
    [ "$mem" -ge 1048576 ] || ABORT_REASON="${ABORT_REASON:-$host MemAvailable fell below 1 GiB}"
    [ "$state" = active ] || ABORT_REASON="${ABORT_REASON:-$host service non-active or unreadable}"
    [ "$fatal" -eq 0 ] || ABORT_REASON="${ABORT_REASON:-$host emitted fatal GPU/engine log}"
    [ "$swap_delta" -le 4096 ] || ABORT_REASON="${ABORT_REASON:-$host swap delta exceeded 4096 pages}"
  done
  if [ "$max_swap" -gt 256 ]; then swap_streak=$((swap_streak+1)); else swap_streak=0; fi
  [ "$swap_streak" -lt 3 ] || ABORT_REASON="${ABORT_REASON:-swap delta exceeded 256 pages for three samples}"
}
while [ ! -e "$WORKDIR/client.done" ]; do
  sample_safety_once
  if [ -n "$ABORT_REASON" ]; then kill "$CLIENT_PID" 2>/dev/null || true; wait "$CLIENT_PID" 2>/dev/null || true; CLIENT_PID=""; break; fi
  sleep 5
done
if [ -n "$CLIENT_PID" ]; then wait "$CLIENT_PID" 2>/dev/null || true; CLIENT_PID=""; fi
RUN_RC=1
[ -s "$WORKDIR/client.rc" ] && RUN_RC="$(last_uint < "$WORKDIR/client.rc")"
if [ ! -s "$WORKDIR/result.json" ]; then
  uv run python - "$WORKDIR/result.json" "$ABORT_REASON" <<'PY'
import json, sys
json.dump({"passed": False, "verdict": "fail", "error": sys.argv[2] or "W6 client result missing"}, open(sys.argv[1], "w"))
PY
fi

COMPLETION_TOKENS="$(uv run python - "$WORKDIR/result.json" <<'PY'
import json, sys
try:
    value = int(json.load(open(sys.argv[1], encoding="utf-8")).get("total_output_tokens", 0))
except Exception:
    value = 0
print(max(0, value))
PY
)"
POWER_JSON="$(uv run python "$BENCH_DIR/power-sample.py" stop --run-id "$POWER_ID" \
  --user "$CLUSTER_USER" --hosts "$HEAD_HOST" "$WORKER_HOST" --completion-tokens "$COMPLETION_TOKENS")"
POWER_ACTIVE=0

# Swap writeback can lag request completion. Keep the energy interval exact,
# then hold a separate safety tail so deferred pressure is attributed to the
# arm that caused it instead of poisoning the next arm's first sample.
if [ -z "$ABORT_REASON" ] && [ "$W6_SAFETY_TAIL_SECONDS" -gt 0 ]; then
  safety_tail_deadline=$((SECONDS + W6_SAFETY_TAIL_SECONDS))
  while [ "$SECONDS" -lt "$safety_tail_deadline" ]; do
    sample_safety_once
    [ -z "$ABORT_REASON" ] || break
    sleep 5
  done
fi

FINGERPRINT="$(uv run python - "$CLUSTER_USER" "$HEAD_HOST" "$WORKER_HOST" "$KIT_DIR" <<'PY'
import hashlib, json, subprocess, sys
user, head, worker, kit = sys.argv[1:5]
nodes = {}
for host in (head, worker):
    raw = subprocess.check_output(["ssh", f"{user}@{host}", "docker inspect vllm-dsv4 --format '{{json .Config.Cmd}} {{json .Config.Env}} {{.Image}}'"], text=True)
    hashes = subprocess.check_output(["ssh", f"{user}@{host}", f"sha256sum {kit}/cluster.env {kit}/docker-compose.dspark.yml"], text=True).split()
    nodes[host] = {"docker_inspect_raw": raw.strip(), "cluster_env_sha256": hashes[0], "compose_sha256": hashes[2]}
print(json.dumps(nodes))
PY
)"
RESULTS="$BENCH_DIR/results"
mkdir -p "$RESULTS"
OUT="$RESULTS/${TEST_ID}__W6-${PROFILE}__rep${REP}__$(date -u +%Y%m%dT%H%M%SZ).json"
uv run python - "$WORKDIR/result.json" "$OUT" "$TEST_ID" "$PROFILE" "$REP" "$POWER_JSON" "$FINGERPRINT" "$ABORT_REASON" "$W6_SAFETY_TAIL_SECONDS" <<'PY'
import json, sys
src, out, test_id, profile, rep, power, fingerprint, abort_reason, tail_seconds = sys.argv[1:10]
data = json.load(open(src, encoding="utf-8"))
data["power"] = json.loads(power)
events = src.rsplit("/", 1)[0] + "/host-events.jsonl"
data["host_safety"] = {"abort_reason": abort_reason or None,
                       "post_request_tail_seconds": int(tail_seconds), "samples": []}
try:
    data["host_safety"]["samples"] = [json.loads(line) for line in open(events, encoding="utf-8") if line.strip()]
except FileNotFoundError:
    pass
doc = {"test_id": test_id, "workload": f"W6-{profile}", "rep": int(rep),
       "fingerprint": json.loads(fingerprint), "data": data,
       "harness_version": "w6-async-v2-c8-tail"}
json.dump(doc, open(out, "w", encoding="utf-8"), indent=2)
PY
echo "== result written: $OUT"
[ -z "$ABORT_REASON" ] && [ "$RUN_RC" -eq 0 ]
