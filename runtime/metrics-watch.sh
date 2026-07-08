#!/usr/bin/env bash
# Recipe #11 — lightweight loopback observability watcher for the vllm-dsv4 head.
# Runs as the cluster user on the head node on a ~45 s systemd user timer
# (vllm-metrics-watch.timer).
# READ-ONLY: scrapes the loopback :8000/metrics + inspects the running container;
# computes interval deltas vs a small state file; WARNs (journald + a rotating
# log) on threshold breach. Closes N-A Caveat A (the TTFT-blind silent revert)
# TWO ways: (1) a direct config-integrity assert on the LIVE .Config.Cmd — the
# check preflight.sh can't do because it runs pre-container — and (2) an interval
# TTFT-mean signal that spikes if the HoL fix ever regresses mid-run.
# Never fails hard: a watcher must survive transient scrape misses.
set -uo pipefail   # deliberately NOT -e

KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

STATE="$KIT/.metrics-watch.state"
LOG_DIR="$KIT/logs"; mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG="$LOG_DIR/metrics-watch.log"
MAXLOG=2000   # lines; simple truncate-rotate

# --- thresholds (tunable) ---
TTFT_WARN_MS="${TTFT_WARN_MS:-3000}"   # interval-mean TTFT; a HoL revert sends short-req TTFT to ~59 s
WAIT_WARN="${WAIT_WARN:-5}"            # sustained requests waiting
KVUTIL_WARN="${KVUTIL_WARN:-0.95}"     # KV pool pressure (OOM risk)
ACCEPT_WARN="${ACCEPT_WARN:-0.30}"     # spec-decode acceptance floor
MIN_DRAFT="${MIN_DRAFT:-50}"          # ignore acceptance when too few draft tokens this interval (noise)

# --- Telegram alerting (optional; strictly no-op unless notify.env supplies creds) ---
# notify.env (gitignored, 0600) defines TG_BOT_TOKEN + TG_CHAT_ID. Absent => silent,
# i.e. exactly the prior journald+log behavior. Alerts dedup on the SET of active WARN
# categories and re-fire a still-active condition only every NOTIFY_COOLDOWN seconds; a
# one-shot "recovered" note is sent when all WARNs clear. Notification never fails the run.
NOTIFY_STATE="$KIT/.metrics-watch.notify-state"
NOTIFY_COOLDOWN="${NOTIFY_COOLDOWN:-900}"   # 15 min between reminders for a persistent condition
case "$NOTIFY_COOLDOWN" in ''|*[!0-9]*) NOTIFY_COOLDOWN=900 ;; esac   # ignore non-numeric overrides
WARN_BUF=""                                  # accumulates this run's WARN lines (see emit); read by the EXIT trap
# shellcheck disable=SC1091
[ -f "$KIT/notify.env" ] && . "$KIT/notify.env" 2>/dev/null || true

notify_send() {  # $1=text; returns 0 ONLY if a message was actually delivered (so state
  # advances only on real delivery); no-op + return 1 without creds or on transport failure.
  [ -n "${TG_BOT_TOKEN:-}" ] && [ -n "${TG_CHAT_ID:-}" ] || return 1
  # TG_THREAD_ID is optional (forum-topic supergroups); omitted -> posts to the chat's General.
  curl -fsS --max-time 8 -X POST \
    "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TG_CHAT_ID}" \
    ${TG_THREAD_ID:+--data-urlencode "message_thread_id=${TG_THREAD_ID}"} \
    --data-urlencode "text=$1" \
    --data-urlencode "disable_web_page_preview=true" >/dev/null 2>&1
}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

emit() {  # route one analyzer line: WARN → journald(stderr)+log+alert buffer, else → log
  case "$1" in
    WARN*) printf '%s %s\n' "$(ts)" "$1" | tee -a "$LOG" >&2
           WARN_BUF="${WARN_BUF}${1}"$'\n' ;;
    *)     printf '%s %s\n' "$(ts)" "$1" >> "$LOG" ;;
  esac
}

notify_finalize() {  # EXIT trap: dedup + cooldown + recovery over the whole run's WARN set
  local warn_lines sig now_epoch prev_sig prev_ts do_notify host
  warn_lines="$(printf '%s' "$WARN_BUF" | sed '/^$/d')"
  sig="$(printf '%s\n' "$warn_lines" | sed -n 's/^WARN \([a-z-]*\):.*/\1/p' | sort -u | tr '\n' ',')"
  now_epoch="$(date +%s 2>/dev/null || echo 0)"
  prev_sig=""; prev_ts=0
  if [ -f "$NOTIFY_STATE" ]; then
    prev_sig="$(sed -n '1p' "$NOTIFY_STATE" 2>/dev/null || true)"
    prev_ts="$(sed -n '2p' "$NOTIFY_STATE" 2>/dev/null || true)"
  fi
  case "$prev_ts" in ''|*[!0-9]*) prev_ts=0 ;; esac
  host="$(hostname -s 2>/dev/null || echo node)"
  if [ -n "$sig" ]; then
    do_notify=0
    if [ "$sig" != "$prev_sig" ]; then do_notify=1
    elif [ "$(( now_epoch - prev_ts ))" -ge "$NOTIFY_COOLDOWN" ]; then do_notify=1; fi
    if [ "$do_notify" = 1 ]; then
      # advance state ONLY on a delivered alert — else retry next run (no missed alerts,
      # no premature cooldown when Telegram is briefly unreachable or creds are absent).
      if notify_send "$(printf '⚠️ vllm-dsv4 [%s] alert:\n%s' "$host" "$warn_lines")"; then
        printf '%s\n%s\n' "$sig" "$now_epoch" > "$NOTIFY_STATE" 2>/dev/null || true
      fi
    fi
  elif [ -n "$prev_sig" ]; then
    # clear the prior condition ONLY once the recovery note is actually delivered.
    if notify_send "$(printf '✅ vllm-dsv4 [%s] recovered — all clear' "$host")"; then
      : > "$NOTIFY_STATE" 2>/dev/null || true
    fi
  fi
}
trap notify_finalize EXIT

# --- 1. config-integrity: the HoL fix must be in the RUNNING container's cmd ---
cmd="$(docker inspect vllm-dsv4 --format '{{json .Config.Cmd}}' 2>/dev/null || true)"
if [ -n "$cmd" ]; then
  if ! printf '%s' "$cmd" | grep -q -- '--long-prefill-token-threshold [1-9]'; then
    emit "WARN hol-revert: running vllm-dsv4 is missing a positive --long-prefill-token-threshold (Caveat A) — short-request TTFT will regress under long prefills"
  fi
else
  emit "note: vllm-dsv4 not inspectable (not running?)"
fi

# --- 2. scrape /metrics ---
metrics="$(curl -fsS --max-time 5 "http://127.0.0.1:$API_PORT/metrics" 2>/dev/null || true)"
if [ -z "$metrics" ]; then
  emit "WARN health: 127.0.0.1:$API_PORT/metrics unreachable"
  exit 0
fi

# --- 3. parse + interval-delta + thresholds ---
# NOTE: metrics go through a TEMP FILE (argv), not stdin — `python3 - <<'PY'`
# already consumes stdin as the program source, so a stdin pipe would be ignored.
mtmp="$(mktemp)"
printf '%s' "$metrics" > "$mtmp"
out="$(python3 - "$STATE" "$TTFT_WARN_MS" "$WAIT_WARN" "$KVUTIL_WARN" "$ACCEPT_WARN" "$MIN_DRAFT" "$mtmp" <<'PY'
import json, re, sys

state_path = sys.argv[1]
TTFT_WARN_MS = float(sys.argv[2]); WAIT_WARN = float(sys.argv[3])
KVUTIL_WARN = float(sys.argv[4]); ACCEPT_WARN = float(sys.argv[5]); MIN_DRAFT = float(sys.argv[6])
metrics_path = sys.argv[7]

LINE = re.compile(r'^(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$')
fam = {}
for line in open(metrics_path, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = LINE.match(line)
    if not m:
        continue
    name, val = m.group(1), m.group(3)
    try:
        fam.setdefault(name, 0.0)
        fam[name] += float(val)   # sum all label series for scalar families
    except ValueError:
        pass

def g(name):
    return fam.get(name, 0.0)

cur = {
    "ttft_sum": g("vllm:time_to_first_token_seconds_sum"),
    "ttft_count": g("vllm:time_to_first_token_seconds_count"),
    "e2e_sum": g("vllm:e2e_request_latency_seconds_sum"),
    "e2e_count": g("vllm:e2e_request_latency_seconds_count"),
    "accepted": g("vllm:spec_decode_num_accepted_tokens_total"),
    "draft": g("vllm:spec_decode_num_draft_tokens_total"),
    "preempt": g("vllm:num_preemptions_total"),
}
waiting = g("vllm:num_requests_waiting")
running = g("vllm:num_requests_running")
kv_util = g("vllm:kv_cache_usage_perc")

try:
    prev = json.load(open(state_path))
except Exception:
    prev = None

warns, ttft_iv, e2e_iv, accept_iv, preempt_d = [], None, None, None, None
if prev:
    d_ttft_c = cur["ttft_count"] - prev.get("ttft_count", 0)
    d_ttft_s = cur["ttft_sum"] - prev.get("ttft_sum", 0)
    d_e2e_c = cur["e2e_count"] - prev.get("e2e_count", 0)
    d_e2e_s = cur["e2e_sum"] - prev.get("e2e_sum", 0)
    d_acc = cur["accepted"] - prev.get("accepted", 0)
    d_draft = cur["draft"] - prev.get("draft", 0)
    preempt_d = cur["preempt"] - prev.get("preempt", 0)
    if d_ttft_c > 0:
        ttft_iv = d_ttft_s / d_ttft_c * 1000.0
    if d_e2e_c > 0:
        e2e_iv = d_e2e_s / d_e2e_c * 1000.0
    if d_draft >= MIN_DRAFT:
        accept_iv = d_acc / d_draft if d_draft > 0 else None

# thresholds (only alert on signals we actually have this interval)
if ttft_iv is not None and ttft_iv > TTFT_WARN_MS:
    warns.append(f"WARN ttft: interval-mean TTFT {ttft_iv:.0f} ms > {TTFT_WARN_MS:.0f} ms (HoL regression? check --long-prefill-token-threshold)")
if waiting > WAIT_WARN:
    warns.append(f"WARN queue: {waiting:.0f} requests waiting > {WAIT_WARN:.0f} (saturation / head-of-line)")
if kv_util > KVUTIL_WARN:
    warns.append(f"WARN kv: kv_cache_usage_perc {kv_util:.3f} > {KVUTIL_WARN:.2f} (OOM risk)")
if accept_iv is not None and accept_iv < ACCEPT_WARN:
    warns.append(f"WARN spec-decode: interval acceptance {accept_iv:.3f} < {ACCEPT_WARN:.2f} (MTP health)")
if preempt_d is not None and preempt_d > 0:
    warns.append(f"WARN preempt: {preempt_d:.0f} preemptions this interval (concurrency thrash at 1M ctx)")

def f(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"
summary = (f"SUMMARY ttft_iv_ms={f(ttft_iv,0)} e2e_iv_ms={f(e2e_iv,0)} accept_iv={f(accept_iv,3)} "
           f"waiting={waiting:.0f} running={running:.0f} kv_util={kv_util:.3f} preempt_d={f(preempt_d,0)}")

for w in warns:
    print(w)
print(summary)

try:
    json.dump(cur, open(state_path, "w"))
except Exception:
    pass
PY
)"
rm -f "$mtmp"

# route analyzer output (WARN → stderr+log, SUMMARY → log)
if [ -n "$out" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && emit "$line"
  done <<< "$out"
fi

# --- simple log rotation ---
if [ -f "$LOG" ]; then
  lines="$(wc -l < "$LOG" 2>/dev/null || echo 0)"
  if [ "$lines" -gt "$MAXLOG" ]; then
    tail -n "$(( MAXLOG / 2 ))" "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
  fi
fi
exit 0
