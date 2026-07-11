#!/usr/bin/env bash
# Recipe #11 — lightweight loopback observability watcher for the vllm-dsv4 head.
# Runs on the head node on a ~45 s systemd user timer (vllm-metrics-watch.timer).
# READ-ONLY: scrapes the loopback :8000/metrics + inspects the running container;
# computes interval deltas vs a small state file; WARNs (journald + a rotating
# log) on threshold breach. Closes N-A Caveat A (the TTFT-blind silent revert)
# TWO ways: (1) a direct config-integrity assert on the LIVE .Config.Cmd — the
# check preflight.sh can't do because it runs pre-container — and (2) an interval
# TTFT-mean signal that spikes if the HoL fix ever regresses mid-run.
# Never fails hard: a watcher must survive transient scrape misses.
#
# Alert design notes (2026-07-10 false-positive fix):
# - Idle short-req TTFT is ~110 ms; under one long prefill with the HoL fix ~5–12 s.
# - A long prefill's OWN TTFT is multi-second by definition (compute-bound prefill)
#   and is NOT HoL regression. Red Hat / vLLM triage: waiting=0 + high TTFT =
#   prefill compute; waiting>0 + high TTFT = queueing / starvation.
# - True silent HoL revert (threshold missing) sends short-req TTFT to ~59 s AND
#   elevates the waiting queue. Alert only on that class, not on long-prefill wall.
# - Metrics are unreachable for ~6 min on every boot (night.sh power-cycle); suppress
#   health WARNs during container warm-up and require a failure streak.
# - Watchdog fires a 1-token canary every ~5 min (required — /health can lie). That is
#   NOT user traffic. metrics-watch correlates .watchdog-probe.state (written by
#   watchdog.sh) so canary-only intervals report ttft_user=n/a and never alert.
set -uo pipefail   # deliberately NOT -e

KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

STATE="$KIT/.metrics-watch.state"
STREAK_STATE="$KIT/.metrics-watch.streaks"
PROBE_STATE="$KIT/.watchdog-probe.state"
LOG_DIR="$KIT/logs"; mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG="$LOG_DIR/metrics-watch.log"
MAXLOG=2000   # lines; simple truncate-rotate

# --- thresholds (tunable via env) ---
# TTFT: only alert when interval-mean is high AND there is queue pressure (or the
# HoL knob is gone). Long-prefill compute alone with waiting=0 is expected.
TTFT_WARN_MS="${TTFT_WARN_MS:-8000}"     # contention TTFT floor (HoL-fixed under-load ~6 s)
TTFT_HOL_MS="${TTFT_HOL_MS:-30000}"      # ≥ this + waiting ⇒ likely true HoL / severe starvation
WAIT_WARN="${WAIT_WARN:-5}"             # sustained requests waiting
KVUTIL_WARN="${KVUTIL_WARN:-0.95}"      # KV pool pressure (OOM risk)
ACCEPT_WARN="${ACCEPT_WARN:-0.30}"      # spec-decode acceptance floor
MIN_DRAFT="${MIN_DRAFT:-50}"            # ignore acceptance when too few draft tokens this interval
# Health: boot-to-serving is ~6 min; require N consecutive scrape misses before WARN.
HEALTH_FAIL_STREAK="${HEALTH_FAIL_STREAK:-4}"   # 4 × ~45 s ≈ 3 min of true down
# Suppress health WARNs while the container is younger than this (boot/warmup).
BOOT_GRACE_SECS="${BOOT_GRACE_SECS:-600}"       # 10 min covers boot-to-serving + warmup.sh
# Consecutive TTFT-contention intervals before Telegram-grade WARN (log still notes earlier).
TTFT_STREAK="${TTFT_STREAK:-2}"

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

# streak helpers: key=value lines in STREAK_STATE
streak_get() {
  local k="$1"
  [ -f "$STREAK_STATE" ] || { echo 0; return; }
  local v
  v="$(sed -n "s/^${k}=//p" "$STREAK_STATE" 2>/dev/null | head -1)"
  case "$v" in ''|*[!0-9]*) echo 0 ;; *) echo "$v" ;; esac
}
streak_set() {
  local k="$1" v="$2" tmp
  tmp="$(mktemp)"
  if [ -f "$STREAK_STATE" ]; then
    grep -v "^${k}=" "$STREAK_STATE" > "$tmp" 2>/dev/null || true
  fi
  printf '%s=%s\n' "$k" "$v" >> "$tmp"
  mv "$tmp" "$STREAK_STATE" 2>/dev/null || true
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

# --- helpers: container age (seconds), 0 if unknown / not running ---
container_age_secs() {
  local started now
  started="$(docker inspect vllm-dsv4 --format '{{.State.StartedAt}}' 2>/dev/null || true)"
  [ -n "$started" ] || { echo 0; return; }
  # StartedAt is RFC3339; use date -d (GNU) on Ubuntu DGX OS.
  now="$(date +%s 2>/dev/null || echo 0)"
  started_epoch="$(date -d "$started" +%s 2>/dev/null || echo 0)"
  if [ "$started_epoch" -gt 0 ] && [ "$now" -ge "$started_epoch" ]; then
    echo $(( now - started_epoch ))
  else
    echo 0
  fi
}

# --- 1. config-integrity: the HoL fix must be in the RUNNING container's cmd ---
hol_ok=1
cmd="$(docker inspect vllm-dsv4 --format '{{json .Config.Cmd}}' 2>/dev/null || true)"
if [ -n "$cmd" ]; then
  if ! printf '%s' "$cmd" | grep -q -- '--long-prefill-token-threshold [1-9]'; then
    hol_ok=0
    emit "WARN hol-revert: running vllm-dsv4 is missing a positive --long-prefill-token-threshold (Caveat A) — short-request TTFT will regress under long prefills"
  fi
else
  emit "note: vllm-dsv4 not inspectable (not running?)"
fi

# --- 2. scrape /metrics (with boot-grace + streak before Telegram WARN) ---
metrics="$(curl -fsS --max-time 5 "http://127.0.0.1:$API_PORT/metrics" 2>/dev/null || true)"
if [ -z "$metrics" ]; then
  age="$(container_age_secs)"
  fails="$(streak_get health_fails)"
  fails=$(( fails + 1 ))
  streak_set health_fails "$fails"
  # Reset TTFT streak on outage so a post-boot first request doesn't inherit it.
  streak_set ttft_cont 0

  if [ "$age" -gt 0 ] && [ "$age" -lt "$BOOT_GRACE_SECS" ]; then
    emit "note: health: metrics unreachable during boot/warmup (container age ${age}s < ${BOOT_GRACE_SECS}s grace) — not alerting"
  elif [ "$fails" -lt "$HEALTH_FAIL_STREAK" ]; then
    emit "note: health: metrics unreachable (streak ${fails}/${HEALTH_FAIL_STREAK}) — not alerting yet"
  else
    emit "WARN health: 127.0.0.1:$API_PORT/metrics unreachable for ${fails} consecutive scrapes (~$(( fails * 45 ))s)"
  fi
  exit 0
fi
streak_set health_fails 0

# --- 2b. capacity: requests waiting specifically on KV capacity (reason label) ---
capacity_waiting="$(printf '%s\n' "$metrics" | awk '
  /^#/ { next }
  /^vllm:num_requests_waiting_by_reason(\{|[ \t])/ && /reason="capacity"/ { sum += $NF }
  END { printf "%.0f", sum + 0 }
')"
if [ "${capacity_waiting:-0}" -gt 0 ] 2>/dev/null; then
  emit "WARN capacity: ${capacity_waiting} request(s) waiting on KV capacity"
fi

# --- 3. parse + interval-delta + thresholds ---
# Analyzer lives in metrics-watch-analyze.py (keeps bash portable; avoids a
# giant heredoc inside $() that macOS bash 3.2 mis-parses). Metrics go via a
# TEMP FILE argv so the analyzer never has to share stdin with a heredoc.
mtmp="$(mktemp)"
printf '%s' "$metrics" > "$mtmp"
out="$(python3 "$KIT/metrics-watch-analyze.py" \
  "$STATE" "$PROBE_STATE" \
  "$TTFT_WARN_MS" "$TTFT_HOL_MS" "$WAIT_WARN" "$KVUTIL_WARN" "$ACCEPT_WARN" "$MIN_DRAFT" \
  "$hol_ok" "$mtmp" || true)"
rm -f "$mtmp"

# route analyzer output + apply TTFT contention streak
ttft_contend_line=""
if [ -n "$out" ]; then
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      TTFT_CONTEND*)
        ttft_contend_line="$line"
        ;;
      *)
        emit "$line"
        ;;
    esac
  done <<< "$out"
fi

if [ -n "$ttft_contend_line" ]; then
  # TTFT_CONTEND hol|elev ttft_iv_ms=N waiting=N
  kind="$(printf '%s' "$ttft_contend_line" | awk '{print $2}')"
  detail="$(printf '%s' "$ttft_contend_line" | sed 's/^TTFT_CONTEND [^ ]* //')"
  cont="$(streak_get ttft_cont)"
  cont=$(( cont + 1 ))
  streak_set ttft_cont "$cont"
  if [ "$cont" -lt "$TTFT_STREAK" ]; then
    emit "note: ttft contention (${kind}) ${detail} (streak ${cont}/${TTFT_STREAK}) — not alerting yet"
  else
    if [ "$kind" = "hol" ]; then
      emit "WARN ttft: interval-mean TTFT high under queue pressure (${detail}) — likely HoL / severe starvation (check --long-prefill-token-threshold)"
    else
      emit "WARN ttft: interval-mean TTFT elevated under queue pressure (${detail}) — short requests waiting behind long prefill"
    fi
  fi
else
  streak_set ttft_cont 0
fi

# --- simple log rotation ---
if [ -f "$LOG" ]; then
  lines="$(wc -l < "$LOG" 2>/dev/null || echo 0)"
  if [ "$lines" -gt "$MAXLOG" ]; then
    tail -n "$(( MAXLOG / 2 ))" "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
  fi
fi
exit 0
