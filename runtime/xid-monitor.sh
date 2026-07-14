#!/usr/bin/env bash
# Continuous NVIDIA Xid monitor for both DGX Spark nodes. Hardware-fault alerts
# are deliberately separate from watchdog.sh: this script never restarts vLLM.
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

KIT="$(cd "$(dirname "$0")" && pwd)"
XID_TEST_MODE=0
if [ "${1:-}" = "--test" ]; then
  shift
  XID_TEST_MODE=1
else
  # shellcheck disable=SC1091
  source "$KIT/cluster.env"
  # shellcheck disable=SC1091
  [ -f "$KIT/notify.env" ] && . "$KIT/notify.env" 2>/dev/null || true
fi

INCIDENT_DIR="${XID_INCIDENT_DIR:-$KIT/logs/xid-incidents}"
NOTIFY_COOLDOWN_SEC="${XID_NOTIFY_COOLDOWN_SEC:-300}"
CATASTROPHIC_XIDS=" 48 79 94 95 119 140 154 "

case "$NOTIFY_COOLDOWN_SEC" in
  ''|*[!0-9]*)
    echo "xid-monitor: XID_NOTIFY_COOLDOWN_SEC must be a non-negative integer (got '$NOTIFY_COOLDOWN_SEC')" >&2
    exit 1
    ;;
esac

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

notify_send() {
  local message="$1"
  [ -n "${TG_BOT_TOKEN:-}" ] && [ -n "${TG_CHAT_ID:-}" ] || return 1
  curl -fsS --max-time 8 -X POST \
    "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TG_CHAT_ID}" \
    ${TG_THREAD_ID:+--data-urlencode "message_thread_id=${TG_THREAD_ID}"} \
    --data-urlencode "text=$message" \
    --data-urlencode "disable_web_page_preview=true" >/dev/null 2>&1
}

handle_xid() {
  local code="$1" line="$2" host stamp cur_log prev_log now last last_file
  host="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-host)"

  if [[ "$CATASTROPHIC_XIDS" != *" $code "* ]]; then
    echo "$(ts) xid-monitor: Xid $code seen (log-only) — $line" >&2
    return 0
  fi

  echo "$(ts) xid-monitor: CATASTROPHIC Xid $code on $host — $line" >&2
  echo "$(ts) xid-monitor: NOT attempting a service bounce — inspect hardware and power-cycle if required" >&2

  # --test is classification-only: no filesystem writes and no outbound alerts.
  if [ "$XID_TEST_MODE" = "1" ]; then
    echo "$(ts) xid-monitor: test mode — capture and notification suppressed" >&2
    return 0
  fi

  mkdir -p "$INCIDENT_DIR" 2>/dev/null \
    || { echo "$(ts) xid-monitor: cannot create incident directory $INCIDENT_DIR" >&2; return 1; }
  now="$(date +%s)"
  last_file="$INCIDENT_DIR/.last-xid-$code"
  last=0
  if [ -f "$last_file" ]; then
    last="$(cat "$last_file" 2>/dev/null || echo 0)"
    case "$last" in ''|*[!0-9]*) last=0 ;; esac
  fi
  if [ $((now - last)) -lt "$NOTIFY_COOLDOWN_SEC" ]; then
    echo "$(ts) xid-monitor: Xid $code repeated within ${NOTIFY_COOLDOWN_SEC}s — capture/notify suppressed" >&2
    return 0
  fi
  printf '%s\n' "$now" >"$last_file" 2>/dev/null \
    || echo "$(ts) xid-monitor: cooldown state write failed; continuing with evidence capture" >&2
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  cur_log="$INCIDENT_DIR/${stamp}_xid${code}_current-boot.log"
  prev_log="$INCIDENT_DIR/${stamp}_xid${code}_previous-boot.log"
  journalctl -k -b >"$cur_log" 2>/dev/null \
    || echo "$(ts) xid-monitor: current-boot kernel-log capture failed" >&2
  journalctl -k -b -1 >"$prev_log" 2>/dev/null || true
  echo "$(ts) xid-monitor: diagnostic logs saved under $INCIDENT_DIR" >&2

  notify_send "XID-FAULT (hardware) on ${host}: Xid ${code} detected. No automatic vLLM restart was attempted. Inspect the node; a physical power-cycle may be required. Logs: ${cur_log}." \
    || echo "$(ts) xid-monitor: alert not sent (notify.env absent or transport failed)" >&2
}

process_line() {
  local line="$1" code after_xid stripped
  case "$line" in
    *Xid*)
      # Typical form: "Xid (PCI:0000:0f:00): 119, ...". Strip the PCI
      # parenthetical before taking the first digit run, or 0000 wins.
      after_xid="${line#*Xid}"
      stripped="$(printf '%s' "$after_xid" | sed -E 's/^[^0-9(]*\([^)]*\)//')"
      code="$(printf '%s\n' "$stripped" | grep -oE '[0-9]+' | head -1)"
      if [ -n "$code" ]; then
        handle_xid "$code" "$line"
      else
        echo "$(ts) xid-monitor: matched Xid but could not parse a code — $line" >&2
      fi
      ;;
  esac
}

if [ "$XID_TEST_MODE" = "1" ]; then
  process_line "${1:-NVRM: Xid (PCI:0000:0f:00): 119, synthetic test injection}"
  exit 0
fi

echo "$(ts) xid-monitor: following kernel journal" >&2
stdbuf -oL journalctl -k -f -o cat 2>/dev/null | while IFS= read -r line; do
  process_line "$line"
done
echo "$(ts) xid-monitor: kernel journal follower ended; systemd will restart it" >&2
