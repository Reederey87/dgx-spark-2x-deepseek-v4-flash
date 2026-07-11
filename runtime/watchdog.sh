#!/usr/bin/env bash
# Inference-level watchdog for the head node (drill-proven need: the API
# process survives a lost TP peer, so /health lies while inference hangs).
# Logic: if /health answers (engine fully initialized) but a real 1-token
# completion times out, restart the head unit. During startup /health is
# down, so this never fires mid-load.
#
# Canary design (2026-07-10):
# - KEEP the live 1-token probe — it is the only signal that inference works.
# - Tag it with OpenAI "user"=vllm-dsv4-watchdog for log forensics (vLLM TTFT
#   histograms on this image are NOT labeled by user, so exclusion is via the
#   side-channel state file below).
# - Record every probe outcome to $KIT/.watchdog-probe.state so metrics-watch can
#   treat canary-only intervals as standby (ttft_user=n/a) instead of user traffic.
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

PROBE_STATE="$KIT/.watchdog-probe.state"
PROBE_USER="vllm-dsv4-watchdog"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-90}"
SATN_STATE="$KIT/.watchdog-satn.state"

record_probe() {  # $1=ok|fail  $2=e2e_ms (int, or 0)
  local ok="$1" e2e_ms="${2:-0}"
  python3 - "$PROBE_STATE" "$ok" "$e2e_ms" <<'PY'
import json, sys, time
path, ok, e2e = sys.argv[1], sys.argv[2], int(float(sys.argv[3]))
try:
    st = json.load(open(path))
except Exception:
    st = {"count": 0, "ok": 0, "fail": 0}
st["count"] = int(st.get("count", 0)) + 1
if ok == "ok":
    st["ok"] = int(st.get("ok", 0)) + 1
else:
    st["fail"] = int(st.get("fail", 0)) + 1
st["last_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
st["last_ok"] = ok == "ok"
st["last_e2e_ms"] = e2e
tmp = path + ".tmp"
json.dump(st, open(tmp, "w"))
import os
os.replace(tmp, path)
PY
}

# Consecutive-KV-saturation counter (2026-07-11 saturation/wedge discrimination).
# $1=reset|bump  $2=kv (for the state record); prints the resulting count.
satn_update() {
  local mode="$1" kv="${2:-0}"
  python3 - "$SATN_STATE" "$mode" "$kv" <<'PY'
import json, os, sys, time
path, mode, kv = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    st = json.load(open(path))
except Exception:
    st = {"consecutive": 0}
if mode == "bump":
    st["consecutive"] = int(st.get("consecutive", 0)) + 1
else:
    st["consecutive"] = 0
st["last_kv"] = float(kv)
st["last_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
tmp = path + ".tmp"
json.dump(st, open(tmp, "w"))
os.replace(tmp, path)
print(st["consecutive"])
PY
}

# One /metrics scrape reduced to "<kv> <gen>"; nonzero if the scrape fails or
# the KV gauge is missing. kv = max across label sets (single engine expected);
# gen = sum of all generation_tokens_total series (multi-request coverage).
metrics_snapshot() {
  local raw
  raw="$(curl -fsS --max-time 10 "http://127.0.0.1:$API_PORT/metrics" 2>/dev/null)" || return 1
  printf '%s\n' "$raw" | awk '
    /^#/ { next }
    /^vllm:kv_cache_usage_perc(\{|[ \t])/ {
      v = $NF + 0
      if (!kv_seen || v > kv) kv = v
      kv_seen = 1
    }
    /^vllm:generation_tokens_total(\{|[ \t])/ { gen += $NF }
    END {
      if (!kv_seen) exit 1
      printf "%s %s\n", kv, gen + 0
    }
  '
}

curl -fsS --max-time 5 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 || exit 0  # not up yet — not our problem

# Tagged 1-token canary. "user" is forensics + future label support; metrics-watch
# correlates via .watchdog-probe.state because this image's TTFT series has no user label.
body="$(python3 -c 'import json,sys; print(json.dumps({
  "model": sys.argv[1],
  "messages": [{"role": "user", "content": "hi"}],
  "max_tokens": 1,
  "temperature": 0,
  "user": sys.argv[2],
}))' "$SERVED_MODEL_NAME" "$PROBE_USER")"

t0="$(python3 -c 'import time; print(time.time())')"
if curl -fsS --max-time "$PROBE_TIMEOUT" -H 'Content-Type: application/json' \
     -d "$body" \
     "http://127.0.0.1:$API_PORT/v1/chat/completions" >/dev/null 2>&1; then
  t1="$(python3 -c 'import time; print(time.time())')"
  e2e_ms="$(python3 -c 'import sys; print(int((float(sys.argv[2])-float(sys.argv[1]))*1000))' "$t0" "$t1")"
  record_probe ok "$e2e_ms"
  satn_update reset "0" >/dev/null
  exit 0
fi

# Timed out or transport error — triage before bouncing: a saturated-but-still-
# generating engine isn't wedged and doesn't need the pair bounced, which would
# only kill in-flight long-context requests for nothing (2026-07-10/11: this
# happened twice on pure KV pressure).
# NOTE: the fail is recorded AFTER triage (just before the bounce/exit decision),
# not here — a retry-success cycle must record exactly ONE probe outcome, because
# metrics-watch-analyze.py's canary-interval logic assumes ≤1 record per cycle
# (gpt-5.6 review 2026-07-11: a fail+ok pair in one 45 s bucket could mask a real
# TTFT_CONTEND signal during the exact saturation window that matters).

bounce=1
snap_a="$(metrics_snapshot)"
if [ -z "$snap_a" ]; then
  echo "watchdog: metrics scrape failed during probe-timeout triage — treating as wedge" >&2
else
  kv_a="$(printf '%s' "$snap_a" | awk '{print $1}')"
  gen_a="$(printf '%s' "$snap_a" | awk '{print $2}')"
  if awk -v kv="$kv_a" 'BEGIN{exit !(kv < 0.95)}'; then
    satn_update reset "$kv_a" >/dev/null
  else
    # Adversarial-review fix (2026-07-11): instead of a blind sleep, retry the
    # 1-token canary with a 30 s cap — ADMISSION recovery is the true health
    # signal; an aggregate token counter can't prove the canary path works
    # (draining requests keep it climbing even when admission is stuck).
    t0r="$(python3 -c 'import time; print(time.time())')"
    if curl -fsS --max-time 30 -H 'Content-Type: application/json' \
         -d "$body" \
         "http://127.0.0.1:$API_PORT/v1/chat/completions" >/dev/null 2>&1; then
      t1r="$(python3 -c 'import time; print(time.time())')"
      e2er="$(python3 -c 'import sys; print(int((float(sys.argv[2])-float(sys.argv[1]))*1000))' "$t0r" "$t1r")"
      record_probe ok "$e2er"
      satn_update reset "$kv_a" >/dev/null
      echo "watchdog: canary retry succeeded during saturation triage (kv=${kv_a}) — admission recovered, NOT bouncing" >&2
      exit 0
    fi
    # Retry failed — preserve the full ~30 s window for the counter comparison
    # even if the retry returned early (fast transport error).
    rem="$(python3 -c 'import sys,time; print(max(0, int(30 - (time.time()-float(sys.argv[1])))))' "$t0r")"
    [ "${rem:-0}" -gt 0 ] 2>/dev/null && sleep "$rem"
    snap_b="$(metrics_snapshot)"
    if [ -z "$snap_b" ]; then
      echo "watchdog: metrics scrape failed on saturation follow-up — treating as wedge" >&2
      satn_update reset "$kv_a" >/dev/null
    else
      kv_b="$(printf '%s' "$snap_b" | awk '{print $1}')"
      gen_b="$(printf '%s' "$snap_b" | awk '{print $2}')"
      if awk -v a="$gen_a" -v b="$gen_b" 'BEGIN{exit !(b > a)}'; then
        cons="$(satn_update bump "$kv_b")"
        delta="$(awk -v a="$gen_a" -v b="$gen_b" 'BEGIN{printf "%.0f", b - a}')"
        echo "watchdog: KV saturation (kv=${kv_b}, +${delta} gen tokens in 30s) — probe timeout attributed to capacity, NOT bouncing" >&2
        if [ "$cons" -ge 3 ]; then
          echo "watchdog: KV saturation persisted >=3 checks — escalating to pair bounce (livelock backstop)" >&2
          satn_update reset "$kv_b" >/dev/null
        else
          bounce=0
        fi
      else
        satn_update reset "$kv_b" >/dev/null
      fi
    fi
  fi
fi

# Triage concluded without admission recovery — record the single fail outcome
# for this cycle (covers both the saturation-no-bounce and the bounce paths).
record_probe fail "$(( PROBE_TIMEOUT * 1000 ))"

[ "$bounce" = 0 ] && exit 0

# Bounce the PAIR in strict order (drill-proven twice):
#   1. STOP the head first — its stale rendezvous store on :25000 must be gone
#      before the worker restarts, or the fresh worker joins the dead group
#      and zombifies (never exits, Restart= can't help).
#   2. Restart the worker — headless vLLM waits for a master to appear.
#   3. Start the head — preflight re-checks the worker, then re-rendezvous.
# reset-failed throughout: flap windows exhaust StartLimitBurst, and this
# watchdog's 5-min period is the real rate limiter.
echo "watchdog: /health OK but inference timed out — bouncing pair (head stop, worker restart, head start)" >&2
systemctl --user stop vllm-dsv4-head.service
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
  docker inspect vllm-dsv4 >/dev/null 2>&1 || break
  sleep 5
done
docker rm -f vllm-dsv4 >/dev/null 2>&1 || true
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  "$CLUSTER_USER@$WORKER_R1" 'systemctl --user reset-failed vllm-dsv4-worker.service 2>/dev/null; systemctl --user restart vllm-dsv4-worker.service' \
  || echo "watchdog: worker restart over QSFP failed — starting head anyway (preflight will revive it)" >&2
systemctl --user reset-failed vllm-dsv4-head.service 2>/dev/null || true
systemctl --user start vllm-dsv4-head.service
