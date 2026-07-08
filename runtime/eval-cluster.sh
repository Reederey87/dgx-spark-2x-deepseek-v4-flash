#!/usr/bin/env bash
# Evaluate the running cluster through the head node loopback API.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

garble_check() {
  local response="$1"
  python3 - "$response" <<'PY'
import json, re, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    content = data["choices"][0]["message"].get("content") or ""
except Exception:
    raise SystemExit(0)
if not content:
    raise SystemExit(0)
if "\ufffd" in content:
    raise SystemExit(1)
# 80+ identical chars in a row = runaway garble; above any legit ASCII banner/divider in code output.
if re.search(r"(.)\1{79,}", content, re.S):
    raise SystemExit(1)
words = re.findall(r"[A-Za-z0-9_]+", content.lower())
if len(words) > 20 and len(set(words)) / len(words) < 0.15:
    raise SystemExit(1)
# Consecutive phrase-loop (DSpark's classic garble signature): a 4-word window
# repeated 4+ times back-to-back is never legitimate output.
if re.search(r"\b(\w+ \w+ \w+ \w+ )\1{3,}", content.lower()):
    raise SystemExit(1)
PY
}

print_prefix_cache_hit_rate() {
  local metrics hit
  metrics="$(ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/metrics 2>/dev/null" 2>/dev/null || true)"
  # V1 vLLM exposes hits/queries counters, not a hit_rate gauge → compute the ratio.
  # Skip Prometheus '# HELP'/'# TYPE' comment lines; fall back to a hit_rate gauge if present.
  hit="$(printf '%s\n' "$metrics" | awk '
    /^#/ {next}
    /(^|[[:space:]])(vllm:)?gpu_prefix_cache_hits_total/    {h=$NF}
    /(^|[[:space:]])(vllm:)?gpu_prefix_cache_queries_total/ {q=$NF}
    /(^|[[:space:]])(vllm:)?gpu_prefix_cache_hit_rate/      {g=$NF}
    END {
      if (q+0 > 0) { printf "%.4f", h/q }
      else if (g != "") { printf "%s", g }
      else { exit 1 }
    }' || true)"
  if [ -n "$hit" ]; then
    echo "prefix_cache_hit_rate: $hit"
  else
    echo "prefix_cache_hit_rate: n/a"
  fi
}

# --- fractional wall clock (BSD date has no %N; python is the only portable sub-second source) ---
now() { python3 -c 'import time;print(time.time())'; }

# --- read-only Prometheus scrape to a local file (skip-on-failure, never aborts the eval) ---
scrape_metrics() {   # $1 = destination local path
  ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/metrics 2>/dev/null" \
    > "$1" 2>/dev/null || true
}

ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/health" >/dev/null \
  || fail "API health check failed" "start the cluster before evaluation"
echo "ok: health"

WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

# ---- server-side streaming TTFT reader (runs on head; loopback -> pure server first-token time) ----
cat > "$WORKDIR/sse_ttft.py" <<'PY'
import json, sys, time, urllib.request
url, model, prompt, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = json.dumps({"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": max_tokens,
                   "stream": True}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
t0 = time.time(); ttft = None
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta", {}) or {}
            piece = delta.get("reasoning") or delta.get("content") or ""
            if piece:
                ttft = time.time() - t0
                break
except Exception:
    pass
print(f"{ttft*1000:.1f}" if ttft is not None else "NA")
PY

# ---- server-side sequential latency burst (p50/p95/p99 over N short non-stream requests) ----
cat > "$WORKDIR/lat_burst.py" <<'PY'
import json, math, sys, time, urllib.request
url, model, prompt, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = json.dumps({"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 32, "stream": False}).encode()
times = []
for _ in range(n):
    t0 = time.time()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=120).read()
        times.append((time.time() - t0) * 1000.0)
    except Exception:
        pass
if not times:
    print("NA NA NA"); sys.exit(0)
times.sort()
def pct(p):
    k = (len(times) - 1) * p; f = math.floor(k); c = math.ceil(k)
    return times[int(k)] if f == c else times[f] + (times[c] - times[f]) * (k - f)
print(f"{pct(0.50):.1f} {pct(0.95):.1f} {pct(0.99):.1f}")
PY
scp "$WORKDIR/sse_ttft.py"  "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-sse-ttft.py" >/dev/null
scp "$WORKDIR/lat_burst.py" "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-lat-burst.py" >/dev/null

# ---- baseline metrics snapshot + eval-body wall clock start (delta-based counters) ----
metrics_before="$WORKDIR/metrics.before"
scrape_metrics "$metrics_before"
eval_t0="$(now)"

run_prompt() {
  local name="$1" prompt="$2" assert_mode="$3" max_tokens="${4:-800}"
  local payload="$WORKDIR/$name.json" response="$WORKDIR/$name.out" http="$WORKDIR/$name.http"
  python3 - "$payload" "$SERVED_MODEL_NAME" "$prompt" "$max_tokens" <<'PY'
import json, sys
path, model, prompt, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": max_tokens,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY
  scp "$payload" "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-eval-$name.json" >/dev/null
  local start end elapsed status tokens content pass tokps
  start="$(date +%s)"
  ssh "$CLUSTER_USER@$HEAD_HOST" "curl -sS --max-time 300 -o /tmp/dspark-eval-$name.out -w '%{http_code}' -H 'Content-Type: application/json' --data @/tmp/dspark-eval-$name.json http://127.0.0.1:$API_PORT/v1/chat/completions" > "$http" \
    || return 1
  end="$(date +%s)"
  elapsed=$(( end - start ))
  [ "$elapsed" -gt 0 ] || elapsed=1
  status="$(cat "$http")"
  [ "$status" = "200" ] || { printf '%s|%s|0|0|fail\n' "$name" "$elapsed"; return 0; }
  ssh "$CLUSTER_USER@$HEAD_HOST" "cat /tmp/dspark-eval-$name.out" > "$response"
  read -r tokens content < <(python3 - "$response" <<'PY'
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
content=data["choices"][0]["message"].get("content") or ""
tokens=data.get("usage", {}).get("completion_tokens", 0)
print(tokens, content.replace("\n", "\\n"))
PY
)
  pass=pass
  [ -n "$content" ] || pass=fail
  garble_check "$response" || pass=fail
  if [ "$assert_mode" = "contains4" ] && ! printf '%s\n' "$content" | grep -q '4'; then pass=fail; fi
  if [ "$assert_mode" = "json" ]; then
    python3 -m json.tool "$response" >/dev/null || pass=fail
    python3 - "$response" <<'PY' >/dev/null || pass=fail
import json, sys
content=json.load(open(sys.argv[1], encoding="utf-8"))["choices"][0]["message"].get("content") or ""
json.loads(content)
PY
  fi
  tokps="$(awk -v t="$tokens" -v e="$elapsed" 'BEGIN {printf "%.2f", t/e}')"
  printf '%s|%s|%s|%s|%s\n' "$name" "$elapsed" "$tokens" "$tokps" "$pass"
}

run_toolcall() {
  local payload="$WORKDIR/toolcall.json" response="$WORKDIR/toolcall.out" http="$WORKDIR/toolcall.http"
  python3 - "$payload" "$SERVED_MODEL_NAME" <<'PY'
import json, sys
path, model = sys.argv[1], sys.argv[2]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256,
}
json.dump(payload, open(path, "w", encoding="utf-8"))
PY
  scp "$payload" "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-eval-toolcall.json" >/dev/null
  local start end elapsed status tokens pass tokps
  start="$(date +%s)"
  ssh "$CLUSTER_USER@$HEAD_HOST" "curl -sS --max-time 300 -o /tmp/dspark-eval-toolcall.out -w '%{http_code}' -H 'Content-Type: application/json' --data @/tmp/dspark-eval-toolcall.json http://127.0.0.1:$API_PORT/v1/chat/completions" > "$http" \
    || return 1
  end="$(date +%s)"; elapsed=$(( end - start )); [ "$elapsed" -gt 0 ] || elapsed=1
  status="$(cat "$http")"
  [ "$status" = "200" ] || { printf 'toolcall|%s|0|0|fail\n' "$elapsed"; return 0; }
  ssh "$CLUSTER_USER@$HEAD_HOST" "cat /tmp/dspark-eval-toolcall.out" > "$response"
  read -r pass tokens < <(python3 - "$response" <<'PY'
import json, re, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
tokens = data.get("usage", {}).get("completion_tokens", 0)
name = None; args = None
calls = msg.get("tool_calls") or []
if calls:
    fn = calls[0].get("function", {}) or {}
    name = fn.get("name")
    raw = fn.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        args = None
else:
    content = msg.get("content") or ""
    for m in re.finditer(r"\{.*?\}", content, re.S):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if obj.get("name") == "get_weather":
            name = obj.get("name")
            a = obj.get("arguments")
            args = json.loads(a) if isinstance(a, str) else a
            break
ok = (name == "get_weather"
      and isinstance(args, dict)
      and "location" in args
      and "tokyo" in str(args.get("location", "")).lower())
print("pass" if ok else "fail", tokens)
PY
)
  tokps="$(awk -v t="$tokens" -v e="$elapsed" 'BEGIN {printf "%.2f", t/e}')"
  printf 'toolcall|%s|%s|%s|%s\n' "$elapsed" "$tokens" "$tokps" "$pass"
}

results="$WORKDIR/results.tsv"
: > "$results"
run_prompt arithmetic "What is 2+2? Reply with only the answer." contains4 >> "$results" || fail "arithmetic request failed" "inspect API logs"
run_prompt summary "Summarize why RDMA matters for distributed inference in one sentence." nonempty >> "$results" || fail "summary request failed" "inspect API logs"
run_prompt python30 "Write a 30-line Python function that validates and normalizes a list of host records." nonempty >> "$results" || fail "python request failed" "inspect API logs"
run_prompt json "Return exactly valid JSON with keys status and reason. No markdown." json >> "$results" || fail "JSON request failed" "inspect API logs"
run_prompt story "Write a 200-word story about a quiet datacenter cutover." nonempty >> "$results" || fail "story request failed" "inspect API logs"
run_toolcall >> "$results" || fail "tool-call request failed" "inspect API logs"

story_payload="$WORKDIR/story.json"
python3 - "$story_payload" "$SERVED_MODEL_NAME" <<'PY'
import json, sys
path, model = sys.argv[1], sys.argv[2]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Write a 200-word story about a quiet datacenter cutover."}],
    "temperature": 0,
    "max_tokens": 800,
}
json.dump(payload, open(path, "w", encoding="utf-8"))
PY
scp "$story_payload" "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-eval-story-parallel.json" >/dev/null
con_start="$(date +%s)"
for i in 1 2 3; do
  ssh "$CLUSTER_USER@$HEAD_HOST" "curl -sS --max-time 300 -o /tmp/dspark-eval-par-$i.out -w '%{http_code}' -H 'Content-Type: application/json' --data @/tmp/dspark-eval-story-parallel.json http://127.0.0.1:$API_PORT/v1/chat/completions" > "$WORKDIR/par-$i.http" &
done
scrape_metrics "$WORKDIR/metrics.qdepth"   # sampled while the 3 concurrent requests are in flight
wait
con_end="$(date +%s)"
con_elapsed=$(( con_end - con_start ))
[ "$con_elapsed" -gt 0 ] || con_elapsed=1
con_tokens=0
con_pass=pass
for i in 1 2 3; do
  [ "$(cat "$WORKDIR/par-$i.http")" = "200" ] || con_pass=fail
  ssh "$CLUSTER_USER@$HEAD_HOST" "cat /tmp/dspark-eval-par-$i.out" > "$WORKDIR/par-$i.out"
  tokens="$(python3 - "$WORKDIR/par-$i.out" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("usage", {}).get("completion_tokens", 0))
PY
)"
  garble_check "$WORKDIR/par-$i.out" || con_pass=fail
  con_tokens=$(( con_tokens + tokens ))
done
con_tokps="$(awk -v t="$con_tokens" -v e="$con_elapsed" 'BEGIN {printf "%.2f", t/e}')"
printf 'concurrency3|%s|%s|%s|%s\n' "$con_elapsed" "$con_tokens" "$con_tokps" "$con_pass" >> "$results"

if [ "${SKIP_LONGCTX:-0}" = "1" ]; then
  printf 'longctx|0|0|0|skip\n' >> "$results"
else
  long_prompt="$WORKDIR/longctx.txt"
  long_payload="$WORKDIR/longctx.json"
  python3 - "$long_prompt" <<'PY'
import sys
para = "Distributed inference depends on predictable transport, bounded queues, and careful memory planning. "
# ~104 chars/para ≈ 20 tokens; 700*14 paras ≈ 1.0M chars ≈ 200K tokens.
chunks = [para * 700 for _ in range(14)]
chunks.insert(len(chunks)//2, " The codeword mentioned exactly once is ORCHID-177. ")
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(chunks) + "\nWhat is the codeword mentioned exactly once above?")
PY
  python3 - "$long_payload" "$SERVED_MODEL_NAME" "$long_prompt" <<'PY'
import json, sys
path, model, prompt_path = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open(prompt_path, encoding="utf-8").read()
# max_tokens must clear the reasoning block: with DSPARK_REASONING=on the model spends the
# early tokens in .reasoning before emitting </think> + the answer into .content. 64 truncated
# mid-think (finish=length, empty content); 1024 lets both modes finish (non-think ~10 tok, think ~93).
json.dump({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1024}, open(path, "w", encoding="utf-8"))
PY
  scp "$long_payload" "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-eval-longctx.json" >/dev/null
  long_start="$(date +%s)"
  long_status="$(ssh "$CLUSTER_USER@$HEAD_HOST" "curl -sS --max-time 900 -o /tmp/dspark-eval-longctx.out -w '%{http_code}' -H 'Content-Type: application/json' --data @/tmp/dspark-eval-longctx.json http://127.0.0.1:$API_PORT/v1/chat/completions")" || long_status=000
  long_end="$(date +%s)"
  long_elapsed=$(( long_end - long_start ))
  [ "$long_elapsed" -gt 0 ] || long_elapsed=1
  ssh "$CLUSTER_USER@$HEAD_HOST" "cat /tmp/dspark-eval-longctx.out" > "$WORKDIR/longctx.out" || true
  long_tokens="$(python3 - "$WORKDIR/longctx.out" <<'PY' 2>/dev/null || echo 0
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("usage", {}).get("completion_tokens", 0))
PY
)"
  long_pass=pass
  [ "$long_status" = "200" ] || long_pass=fail
  python3 - "$WORKDIR/longctx.out" <<'PY' >/dev/null || long_pass=fail
import json, sys
content=json.load(open(sys.argv[1], encoding="utf-8"))["choices"][0]["message"].get("content") or ""
raise SystemExit(0 if "ORCHID-177" in content else 1)
PY
  garble_check "$WORKDIR/longctx.out" || long_pass=fail
  long_tokps="$(awk -v t="$long_tokens" -v e="$long_elapsed" 'BEGIN {printf "%.2f", t/e}')"
  printf 'longctx_ttft_total|%s|%s|%s|%s\n' "$long_elapsed" "$long_tokens" "$long_tokps" "$long_pass" >> "$results"
fi

# ---- closing metrics snapshot + eval-body wall clock end (pair with metrics_before/eval_t0) ----
eval_t1="$(now)"
metrics_after="$WORKDIR/metrics.after"
scrape_metrics "$metrics_after"

# ---- client-side streaming TTFT (soft; SKIP_TTFT guard) ----
ttft_idle_ms="NA"; ttft_load_ms="NA"
if [ "${SKIP_TTFT:-0}" != "1" ]; then
  ttft_idle_ms="$(ssh "$CLUSTER_USER@$HEAD_HOST" "python3 /tmp/dspark-sse-ttft.py 'http://127.0.0.1:$API_PORT/v1/chat/completions' '$SERVED_MODEL_NAME' 'Say hello.' 64" 2>/dev/null || echo NA)"
  python3 - "$WORKDIR/ttftload.json" "$SERVED_MODEL_NAME" <<'PY'
import json, sys
path, model = sys.argv[1], sys.argv[2]
para = "Distributed inference depends on predictable transport, bounded queues, and careful memory planning. "
prompt = para * 6500 + "\nSummarize the above in one word."
json.dump({"model": model, "messages": [{"role": "user", "content": prompt}],
           "temperature": 0, "max_tokens": 16}, open(path, "w", encoding="utf-8"))
PY
  scp "$WORKDIR/ttftload.json" "$CLUSTER_USER@$HEAD_HOST:/tmp/dspark-eval-ttftload.json" >/dev/null
  ssh "$CLUSTER_USER@$HEAD_HOST" "curl -sS --max-time 300 -o /dev/null --data @/tmp/dspark-eval-ttftload.json -H 'Content-Type: application/json' http://127.0.0.1:$API_PORT/v1/chat/completions" >/dev/null 2>&1 &
  load_pid=$!
  python3 -c 'import time;time.sleep(2)'
  ttft_load_ms="$(ssh "$CLUSTER_USER@$HEAD_HOST" "python3 /tmp/dspark-sse-ttft.py 'http://127.0.0.1:$API_PORT/v1/chat/completions' '$SERVED_MODEL_NAME' 'Say hello.' 64" 2>/dev/null || echo NA)"
  wait "$load_pid" 2>/dev/null || true
fi

# ---- client-side latency percentiles p50/p95/p99 (soft; SKIP_LATENCY guard) ----
lat_p50_ms="NA"; lat_p95_ms="NA"; lat_p99_ms="NA"
if [ "${SKIP_LATENCY:-0}" != "1" ]; then
  read -r lat_p50_ms lat_p95_ms lat_p99_ms < <(ssh "$CLUSTER_USER@$HEAD_HOST" "python3 /tmp/dspark-lat-burst.py 'http://127.0.0.1:$API_PORT/v1/chat/completions' '$SERVED_MODEL_NAME' 'Reply with the single word: ok.' 10" 2>/dev/null || echo "NA NA NA")
fi

cat > "$WORKDIR/soft_metrics.py" <<'PY'
import json, re, sys, os

metrics_before, metrics_after, metrics_qdepth, results_tsv = sys.argv[1:5]
wall = float(sys.argv[5])
ttft_idle, ttft_load = sys.argv[6], sys.argv[7]
lat_p50, lat_p95, lat_p99 = sys.argv[8], sys.argv[9], sys.argv[10]

LINE = re.compile(r'^(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$')
def load(path):
    fam = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE.match(line)
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2) or "", m.group(3)
            try:
                fam.setdefault(name, []).append((labels, float(val)))
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    return fam

def total(fam, name):
    return sum(v for _, v in fam.get(name, []))

def num(x):
    try:
        return float(x)
    except Exception:
        return None

before, after, qd = load(metrics_before), load(metrics_after), load(metrics_qdepth)

d_acc   = total(after, "vllm:spec_decode_num_accepted_tokens_total") - total(before, "vllm:spec_decode_num_accepted_tokens_total")
d_draft = total(after, "vllm:spec_decode_num_draft_tokens_total")    - total(before, "vllm:spec_decode_num_draft_tokens_total")
d_drafts= total(after, "vllm:spec_decode_num_drafts_total")          - total(before, "vllm:spec_decode_num_drafts_total")
accept_rate = (d_acc / d_draft) if d_draft > 0 else None
mean_acc_len = (d_acc / d_drafts) if d_drafts > 0 else None

def per_pos(fam):
    out = {}
    for labels, v in fam.get("vllm:spec_decode_num_accepted_tokens_per_pos_total", []):
        m = re.search(r'(?:position|pos)="?(\d+)"?', labels)
        if m:
            out[int(m.group(1))] = out.get(int(m.group(1)), 0.0) + v
    return out
pb, pa = per_pos(before), per_pos(after)
pos_decay = []
for p in sorted(set(pb) | set(pa)):
    dv = pa.get(p, 0.0) - pb.get(p, 0.0)
    pos_decay.append(round(dv / d_drafts, 3) if d_drafts > 0 else None)

d_prompt = total(after, "vllm:prompt_tokens_total")     - total(before, "vllm:prompt_tokens_total")
d_gen    = total(after, "vllm:generation_tokens_total")  - total(before, "vllm:generation_tokens_total")
prefill_tps = d_prompt / wall if wall > 0 else 0.0
decode_tps  = d_gen    / wall if wall > 0 else 0.0

q_run  = total(qd, "vllm:num_requests_running")
q_wait = total(qd, "vllm:num_requests_waiting")

def hist_mean_ms(fam, base):
    s = total(fam, base + "_sum"); c = total(fam, base + "_count")
    return (s / c * 1000.0) if c > 0 else None
ttft_life  = hist_mean_ms(after, "vllm:time_to_first_token_seconds")
e2e_life   = hist_mean_ms(after, "vllm:e2e_request_latency_seconds")
tpot_life  = hist_mean_ms(after, "vllm:request_time_per_output_token_seconds")

rows = [ln.split("|") for ln in open(results_tsv, encoding="utf-8") if ln.strip()]
funct = [r for r in rows if len(r) >= 5 and r[4].strip() != "skip"]
n_pass = sum(1 for r in funct if r[4].strip() == "pass")
correctness = (n_pass / len(funct)) if funct else 0.0

garble_clean = num(os.environ.get("GARBLE_CLEAN", "1.0")) or 0.0

SLO = {"ttft_idle_ms": 1000.0, "ttft_load_ms": 2000.0, "e2e_p95_ms": 8000.0,
       "accept_rate": 0.60, "accept_len": 2.0}
WEIGHTS = {"correctness": 0.50, "garble": 0.15, "latency_slo": 0.25, "spec_decode": 0.10}

def clamp01(x): return max(0.0, min(1.0, x))

lat_parts = []
ti = num(ttft_idle);  p95 = num(lat_p95)
if ti  is not None: lat_parts.append(1.0 if ti  <= SLO["ttft_idle_ms"] else clamp01(SLO["ttft_idle_ms"]/ti))
if p95 is not None: lat_parts.append(1.0 if p95 <= SLO["e2e_p95_ms"]   else clamp01(SLO["e2e_p95_ms"]/p95))
latency_slo = sum(lat_parts)/len(lat_parts) if lat_parts else None

spec_health = clamp01(accept_rate / SLO["accept_rate"]) if accept_rate is not None else None

comp = {"correctness": correctness, "garble": garble_clean,
        "latency_slo": latency_slo, "spec_decode": spec_health}
num_w = {k: WEIGHTS[k] for k, v in comp.items() if v is not None}
den = sum(num_w.values()) or 1.0
composite = 100.0 * sum(WEIGHTS[k]*comp[k] for k in num_w) / den

def fmt(x, nd=1): return f"{x:.{nd}f}" if isinstance(x, float) else ("n/a" if x is None else str(x))
print(f"spec_decode_acceptance: draft_acceptance_rate={fmt(accept_rate,3)} mean_accepted_len={fmt(mean_acc_len,2)} per_pos_decay={pos_decay if pos_decay else 'n/a'}")
print(f"throughput_split: prefill_tok_s={fmt(prefill_tps,1)} decode_tok_s={fmt(decode_tps,1)} (aggregate over eval body, all requests)")
print(f"queue_depth_at_concurrency: running={fmt(q_run,0)} waiting={fmt(q_wait,0)}")
print(f"server_hist_lifetime_means: ttft_ms={fmt(ttft_life,1)} e2e_ms={fmt(e2e_life,1)} tpot_ms={fmt(tpot_life,1)} (cumulative-since-boot; context only)")
print(f"client_ttft: idle_ms={ttft_idle} under_load_ms={ttft_load}")
print(f"client_latency: p50_ms={lat_p50} p95_ms={lat_p95} p99_ms={lat_p99}")
print(f"composite_score: {composite:.1f}/100  (correctness={fmt(correctness,2)} garble={fmt(garble_clean,2)} latency_slo={fmt(latency_slo,2)} spec_decode={fmt(spec_health,2)})")
PY

echo "test | time | completion tokens | tok/s | pass/fail"
awk -F '|' '{printf "%s | %s | %s | %s | %s\n", $1, $2, $3, $4, $5}' "$results"
print_prefix_cache_hit_rate

# independent garble-clean signal: re-scan every captured response (does not affect the gate)
garble_fails=0; garble_total=0
for f in "$WORKDIR"/*.out; do
  [ -e "$f" ] || continue
  garble_total=$(( garble_total + 1 ))
  garble_check "$f" || garble_fails=$(( garble_fails + 1 ))
done
garble_clean="$(awk -v b="$garble_fails" -v t="$garble_total" 'BEGIN{printf "%.3f", (t>0 ? (t-b)/t : 1)}')"

eval_wall="$(awk -v a="$eval_t0" -v b="$eval_t1" 'BEGIN{printf "%.3f", ((b-a)>0 ? (b-a) : 1)}')"
GARBLE_CLEAN="$garble_clean" python3 "$WORKDIR/soft_metrics.py" \
  "$metrics_before" "$metrics_after" "${WORKDIR}/metrics.qdepth" "$results" \
  "$eval_wall" "$ttft_idle_ms" "$ttft_load_ms" \
  "$lat_p50_ms" "$lat_p95_ms" "$lat_p99_ms" || echo "composite_score: n/a (soft-metrics analyzer error)"
if awk -F '|' '$5 == "fail" {bad=1} END {exit bad}' "$results"; then
  echo "ok: eval complete"
else
  fail "one or more eval probes failed" "review summary table and vLLM logs"
fi
