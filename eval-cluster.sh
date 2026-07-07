#!/usr/bin/env bash
# Evaluate the running cluster through the head node loopback API.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

ssh "$CLUSTER_USER@$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/health" >/dev/null \
  || fail "API health check failed" "start the cluster before evaluation"
echo "ok: health"

WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

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

results="$WORKDIR/results.tsv"
: > "$results"
run_prompt arithmetic "What is 2+2? Reply with only the answer." contains4 >> "$results" || fail "arithmetic request failed" "inspect API logs"
run_prompt summary "Summarize why RDMA matters for distributed inference in one sentence." nonempty >> "$results" || fail "summary request failed" "inspect API logs"
run_prompt python30 "Write a 30-line Python function that validates and normalizes a list of host records." nonempty >> "$results" || fail "python request failed" "inspect API logs"
run_prompt json "Return exactly valid JSON with keys status and reason. No markdown." json >> "$results" || fail "JSON request failed" "inspect API logs"
run_prompt story "Write a 200-word story about a quiet datacenter cutover." nonempty >> "$results" || fail "story request failed" "inspect API logs"

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
json.dump({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 64}, open(path, "w", encoding="utf-8"))
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
  long_tokps="$(awk -v t="$long_tokens" -v e="$long_elapsed" 'BEGIN {printf "%.2f", t/e}')"
  printf 'longctx_ttft_total|%s|%s|%s|%s\n' "$long_elapsed" "$long_tokens" "$long_tokps" "$long_pass" >> "$results"
fi

echo "test | time | completion tokens | tok/s | pass/fail"
awk -F '|' '{printf "%s | %s | %s | %s | %s\n", $1, $2, $3, $4, $5}' "$results"
if awk -F '|' '$5 == "fail" {bad=1} END {exit bad}' "$results"; then
  echo "ok: eval complete"
else
  fail "one or more eval probes failed" "review summary table and vLLM logs"
fi
