#!/usr/bin/env bash
# Offline structural validation for the generic deployment kit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() { echo "FAIL: $*" >&2; exit 1; }
need() { grep -Fq -- "$1" "$2" || fail "$2 is missing: $1"; }

scripts=()
while IFS= read -r script; do
  scripts+=("$script")
done < <(find bringup runtime patches tests -type f -name '*.sh' -print | sort)
for script in "${scripts[@]}"; do
  bash -n "$script"
done
echo "ok: bash syntax (${#scripts[@]} scripts)"

# Production defaults: full-source c8r lane (docs/14).
need 'DSPARK_VLLM_BASE_IMAGE=vllm-dspark-src:v0261-main-c8' runtime/cluster.env.example
need 'DSPARK_VLLM_IMAGE=vllm-dspark-runtime:v0261-main-c8r' runtime/cluster.env.example
need 'vllm-dspark-runtime:v0261-main-c8r' runtime/docker-compose.dspark.yml
need '48bada6ea4' patches/vllm-0261-main-c8r/README.md
need 'current production image recipe' patches/vllm-0261-main-c8r/README.md
need 'overlay0261' patches/vllm-0261-main-c8r/README.md
# Full-source image keeps its own compile-cache roots (the -c8r set) — see docs/14.
need 'vllm-cache-c8r' runtime/docker-compose.dspark.yml
need 'triton-cache-c8r' runtime/cluster.env.example
need 'tilelang-c8r' runtime/cluster.env.example
# cand7 remains the first image rollback rung (prior build script + docs/13).
need 'v026-gx10-cand7-backports' runtime/cluster.env.example
need 'cand7 lane' patches/vllm-026-rebase/README.md
need 'vLLM PR #47356' patches/vllm-pr47356-vgx10/README.md
need 'kv_cache_memory_bytes' patches/vllm-pr47356-vgx10/cache-hash-exclude-kv-bytes.patch
need 'bringup/05-build-image.sh' bringup/05-build-image.sh
need 'patches/vllm-0261-main-c8r/build-0261-image.sh' bringup/05-build-image.sh
need 'GLOO_SOCKET_IFNAME=enp1s0f1np1' runtime/cluster.env.example
need 'SHUTDOWN_TIMEOUT=30' runtime/cluster.env.example
need '--shutdown-timeout ${SHUTDOWN_TIMEOUT:-30}' runtime/docker-compose.dspark.yml
need 'vllm-dsv4-xid-monitor.service' bringup/install-services.sh
need 'XID_NOTIFY_COOLDOWN_SEC=300' runtime/cluster.env.example
need '.last-xid-$code' runtime/xid-monitor.sh
need 'FlashInfer PR #3615' patches/flashinfer-pr3615/README.md
need 'docker save "$DSPARK_VLLM_IMAGE" "$DSPARK_VLLM_BASE_IMAGE"' bringup/06-distribute-image.sh
need '*/cached_ops/sampling' patches/flashinfer-pr3615/clear-sampling-cache.sh

# The shipped defaults must satisfy the distributed and graph-capture invariants.
# shellcheck disable=SC1091
source runtime/cluster.env.example
[ "$MASTER_ADDR" = "$HEAD_R1" ] || fail "MASTER_ADDR must equal HEAD_R1"
[ -n "$GLOO_SOCKET_IFNAME" ] || fail "GLOO_SOCKET_IFNAME must be pinned"
required_capture=$((MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1) * 2))
[ "$MAX_CUDAGRAPH_CAPTURE_SIZE" -ge "$required_capture" ] \
  || fail "MAX_CUDAGRAPH_CAPTURE_SIZE must be >= $required_capture (the vLLM 0.25+ capture-ceiling formula)"
echo "ok: serve invariants"

# Test-only Xid injection must work without cluster.env, write no files, and
# distinguish catastrophic from log-only codes.
xid_bad="$(bash runtime/xid-monitor.sh --test \
  'NVRM: Xid (PCI:0000:0f:00): 119, synthetic classification test' 2>&1)"
grep -Fq 'CATASTROPHIC Xid 119' <<<"$xid_bad" || fail "Xid 119 classification failed"
grep -Fq 'capture and notification suppressed' <<<"$xid_bad" || fail "Xid test mode is unsafe"
xid_info="$(bash runtime/xid-monitor.sh --test \
  'NVRM: Xid (PCI:0000:0f:00): 13, synthetic classification test' 2>&1)"
grep -Fq 'Xid 13 seen (log-only)' <<<"$xid_info" || fail "Xid 13 classification failed"
if invalid_cooldown="$(XID_NOTIFY_COOLDOWN_SEC=invalid bash runtime/xid-monitor.sh --test 2>&1)"; then
  fail "invalid Xid cooldown was accepted"
fi
grep -Fq 'XID_NOTIFY_COOLDOWN_SEC must be a non-negative integer' <<<"$invalid_cooldown" \
  || fail "invalid Xid cooldown failed without the expected diagnostic"
echo "ok: Xid classification"

# Render and parse Compose in an isolated copy so validation never creates a
# site-local runtime/cluster.env in the checkout.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -R runtime "$tmp/runtime"
cp "$tmp/runtime/cluster.env.example" "$tmp/runtime/cluster.env"
bash "$tmp/runtime/render-env.sh" head >/dev/null
need 'GLOO_SOCKET_IFNAME=enp1s0f1np1' "$tmp/runtime/.env.dspark"
need 'SHUTDOWN_TIMEOUT=30' "$tmp/runtime/.env.dspark"

# A mistyped reasoning value must fail render-env.sh fast, not render a silently-wrong
# .env.dspark (cluster.env assigns unconditionally, so the bad value goes into the copy).
sed -i.bak 's/^DSPARK_REASONING_EFFORT=.*/DSPARK_REASONING_EFFORT=hihg/' "$tmp/runtime/cluster.env" \
  && rm -f "$tmp/runtime/cluster.env.bak"
if bad_effort="$(bash "$tmp/runtime/render-env.sh" head 2>&1)"; then
  fail "invalid DSPARK_REASONING_EFFORT was accepted"
fi
grep -Fq 'DSPARK_REASONING_EFFORT must be high|max' <<<"$bad_effort" \
  || fail "invalid DSPARK_REASONING_EFFORT failed without the expected diagnostic"
echo "ok: reasoning-value guard"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  docker compose --env-file "$tmp/runtime/.env.dspark" \
    -f "$tmp/runtime/docker-compose.dspark.yml" config >/dev/null
  echo "ok: docker compose config"
else
  echo "skip: docker compose config (Docker Compose unavailable)"
fi

echo "PASS: offline runtime validation"
