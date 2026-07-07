#!/usr/bin/env bash
# Copy model weights from head to worker over QSFP.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

remote_model_dir="$HF_CACHE/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"

ssh "$CLUSTER_USER@$WORKER_HOST" "mkdir -p '$HF_CACHE/hub'" \
  || fail "could not create worker HF hub dir" "check worker SSH and permissions"
echo "ok: worker HF hub dir exists"

ssh "$CLUSTER_USER@$HEAD_HOST" "rsync -a --partial --info=progress2 '$remote_model_dir' '$CLUSTER_USER@$WORKER_R1:$HF_CACHE/hub/'" \
  || fail "weights rsync failed" "verify head-to-worker QSFP SSH and disk space"
echo "ok: weights rsync completed"

stats() {
  local host="$1"
  ssh "$CLUSTER_USER@$host" "DIR='$remote_model_dir' bash -s" <<'REMOTE'
set -euo pipefail
files="$(find "$DIR" -type f | wc -l | tr -d ' ')"
bytes="$(du -sb "$DIR" | awk '{print $1}')"
printf '%s %s\n' "$files" "$bytes"
REMOTE
}

head_stats="$(stats "$HEAD_HOST")" || fail "could not stat head weights" "verify download completed"
worker_stats="$(stats "$WORKER_HOST")" || fail "could not stat worker weights" "verify rsync completed"

echo "head weights:   $head_stats"
echo "worker weights: $worker_stats"

if [ "$head_stats" != "$worker_stats" ]; then
  echo "WARN: weight stats mismatch; retrying rsync with checksum" >&2
  ssh "$CLUSTER_USER@$HEAD_HOST" "rsync -a -c --partial --info=progress2 '$remote_model_dir' '$CLUSTER_USER@$WORKER_R1:$HF_CACHE/hub/'" \
    || fail "checksum rsync failed" "verify QSFP SSH and disk space"
  head_stats="$(stats "$HEAD_HOST")" || fail "could not re-stat head weights" "verify source dir"
  worker_stats="$(stats "$WORKER_HOST")" || fail "could not re-stat worker weights" "verify destination dir"
fi

echo "head weights final:   $head_stats"
echo "worker weights final: $worker_stats"
[ "$head_stats" = "$worker_stats" ] || fail "weight file counts or byte totals differ" "rerun 08-distribute-weights.sh after checking disk space"
echo "ok: weight stats match"
