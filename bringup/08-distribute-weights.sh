#!/usr/bin/env bash
# Copy model weights from head to worker over QSFP.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

remote_model_dir="$HF_CACHE/hub/models--${DSPARK_MODEL//\//--}"

ssh "$CLUSTER_USER@$WORKER_HOST" "mkdir -p '$HF_CACHE/hub'" \
  || fail "could not create worker HF hub dir" "check worker SSH and permissions"
echo "ok: worker HF hub dir exists"

# --delete is scoped to the transferred model dir only: without it, a stale worker file
# (e.g. an older snapshot left by a revision flip) makes the file/byte parity gate below
# permanently unsatisfiable — the checksum retry re-copies but never removes.
ssh "$CLUSTER_USER@$HEAD_HOST" "rsync -a --delete --partial --info=progress2 '$remote_model_dir' '$CLUSTER_USER@$WORKER_R1:$HF_CACHE/hub/'" \
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
  ssh "$CLUSTER_USER@$HEAD_HOST" "rsync -a -c --delete --partial --info=progress2 '$remote_model_dir' '$CLUSTER_USER@$WORKER_R1:$HF_CACHE/hub/'" \
    || fail "checksum rsync failed" "verify QSFP SSH and disk space"
  head_stats="$(stats "$HEAD_HOST")" || fail "could not re-stat head weights" "verify source dir"
  worker_stats="$(stats "$WORKER_HOST")" || fail "could not re-stat worker weights" "verify destination dir"
fi

echo "head weights final:   $head_stats"
echo "worker weights final: $worker_stats"
[ "$head_stats" = "$worker_stats" ] || fail "weight file counts or byte totals differ" "rerun 08-distribute-weights.sh after checking disk space"
echo "ok: weight stats match"

# Shard-level completeness on the worker, same check 07 runs head-side: file/byte parity
# can't see a snapshot whose shard symlinks point at blobs that never arrived.
ssh "$CLUSTER_USER@$WORKER_HOST" "DIR='$remote_model_dir' bash -s" <<'REMOTE' \
  || fail "worker shard verification failed" "rerun 08-distribute-weights.sh after checking disk space"
set -euo pipefail
# Resolve THE snapshot offline serving will use (refs/main), not whichever snapshot
# find(1) hits first — same rule as 07's head-side check.
[ -f "$DIR/refs/main" ] || { echo "FAIL: $DIR/refs/main missing on the worker — rerun 07 then 08" >&2; exit 1; }
snap_dir="$DIR/snapshots/$(cat "$DIR/refs/main")"
[ -d "$snap_dir" ] || { echo "FAIL: refs/main points at a missing snapshot ($snap_dir)" >&2; exit 1; }
index="$snap_dir/model.safetensors.index.json"
[ -f "$index" ] || { echo "FAIL: model.safetensors.index.json missing under $snap_dir" >&2; exit 1; }
missing_shards="$(python3 - "$index" <<'PY'
import json, os, sys
index_path = sys.argv[1]
snap_dir = os.path.dirname(index_path)
weight_map = json.load(open(index_path, encoding="utf-8"))["weight_map"]
missing = sorted({shard for shard in weight_map.values()
                  if not os.path.isfile(os.path.join(snap_dir, shard))})
for shard in missing:
    print("missing shard:", shard, file=sys.stderr)
print(len(missing))
PY
)"
echo "missing_shards=$missing_shards"
[ "$missing_shards" = "0" ] || { echo "FAIL: $missing_shards safetensors shards missing on the worker" >&2; exit 1; }
REMOTE
echo "ok: worker safetensors shards complete"
