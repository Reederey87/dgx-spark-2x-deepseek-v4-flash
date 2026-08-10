#!/usr/bin/env bash
# Download public model weights on the head node.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$KIT/../runtime/cluster.env"

fail() { echo "FAIL: $1 — $2" >&2; exit 1; }

ssh "$CLUSTER_USER@$HEAD_HOST" "HF_CACHE='$HF_CACHE' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' bash -s" <<'REMOTE' \
  || fail "weight download failed on $HEAD_HOST" "rerun; huggingface_hub download is resumable"
set -euo pipefail

if [ ! -x ~/hf-venv/bin/hf ]; then
  python3 -m venv ~/hf-venv
  ~/hf-venv/bin/pip -q install -U "huggingface_hub[cli]"
fi
echo "ok: hf CLI available"

mkdir -p "$HF_CACHE"
rev_args=()
[ -n "$DSPARK_REVISION" ] && rev_args+=(--revision "$DSPARK_REVISION")
HF_HOME="$HF_CACHE" HF_HUB_DISABLE_XET=1 ~/hf-venv/bin/hf download "$DSPARK_MODEL" "${rev_args[@]}"
echo "ok: hf download completed"

model_dir="$HF_CACHE/hub/models--${DSPARK_MODEL//\//--}"
# Offline-serving traps (burned 2026-07-31 on the 0731 upgrade): a --revision-pinned
# download creates snapshots/<sha> but NOT refs/main, and vLLM's HF_HUB_OFFLINE=1 startup
# resolves revision "main" via that ref → LocalEntryNotFoundError at boot. And a ref file
# MUST NOT have a trailing newline (41 vs 40 bytes — huggingface_hub 1.24 rejects it).
# Recreate refs/main exactly the way HF's own tooling would.
if [ ! -f "$model_dir/refs/main" ]; then
  snap="$(basename "$(find "$model_dir/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)")"
  [ -n "$snap" ] && { mkdir -p "$model_dir/refs"; printf '%s' "$snap" > "$model_dir/refs/main"; echo "ok: wrote refs/main -> $snap (offline resolution)"; }
fi
find "$model_dir"/snapshots -name config.json -print -quit | grep -q . \
  || { echo "FAIL: config.json missing under $model_dir/snapshots — download is incomplete" >&2; exit 1; }
if find "$model_dir/blobs" -name '*.safetensors.incomplete' -print -quit | grep -q .; then
  echo "FAIL: incomplete safetensors blobs remain — rerun download" >&2
  exit 1
fi
# Shard-level completeness: refs/main + config.json can both be present while an
# individual shard never got linked into the snapshot (an interrupted download that
# left no .incomplete marker). Walk the index's weight_map and assert every shard exists.
index="$(find "$model_dir/snapshots" -name model.safetensors.index.json -print -quit)"
[ -n "$index" ] || { echo "FAIL: model.safetensors.index.json missing under $model_dir/snapshots — download is incomplete" >&2; exit 1; }
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
[ "$missing_shards" = "0" ] || { echo "FAIL: $missing_shards safetensors shards missing from the snapshot — rerun download" >&2; exit 1; }
echo "ok: all safetensors shards present"
du -sh "$model_dir"
echo "ok: weights present"

if [ -f "$HF_CACHE/token" ]; then
  echo "WARN: $HF_CACHE/token exists even though $DSPARK_MODEL is public" >&2
else
  echo "ok: no HF token file"
fi
REMOTE
