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
du -sh "$model_dir"
echo "ok: weights present"

if [ -f "$HF_CACHE/token" ]; then
  echo "WARN: $HF_CACHE/token exists even though $DSPARK_MODEL is public" >&2
else
  echo "ok: no HF token file"
fi
REMOTE
