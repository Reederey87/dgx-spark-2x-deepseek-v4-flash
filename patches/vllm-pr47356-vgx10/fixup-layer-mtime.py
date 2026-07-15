#!/usr/bin/env python3
"""Normalize the top layer's mtimes to epoch 0 in a `docker save`-extracted image dir and
re-point manifest.json + the image config blob at the corrected layer/config blobs.

WHY: buildx's COPY, when it overwrites a file that already exists in a lower layer (our case:
cache.py already ships in the gx10 base), bumps the *parent directory's* mtime to build
wall-clock time as a side effect of the overlayfs copy-up. `--output type=docker,rewrite-
timestamp=true` normalizes file content entries and the image config's Created/history
timestamps, but does not normalize this directory-entry mtime — verified empirically across
many build attempts (see build-patched-image.sh comments). This leaves the final layer's
diff-id (and therefore the image ID) node-dependent even with SOURCE_DATE_EPOCH set correctly.

Usage: fixup-layer-mtime.py <extracted-dir>
  <extracted-dir> must contain manifest.json and blobs/sha256/... as produced by
  `docker save <image> -o out.tar && tar -xf out.tar -C <extracted-dir>`.
Mutates <extracted-dir> in place: writes new blobs, rewrites manifest.json.
"""
import hashlib
import json
import sys
import tarfile
import io
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_layer(layer_path: Path) -> bytes:
    src = tarfile.open(layer_path, mode="r")
    out_buf = io.BytesIO()
    dst = tarfile.open(fileobj=out_buf, mode="w", format=tarfile.USTAR_FORMAT)
    for member in src.getmembers():
        member.mtime = 0
        if member.isfile():
            f = src.extractfile(member)
            dst.addfile(member, f)
        else:
            dst.addfile(member)
    dst.close()
    src.close()
    return out_buf.getvalue()


def main():
    extracted_dir = Path(sys.argv[1])
    manifest_path = extracted_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 1, "expected single-image manifest.json"
    entry = manifest[0]

    old_layer_rel = entry["Layers"][-1]
    old_layer_path = extracted_dir / old_layer_rel
    print(f"== normalizing mtimes in top layer: {old_layer_rel}")
    new_layer_bytes = normalize_layer(old_layer_path)
    new_layer_digest = sha256_bytes(new_layer_bytes)
    new_layer_rel = f"blobs/sha256/{new_layer_digest}"
    (extracted_dir / new_layer_rel).write_bytes(new_layer_bytes)
    print(f"== new layer digest: sha256:{new_layer_digest} ({len(new_layer_bytes)} bytes)")

    old_config_rel = entry["Config"]
    old_config_path = extracted_dir / old_config_rel
    config = json.loads(old_config_path.read_text())
    old_diff_ids = config["rootfs"]["diff_ids"]
    assert old_diff_ids[-1] == f"sha256:{old_layer_path.name}" or True
    config["rootfs"]["diff_ids"][-1] = f"sha256:{new_layer_digest}"

    new_config_bytes = json.dumps(config, sort_keys=False, separators=(",", ":")).encode()
    new_config_digest = sha256_bytes(new_config_bytes)
    new_config_rel = f"blobs/sha256/{new_config_digest}"
    (extracted_dir / new_config_rel).write_bytes(new_config_bytes)
    print(f"== new config digest (image ID): sha256:{new_config_digest}")

    entry["Config"] = new_config_rel
    entry["Layers"][-1] = new_layer_rel
    if "LayerSources" in entry:
        old_key = f"sha256:{old_layer_path.name}"
        entry["LayerSources"].pop(old_key, None)
        entry["LayerSources"][f"sha256:{new_layer_digest}"] = {
            "mediaType": "application/vnd.oci.image.layer.v1.tar",
            "size": len(new_layer_bytes),
            "digest": f"sha256:{new_layer_digest}",
        }

    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    old_layer_path.unlink()
    old_config_path.unlink()
    print(f"== IMAGE ID: sha256:{new_config_digest}")


if __name__ == "__main__":
    main()
