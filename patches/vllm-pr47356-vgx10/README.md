# vLLM PR #47356 patch — exclude `kv_cache_memory_bytes` from `CacheConfig.compute_hash`

This is the source-patch layer for the **current default** serving lane: the `dspark-vllm-gx10`
base (vLLM 0.25.1) with a single upstream fix applied on top. It plays the same role for the 0.25.1
lane that `patches/flashinfer-pr3615/` plays for the prior 0.21.x lane — a thin, source-only overlay
on a pinned base image, built and verified by `bringup/05-build-image.sh`.

## What & why

The base image is
`ghcr.io/anemll/dspark-vllm-gx10@sha256:d0a0c050252dd0b64a3213c728d1c15db9eb37602593ba37534bc708dd223ae7`
(a community GB10 port by GitHub user `anemll`), pinning vLLM `v0.25.1`
(`752a3a504485790a2e8491cacbb35c137339ad34`). Its vLLM pin is dated 2026-07-12 but predates
upstream **vLLM [PR #47356](https://github.com/vllm-project/vllm/pull/47356)**, which merged
2026-07-07 — the base is newer overall yet just missed this one fix.

Pre-patch, `CacheConfig.compute_hash()` in `vllm/config/cache.py` does not exclude
`kv_cache_memory_bytes` from its hash. That field is a runtime-measured KV-sizing knob, **not** a
compiled-graph-shape factor, but leaking it into the hash means setting it perturbs the
`torch.compile` cache key and needlessly invalidates the warm FlashInfer/Triton autotune caches
across every boot. The fix adds it to the `ignored_factors` set alongside the equivalent
`gpu_memory_utilization`.

The patch content is **identical to the literal upstream diff** — no hand-adaptation was needed.
The upstream diff's context (`"gpu_memory_utilization",` immediately followed by
`"is_attention_free",` inside the `ignored_factors` set literal) matches the base's `cache.py`
exactly, even though the base's set has several unrelated entries further down. `git apply --check`
of the trimmed `cache.py`-only hunk applies with zero fuzz.

## In-image path differs from the prior lane

The gx10 base installs vLLM at
`/usr/local/lib/python3.12/dist-packages/vllm/config/cache.py`, **not** the prior 0.21.x stage-c
base's `/opt/env/lib/python3.12/site-packages/...`. If you are adapting the prior lane's
`patches/vllm-pr47356/`-style muscle memory, this path change is the one thing to re-check — the
patch content itself is the same.

## Files

- `cache-hash-exclude-kv-bytes.patch` — the upstream diff, trimmed to the `vllm/config/cache.py`
  hunk (the PR also adds a unit test under `tests/`, which a served image does not need).
- `Dockerfile` — thin layer: `FROM` the digest-pinned base + `COPY cache.py` over the installed
  one. No `RUN` steps (see reproducibility note).
- `build-patched-image.sh` — extracts `cache.py` from the base, applies + verifies the patch,
  builds the tag reproducibly, runs the mtime fixup pass, and asserts the hash behavior directly.
- `fixup-layer-mtime.py` — post-build normalization pass; see below.
- `cache.py` — **not checked in**; produced transiently at build time and deleted afterward.

## Cross-node reproducibility (byte-identical image ID)

The kit distributes one built image head → worker (`bringup/06-distribute-image.sh`), so both nodes
run bit-identical bytes regardless. But the builder is also written to be **reproducible** — running
it independently on each node yields the same image ID — which took four things together, each
verified insufficient alone:

1. **Host-side mtime pin** on the COPY source file (`touch -d "@0"`) — buildx's `COPY` otherwise
   carries the source file's real (node- and time-dependent) mtime into the layer.
2. **`--output type=docker,rewrite-timestamp=true`** with `SOURCE_DATE_EPOCH=0` as a shell env var
   (not a `--build-arg`, which this flag does not read) — normalizes the image config's `Created`
   timestamp and other export-time metadata.
3. **`--no-cache`** — a cached COPY layer from a prior, non-normalized build exports with its stale
   diff-id even under `rewrite-timestamp=true` on a later invocation.
4. **A post-build fixup pass** (`fixup-layer-mtime.py`). `cache.py` already exists in the base, so
   the `COPY` overwrites it; overlayfs then bumps the **parent directory's** mtime to build
   wall-clock time as a side effect of the copy-up. `rewrite-timestamp=true` does not normalize that
   directory entry — only file content entries and the image config's own timestamps. This alone
   left the final layer's diff-id, and therefore the image ID, node-dependent. The fixup pass
   re-opens the saved image tar, zeroes every top-layer entry's mtime (including the directory
   entry), recomputes the diff-id, rewrites the config + manifest to match, and `docker load`s the
   corrected image back under the same tag.

`RUN` steps are avoided in the Dockerfile for the same underlying reason: any `RUN` stamps the
container's `/etc`, `/proc`, `/sys` mountpoint dirs with build-wall-clock time regardless of
`SOURCE_DATE_EPOCH`.

## Result (verified 2026-07-15)

- **Base:** `ghcr.io/anemll/dspark-vllm-gx10@sha256:d0a0c050…223ae7`
- **Patched serving tag:** `vllm-dspark-runtime:vgx10-cand1-pr47356`
- **Image ID (both nodes, byte-identical):**
  `sha256:99b05acd7e23c447e549122ed8a932d6ca959aa8671c32b6ee7f85f821222a22`
- **Smoke test (both nodes):** `OK: hash unaffected by kv_cache_memory_bytes`
  (`CacheConfig().compute_hash() == CacheConfig(kv_cache_memory_bytes=1<<30).compute_hash()`)

## Rollback

The unpatched base image is the rollback target: set `DSPARK_VLLM_IMAGE` to the value of
`DSPARK_VLLM_BASE_IMAGE` (the digest-pinned base) in `cluster.env`, re-render both nodes, and do the
normal ordered-pair restart. Unlike the FlashInfer overlay, this patch is a plain Python file swap
with no JIT header involved, so no sampling-cache clear is needed when switching tags.

## Risk

Near-zero. A single-line addition to a Python set literal that already governs an equivalent field
(`gpu_memory_utilization`) the same way — identical to the merged upstream PR's own diff. Worst case
if wrong: cache-key churn (more recompiles), not a crash or a correctness issue.
