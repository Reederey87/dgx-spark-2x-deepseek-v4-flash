# vLLM 0.26.0 gx10-overlay kit (the cand4 production lane)

**Status: this is the current production image recipe (promoted 2026-07-28).** It supersedes
the 0.25.1 lane (`patches/vllm-pr47356-vgx10/`, kept for rollback).

## What & why

The serving image is built **in-house** as a thin Python layer on the official
`linux/arm64` `vllm/vllm-openai:v0.26.0` — no CUDA rebuild. The layer carries
`gx10-overlay-026.patch` (13 files, +1130/−52):

- **the `dspark-vllm-gx10` GB10 overlay** (Anemll, see NOTICE), cherry-picked onto v0.26.0
  with zero conflicts: `nvfp4_ds_mla` KV-quant plumbing, the b12x MXFP4 MoE backend, and
  the FlashInfer SM120/SM121 sparse-MLA bridge;
- **the warn-only `dspark_block_size` guard** — v0.26.0 hard-rejects
  `num_speculative_tokens=3 < dspark_block_size: 5` claiming garbled output; on this stack
  n=3 is measured garble-clean to 944K context (and n=5 costs −12% throughput), so the
  overlay logs and proceeds (upstream issue vllm-project/vllm#50012);
- **the zero-token-prefill-chunk guard** — 0.26's capture/warmup batches (padded to
  `max_num_seqs`) emit zero-token prefill spans that crash flashinfer's sparse-MLA segment
  normalize (`cannot reshape tensor of 0 elements into shape [0, -1]`). The overlay skips
  empty chunks (upstream precedent: #48957). Every DSpark/0.26 deployment with this batch
  shape needs this guard;
- **backports #50004 + #49486** (post-0.26.0 DSv4 perf): adaptive topk width + skipping
  topk/router when not needed. Measured here: they lift spec-decode acceptance from 0.592
  (stock 0.26.0) to **0.713** and throughput +4.3% — a stock 0.26.0 without them carries a
  −7% acceptance residual on this deployment, and reverting the release's own perf kernels
  (#48137/#48660) makes it monotonically *worse*, not better.

## Build (one node), then distribute

```bash
bash build-patched-image.sh    # tag: vllm-dspark-runtime:v026-gx10-cand4-backports

TAG=vllm-dspark-runtime:v026-gx10-cand4-backports
docker save "$TAG" | zstd -T0 -3 > ~/v026c4-image.tar.zst
rsync -a --partial ~/v026c4-image.tar.zst <worker-host>:~/
ssh <worker-host> 'zstd -dc ~/v026c4-image.tar.zst | docker load'
# assert parity — these MUST print the same ID:
docker image inspect --format '{{.Id}}' "$TAG"
ssh <worker-host> "docker image inspect --format '{{.Id}}' $TAG"
```

pip-install layers embed timestamps, so per-node builds are **not** byte-identical —
build once, distribute. The script: locates the base's vllm package dir, extracts the 13
upstream files, `git apply --check`s the patch, builds, and smoke-verifies (version,
both guards, both backports, b12x + flashinfer imports). It needs the official base
pulled locally (see `bringup/05-build-image.sh`).

## Evidence (same-day A/B vs the 0.25.1 lane, 2026-07-28)

C8 throughput **87.94 vs 84.29 tok/s (+4.3%)** · eval acceptance **0.713/0.700 vs 0.638
(+12%)** · composite **100/100 ×2** · bench acceptance flat (0.43) · 3/3 needle retrievals
@944,471 tokens · battery acceptance 0.784. One known cost: the reported KV pool re-baselined
2,948,751 → **2,075,155 tokens** at the unchanged 19.85 GiB byte-pin — **not** a physical
layout change (bytes/blocks byte-identical): 0.26's #47728 doubles the sliding-window
admission reservation under `--async-scheduling`, inflating the synthetic 1M-request figure.
Single 1M-context requests are unaffected. Full record in `docs/10-vllm-026-rebase.md`.
