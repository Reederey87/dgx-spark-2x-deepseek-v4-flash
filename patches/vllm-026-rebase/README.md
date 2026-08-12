# vLLM 0.26.0 gx10-overlay kit (the cand7 lane — first image rollback)

**Status: superseded 2026-08-11 — kept as the first image rollback rung.** The current
production recipe is the full-source main @48bada6ea4 lane in
[patches/vllm-0261-main-c8r/](../vllm-0261-main-c8r/) (see docs/14). This cand7 lane was
promoted 2026-08-10, superseding cand4 (2026-07-28) and the 0.25.1 lane
(`patches/vllm-pr47356-vgx10/`).

## What & why

The serving image is built **in-house** as a thin Python layer on the official
`linux/arm64` `vllm/vllm-openai:v0.26.0` — no CUDA rebuild. The layer carries
`gx10-overlay-026.patch` (15 files, +1177/−64):

- **the `dspark-vllm-gx10` GB10 overlay** (Anemll, see NOTICE), cherry-picked onto v0.26.0
  with zero conflicts: `nvfp4_ds_mla` KV-quant plumbing, the b12x MXFP4 MoE backend, and
  the FlashInfer SM120/SM121 sparse-MLA bridge;
- **the warn-only `dspark_block_size` guard** — v0.26.0 hard-rejects
  `num_speculative_tokens=3 < dspark_block_size: 5` claiming garbled output; on this stack
  n=3 is measured garble-clean to 944K context (and n=5 costs −12% throughput), so the
  overlay logs and proceeds (upstream issue vllm-project/vllm#50012). The guard covers any
  n < 5 — prod runs **n=2** (2026-07-29 K re-tune, see docs/11);
- **the zero-token-prefill-chunk guard** — 0.26's capture/warmup batches (padded to
  `max_num_seqs`) emit zero-token prefill spans that crash flashinfer's sparse-MLA segment
  normalize (`cannot reshape tensor of 0 elements into shape [0, -1]`). The overlay skips
  empty chunks (upstream precedent: #48957). Every DSpark/0.26 deployment with this batch
  shape needs this guard;
- **backports #50004 + #49486** (the cand4 set, post-0.26.0 DSv4 perf): adaptive topk
  width + skipping topk/router when not needed. Measured here: they lift spec-decode
  acceptance from 0.592 (stock 0.26.0) to **0.713** and throughput +4.3% — a stock 0.26.0
  without them carries a −7% acceptance residual on this deployment, and reverting the
  release's own perf kernels (#48137/#48660) makes it monotonically *worse*, not better;
- **backports #48957 + #48047 + #50330** (the cand7 increment, selected from an 836-commit
  post-0.26.0 survey by on-path analysis — see
  [docs/13-vllm-026-cand7.md](../../docs/13-vllm-026-cand7.md)):
  **#48957** skips the empty c128 kernel launch on non-FULL cudagraph paths (prefill-side);
  **#48047** adds the q-head padding helper (alignment-only on this path — the overlay
  already carried the SM120 head ladder; the non-SM120 fallback class, dead on GB10, is
  also re-routed to the same upstream helper); **#50330** is a 4-line DSpark
  `get_draft_quant_config` override — a **correctness fix** for draft-quant handling.
  cand7 is branched from cand4 and deliberately does **not** carry the rejected cand6
  picks (#49731/#48407 — C8 −10.7% at bench) nor #47574 (boot-breaking on 0.26.0 without
  its companion #49903; the build smoke asserts its absence).

## Build (one node), then distribute

```bash
bash build-patched-image.sh    # tag: vllm-dspark-runtime:v026-gx10-cand7-backports

TAG=vllm-dspark-runtime:v026-gx10-cand7-backports
docker save "$TAG" | zstd -T0 -3 > ~/v026c7-image.tar.zst
rsync -a --partial ~/v026c7-image.tar.zst <worker-host>:~/
ssh <worker-host> 'zstd -dc ~/v026c7-image.tar.zst | docker load'
# assert parity — these MUST print the same ID:
docker image inspect --format '{{.Id}}' "$TAG"
ssh <worker-host> "docker image inspect --format '{{.Id}}' $TAG"
```

pip-install layers embed timestamps, so per-node builds are **not** byte-identical —
build once, distribute. The script: locates the base's vllm package dir, extracts the 14
upstream files the patch modifies, `git apply --check`s the patch (creating the 1
overlay-only file), builds, and smoke-verifies: version, both guards, all five backport
markers, the #47574 negative assert, `py_compile` of **every** overlaid file, and
b12x + flashinfer imports. It needs the official base pulled locally (see
`bringup/05-build-image.sh`).

**Cache roots:** a source-patch image must keep its own compile-cache roots
(`vllm-cache-cand7` / `triton-cache-cand7` / `tilelang-cand7`, wired in
`runtime/docker-compose.dspark.yml` + `runtime/cluster.env.example`) — vLLM's compile
cache keys on version+config, never on model source, so shared roots silently reuse stale
graphs and mask the patch. Expect one cold compile (~6 min) on first start with fresh
roots; warm them before benchmarking.

## Evidence

**cand7 gate (same-day arms vs cand4, 2026-08-10):** eval **100/100** warm · KV pool
byte-identical (2,948,751 tokens / 2.81× @1M) · acceptance flat (eval 0.665 vs 0.663,
bench 0.489 vs 0.490) · C1 33.84 vs 34.14 tok/s (−0.9%, Welch95 CI crosses zero — tie) ·
C8 86.81 vs 88.15 (−1.5%, Welch95 CI crosses zero — tie). → **throughput-neutral**: the
picks buy the #50330 correctness fix + #48047 upstream alignment at zero measured cost.
Promoted 2026-08-10; rollback = image `v026-gx10-cand4-backports` + the old prod cache
roots. Full record: [docs/13-vllm-026-cand7.md](../../docs/13-vllm-026-cand7.md).

**cand4 gate (vs the 0.25.1 lane, 2026-07-28):** C8 throughput **87.94 vs 84.29 tok/s
(+4.3%)** · eval acceptance **0.713/0.700 vs 0.638 (+12%)** · composite **100/100 ×2** ·
bench acceptance flat (0.43) · 3/3 needle retrievals @944,471 tokens · battery acceptance
0.784. The first 0.26 boot read a KV pool of 2,075,155 tokens at the unchanged 19.85 GiB
byte-pin — **not** a layout change (bytes/blocks byte-identical): 0.26's #47728 doubles
the sliding-window admission reservation under `--async-scheduling`. Resolved same-day by
serving with `--no-async-scheduling` (throughput-neutral, pool back to 2,948,751 / 2.81× —
see `docs/10-vllm-026-rebase.md` "Open items"). Full record in `docs/10-vllm-026-rebase.md`.
