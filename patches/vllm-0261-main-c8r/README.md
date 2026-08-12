# vLLM main @48bada6ea4 gx10-overlay kit (the c8r production lane — 0.27 content)

**Status: this is the current production image recipe (promoted 2026-08-11).** It supersedes
the 0.26.0 cand7 lane (`patches/vllm-026-rebase/`, promoted 2026-08-10), which is kept as the
first image rollback rung. The full receipt — selection, gate, and the cold-cache regression
lesson — is [docs/14](../../docs/14-vllm-027-c8r.md).

## What & why

Unlike every prior lane, there is **no official arm64 image** for this base: the image is a
**full-source build** of upstream vLLM `main` at the pinned commit
`48bada6ea49ad7f3ecbe03128aa76562089c8b00` — **334 commits past the v0.27.0 release cut**
(release branched at `f5bb701fa2`, an ancestor of the pin; re-pinning to the v0.27.1 tag was
rejected as a content downgrade — its branch-only commits are CI/docs/other-models plus a
FlashInfer bump main already had). Stage-1 is upstream's own `docker/Dockerfile`
(`--target vllm-openai`, `torch_cuda_arch_list=12.1a`, aarch64 manylinux builder); the runtime
layer (`Dockerfile.runtime-0261`) then applies the overlay and pins the vendored bits. All
dependency pins come from main's own `requirements/` at the SHA — see
[upstream-0261.lock](upstream-0261.lock).

On top of the pinned tree, `overlay0261/vllm/` carries **17 whole-file overlays** — 14 from
the port (13 three-way merges `git merge-file <main> <v0.26.0> <c7-overlay>` + the DeepGEMM
SF-layout hotfix), each documented decision-by-decision in [PORT-NOTES.md](PORT-NOTES.md),
plus the 3 files of the #49731 revert (documented in [docs/14](../../docs/14-vllm-027-c8r.md),
not PORT-NOTES):

- **the `dspark-vllm-gx10` GB10 overlay** (Anemll, see NOTICE), forward-ported from the
  0.26.0 lane: `nvfp4_ds_mla` KV-quant plumbing, the b12x MXFP4 MoE backend, the FlashInfer
  SM120/SM121 sparse-MLA bridge;
- **the warn-only `dspark_block_size` guard** — main hard-rejects
  `num_speculative_tokens=2 < dspark_block_size: 5` (#49969); on this stack n=2/n=3 are
  measured garble-clean to 944K context, so the overlay logs and proceeds (upstream issue
  vllm-project/vllm#50012);
- **the SM120/121 DeepGEMM scale-factor layout torch port** (`utils/deep_gemm.py`) — boot
  dies without it (`Unknown SF transformation`); bit-exactness vs the C++ binding is
  provable with [verify-sf-layout.py](verify-sf-layout.py);
- **the #49731 revert** — the upstream "replicate the DSpark Markov draft head across TP
  ranks" change, reverted back to the standard TP-sharded path. **Measured perf-neutral**
  (revert arm vs no-revert arm: +0.62% C8, Welch tie); kept because sharded is the standard
  memory path (no per-rank replica) and matches every other weight. The revert is the 3
  files `qwen3_dspark.py` / `vocab_parallel_embedding.py` / `logits_processor.py`, with the
  later #50424 quant plumbing preserved.

What the base buys this deployment, measured on the gate (vs the 0.26.0 cand7 lane, same-day
arms): the **MXFP4 indexer KV-cache packing (#48993)** — the pinned KV pool grew
**2,948,751 → 3,027,217 tokens (+2.7%)** at the *unchanged* byte pin — plus the native
sparse-MLA/adaptive-topk/workspace-reuse work, at **throughput parity** (warm gate:
C1 −1.14% TIE, C8 +0.11% TIE).

## Build

Runs from the control host; stage-1 is a **multi-hour** nvcc build on the head node, and the
cluster must be stopped (nvcc does not fit next to the serving pool on the 121 GiB UMA):

```bash
git clone https://github.com/vllm-project/vllm.git ~/vllm   # one-time, control host
git -C ~/vllm worktree add ~/vllm-0261-main-wt 48bada6ea4   # one-time; script never creates refs
bash build-0261-image.sh --distribute      # stage-1 + runtime layer + smoke + worker push + parity assert
# overlay-only iteration (hours → minutes): reuse the existing stage-1:
bash build-0261-image.sh --runtime-only --distribute
```

The post-build smoke is assertion-heavy by design: overlay markers, the #49731 revert
markers, the DeepGEMM sm120 include-tree pin, the FlashInfer native pin (0.6.16.post3), and
`py_compile` of every overlaid file — see the script header for the full list.

## Cache-root rule (full-source lane)

A full-source build has a **different version string** from any prior lane, and the compile
cache keys on version+config — so this image keeps its **own cache-root set**
(`vllm-cache-c8r` / `triton-cache-c8r` / `tilelang-c8r`). First boot on a fresh root is a
full cold compile (~12 min) and **the first bench reps run slow** — warm before gating, or
you will measure a phantom regression (the exact mistake docs/14 dissects).

## Rollback

Image `vllm-dspark-runtime:v026-gx10-cand7-backports` + the `-cand7` cache roots (the
`patches/vllm-026-rebase/` lane), then cand4, then `vgx10-011-pr47356` (0.25.1). Rollback
images stay resident on both nodes in the reference deployment.
