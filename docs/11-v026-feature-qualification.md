# vLLM 0.26.0 feature qualification (2026-07-29)

A 12-row audit of the v0.26.0 release against the live 2× GB10 lane
(`vllm-dspark-runtime:v026-gx10-cand4-backports` — superseded by **cand7** on 2026-08-10,
throughput-neutral; see [13](13-vllm-026-cand7.md)), each row closed with source, runtime, or
A/B evidence. Two production changes came out of it (below); everything else was verified,
rejected, or filed as a follow-up.

## Production changes

### 1. DSpark speculative draft length: n=3 → **n=2**

Same-day A/B lanes on the 0.26.0 runtime (one boot per lane, 3 batches each, n=2 confirmed
with a second boot = 6 batches):

| K | C1 (single-stream) tok/s | C8 (concurrency 8) tok/s | bench acceptance |
|---|---|---|---|
| **2** | **35.42** | 91.63 | 0.554 / 0.533 |
| 3 (prior default) | 34.13 | 90.84 | 0.411 / 0.427 |
| 4 | 32.25 | 82.43 | 0.328 / 0.341 |

n=4 loses both axes (consistent with the earlier n=5 history). n=2 vs n=3: **+3.8%
single-stream**, concurrency-8 a tie within batch noise. The primary workload on this lane is
single-stream interactive decode, so n=2 wins. Acceptance rises at shorter drafts exactly as
rejection sampling predicts (eval workload: **0.795–0.806** vs 0.703–0.731). Set via
`MTP_NUM_TOKENS=2`; `MAX_CUDAGRAPH_CAPTURE_SIZE` follows the formula (`12 × (2+1) × 2 = 72`).
Post-promotion gates: eval composite **100/100 ×3**, deep-context needles **3/3 HIT @
944,471 tokens**, battery acceptance **0.842**. Rollback: `MTP_NUM_TOKENS=3`.

### 2. Explicit attention-backend pin

`--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4"}'` is now in the compose argv.
On SM12x this resolves to `DeepseekV4FlashInferSM120Attention` — the **identical** class the
model-driven default already picks (`vllm/models/deepseek_v4/nvidia/model.py:779-782` vs
`:789-790`), so it changes nothing today; it hard-fails loudly if a future image ever changes
the SM12 default. Validated at the promotion boot (engine logs the DSV4 backend enum; the
`nvfp4_ds_mla` KV layout is admitted as before).

## Row-by-row verdicts

| Area | Verdict |
|---|---|
| DeepSeek V4 sparse-MLA route (FLASHINFER_MLA_SPARSE_DSV4) | **Active** — auto-selected on SM12x; now also pinned explicitly (above). `nvfp4_ds_mla` is admitted via the gx10 overlay's layout plumbing, not the stock backend list |
| DSV4 specialized routing kernel (2.94% E2E TPOT, #48660) | **Active-proven** — in-image file identical to the v0.26.0 tag (sha256); auto-selected (256 experts, topk 6, sqrtsoftplus, fp32 router logits); no fallback |
| DSV4 fused topk bias (#47463) | **Active-proven** — in-image file identical to the tag; no old overlaid filter survives |
| Removed DSV4 repeat copy (#48137) | **Active-proven** — in-image files identical to the tag; its TileLang `mhc` kernels are live at runtime. Reverting it on 0.26 measurably hurts acceptance (2026-07-28) — keep |
| Blackwell B12X MoE vs 0.26 modular backend | **Resolved, no change** — the effective backend is `DEEPGEMM_MXFP4` (`DeepGemmFP4Experts`), auto-selected by the overlay's MXFP4 oracle after TRTLLM is filtered on SM121. `VLLM_USE_B12X_MOE=1` is set-but-inert on 0.26 (defined, never consumed). B12X was already A/B'd away on the 0.25.1 lane (flat-to-down) |
| Lower NVFP4 MoE startup peak (#46276) | **Active-by-default** — in-image file identical to the tag; boot memory telemetry clean |
| FlashInfer/CUTLASS dependency lock | **Vendor FlashInfer pin REQUIRED** — see below. cutlass-dsl 4.6.0, tvm-ffi 0.1.10, torch 2.11.0 all exactly on the release lock |
| Per-KV-group attention selection (#48012) | **Ineligible** — DSv4's backend selection is model-driven and never consults `backend_per_kind` |
| FlashInfer autotune policy | **Active** (`--enable-flashinfer-autotune`, DSv4 decode autotune cache hit on both TP ranks) — with a follow-up: the warmup tunes only `num_tokens=16`, so other capture/runtime shapes fall back to the default tactic (`perf cliff` warnings). Widening the tuned bucket set is a queued improvement |
| DSpark MTP/speculative decoding | **Re-tuned: n=2** (above) |
| Expert parallel / new all-to-all paths | **Ineligible** — B12X experts forbid EP; DeepEP targets SM90/SM100; the lane runs TP-sharded experts; EP previously measured −14.8% here |
| CUDA graphs / compile | **Resolved, no change** — breakable CUDA graphs (FULL + PIECEWISE capture incl. the speculator); the only inference-time JIT warnings are one-time, shape-specialized compiles at first traffic, now cached |

## Dependency manifest and the FlashInfer pin (row 7)

In-image vs the v0.26.0 release lock:

| Package | Release lock | In image | Note |
|---|---|---|---|
| torch | 2.11.0 | 2.11.0+cu130 | on lock |
| flashinfer-python | 0.6.14 | **0.6.15-dev @0472b9b3** | vendor pin — **required, proven** |
| flashinfer-cubin / jit-cache | 0.6.14 | 0.6.14 (+cu130) | on lock |
| nvidia-cutlass-dsl | 4.6.0 | 4.6.0 | on lock |
| apache-tvm-ffi | 0.1.10 | 0.1.10 | on lock |
| b12x | — (no lock entry) | 0.15.3 @7dc6fb8f | vendor-only; only used by B12X_MXFP4 diagnostic lanes |

The pin was tested, not inherited: a candidate image identical to prod except FlashInfer at
the stock 0.6.14 wheel **booted clean but crashed on the first decode traffic** —
`flashinfer/mla/_core.py:192`, `_normalize_sparse_mla_indices_and_lens` reshaping an empty
sparse-index batch (`RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1]`),
hanging the head into its 600 s execute-model RPC timeout. The DSpark sparse-MLA path emits
empty index batches that 0.6.14 cannot normalize; `@0472b9b3` carries the fix (the normalize
functions are byte-identical between the refs — the fix is elsewhere in the delta; the
candidate image is retained as the bisect repro if unpinning is ever revisited).

## Evidence pointers

- Bench JSONs: `bench/results/{s10prod,n2probe,n4probe}__*` in the private ops repo.
- Boot markers (promotion boot): DSV4 backend enum logged, `num_speculative_tokens: 2`,
  derived `max_cudagraph_capture_size: 72`, KV pool **2,948,751** unchanged, no fallback warnings.
