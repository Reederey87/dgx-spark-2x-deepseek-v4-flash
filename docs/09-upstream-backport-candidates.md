# Upstream backport candidates for `anemll/dspark-vllm-gx10` — 2026-07-17

**What this is.** An evidence-backed list of post-v0.25.1 upstream vLLM fixes that this kit's
default image (`vllm-dspark-runtime:vgx10-cand1-pr47356`, vLLM `0.25.2.dev0+g752a3a504.d20260714`,
pinned at upstream tag `v0.25.1` = commit `752a3a50`) does **not** carry — verified by grepping
the *live container's* installed package (`/usr/local/lib/python3.12/dist-packages/vllm` on the
head node). Candidates were harvested from the ~347 commits on upstream `main` past the pin's
branch point (merge-base `6db31c8e7`). Purpose: a precise maintainer ask — each item lists the
exact in-container evidence of absence.

**Deployment context.** 2× DGX Spark (GB10, SM121, TP=2 over RoCE), DeepSeek-V4-Flash-DSpark,
`nvfp4_ds_mla` KV, DSpark MTP=3 probabilistic, 1M ctx, 12 seqs, KV pinned via
`--kv-cache-memory-bytes`. All correctness gates green; concurrency-8 throughput ~84.55 tok/s
(~−7.8% vs an older 0.21.1-based image — residual gap under investigation).

## Confirmed ABSENT in the image (probed 2026-07-17)

| # | Upstream commit | What it fixes/improves | In-container evidence of absence | Priority |
|---|---|---|---|---|
| 1 | `442c421e7` **#48137** — [Perf] Remove redundant repeat and copy for dsv4, **1.8% E2E TPOT** | DeepSeek-V4 decode perf via a new fused mhc tilelang kernel (`mhc_pre_big_fuse_broadcast_with_norm_tilelang`) + `deepseek_v4/nvidia/model.py` changes | `model_executor/kernels/mhc/tilelang_kernels.py` exists but **does not contain** `mhc_pre_big_fuse_broadcast_with_norm_tilelang` | **P1 — direct perf win on exactly this model/config** |
| 2 | `95d6d6f4b` **#48046** — Use int8 workspace for FlashInfer MLA decode | Correctness/compat of the FlashInfer MLA-decode workspace (`uint8`→`int8`; the CuteDSL tactic requires int8) | `v1/attention/backends/mla/flashinfer_mla_sparse.py:360` still `dtype=torch.uint8` | **P2 — touches this deployment's exact attention backend file** |
| 3 | `1be6e937b` **#48483** — Lower cudagraph-capture memory for large capture sizes | Frees memory at capture (min-blocks fix, 5 lines, `v1/worker/gpu_model_runner.py`) | "minimum number of blocks required is 1 block" comment **absent** | P2 — headroom win, tiny patch |
| 4 | `ecf4aa5ce` **#48167** — Fix FlashInfer non-causal draft attention (DFlash/DSpark) on Blackwell | DSpark/DFlash draft-attention causality on Blackwell | `v1/worker/gpu/spec_decode/dspark/speculator.py:62` still has the pre-fix `self.dflash_causal = False` | **P3 — correctness-class, but this deployment is symptom-free** (acceptance 0.42–0.44, gates green; likely not exercised on the b12x draft path — awareness, not urgency) |
| 5 | `647213129` **#48379** — Set `kv_quant_mode` on the generic MLA KV-cache spec | 2-line KV-quant mode propagation fix (`mla_attention.py`) | `kv_quant_mode=get_kv_quant_mode` **absent** | P3 — small; relevance depends on whether the DSv4 custom spec path shares it |

## Absent by construction (post-pin; not individually probed — V2-runner spec-decode fixes)

- `85b3a7264` **#47381** — Model Runner V2: order uniform decodes first (spec decodes
  misclassified as prefills). The V2 runner **is present** in the image, so this applies.
- `26587f951` **#48261** — ModelRunner V2: stale attn metadata in speculator prefill cudagraph
  capture.
- `8bfd68390` **#48787** — `kv_cache_dtype` in `speculative_config` (separate draft-KV dtype) —
  confirmed absent (grep); a future lever, not a fix.

## Explicitly NOT requested (triaged out)

- ROCm/XPU DeepSeek-V4 items (#47718, #46275, #48519, #47677) — wrong arch.
- `80eb01e93` #47493 — DeepSeek-V4 TP16 garbage output; this deployment runs TP=2.
- Upstream DSpark PR #46965 itself — still has NVIDIA-backend `hc_pre`/`hc_post_pre` gaps per its
  own thread; the fork's DSpark is ahead of it for this deployment. (Confirmed by probe: the image
  runs the **upstream V2-runner DSpark** at `v1/worker/gpu/spec_decode/dspark/`, not the older
  community overlay.)
- Unmerged branches `fix-mtp`, `fix-mtp-dummy-run-assertion` — target the vanilla eagle/MTP path,
  not DSpark; no observed symptom; unmerged into both the pin and `main` as of 2026-07-17.

## Draft maintainer message (copy-paste)

> Hi anemll — running your `dspark-vllm-gx10` 0.1.0 image (vLLM 0.25.1 @ 752a3a50) in production
> on 2× DGX Spark with DeepSeek-V4-Flash-DSpark (nvfp4_ds_mla, DSpark MTP=3, 1M ctx). Great work
> on the port. While A/B-testing we verified the following post-0.25.1 upstream fixes are absent
> from the image (grepped the installed package in the live container). Any chance the next build
> could pick them up?
>
> 1. **#48137** (`442c421e7`) — 1.8% E2E TPOT for DeepSeek-V4 (fused mhc tilelang kernel). Our top ask.
> 2. **#48046** (`95d6d6f4b`) — int8 workspace for FlashInfer MLA decode (your
>    `flashinfer_mla_sparse.py:360` still has `uint8`).
> 3. **#48483** (`1be6e937b`) — lower cudagraph-capture memory (5 lines).
> 4. **#48167** (`ecf4aa5ce`) — FlashInfer non-causal draft attention on Blackwell
>    (your `dspark/speculator.py:62` still has `self.dflash_causal = False`). We're symptom-free
>    but it's a correctness-class fix on our exact stack.
> 5. **#48379** (`647213129`) — `kv_quant_mode` on the generic MLA KV-cache spec (2 lines).
> 6. V2-runner spec-decode fixes #47381 / #48261 if the build uses the V2 runner.
>
> Happy to A/B any candidate build through our gate suite (correctness composite, concurrency-8
> throughput, KV saturation) and report numbers back.

## Standing rule

Nothing here is a same-image config knob — do not hot-patch the live container. These land via a
maintainer rebuild (or feed the build-vs-buy decision of maintaining an own image). Every image
change still goes through this kit's standard gate list (correctness composite → concurrency-8
throughput vs the 84.55 tok/s baseline → head-of-line probe → KV saturation → restart/watchdog
sanity) before becoming the default.
