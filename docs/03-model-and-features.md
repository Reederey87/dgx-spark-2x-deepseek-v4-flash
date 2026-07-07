# The model stack and its four advanced features

This is the accuracy-critical reference for **what the model stack actually does** and
**where each advanced feature is turned on** in this repo. Every knob named below is
quoted from [`docker-compose.dspark.yml`](../docker-compose.dspark.yml) and
[`cluster.env.example`](../cluster.env.example) — open those alongside this page and you
can trace every claim to a line.

## What this serves

**`deepseek-ai/DeepSeek-V4-Flash-DSpark`** — a **284B-parameter MoE** model with **~13B
active** parameters per token, **text-only**, served **TP=2** across two DGX Spark (GB10 /
`sm_121a`) nodes over the QSFP fabric. Native context is **1M tokens** (the YaRN ceiling is
`65536 × 16 = 1,048,576`, which is the default `MAX_MODEL_LEN`).

The serving substrate — the things every feature below sits on top of — is:

- **KV cache in `nvfp4_ds_mla`** (`--kv-cache-dtype nvfp4_ds_mla --block-size 256`). ⚠️
  **Read this once and remember it:** "NVFP4" here labels the **KV-cache dtype**, not the
  weights. The weights are FP8/MXFP4. Do not read "NVFP4" anywhere in this stack as "4-bit
  weights."
- **DSpark speculative decoding** (multi-token prediction / MTP):
  `--speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'`,
  with the draft length driven by `MTP_NUM_TOKENS` (default `3`).
- **The B12X MoE path** (`VLLM_USE_B12X_MOE=1`) — the fast MoE kernel; the DeepGEMM MoE path
  is the *slow* fallback (see [B12X note](#the-b12x-moe-path-the-single-biggest-speed-lever)).

> ⚠️ **Maturity disclaimer — read before trusting any of this.** All four features below,
> and the image they ship in, trace to a small number of community authors and to
> **prebuilt, non-source-buildable GB10 kernels**. This is experimental, fast-moving work.
> This repo does **not** vendor their source — it *builds from* a pinned recipe and *pulls*
> the weights at deploy time. **Validate on your own hardware behind your own smoke tests.**
> See [`CREDITS.md`](../CREDITS.md) for the full provenance chain.

## Where the features come from (provenance in one breath)

All four named features trace to the community author **"Keys" (GitHub
[`drowzeys`](https://github.com/drowzeys))** and to the **tonyd2wild** Tier-B serving recipe
this repo pins (`RECIPE_SHA` in `cluster.env` → the `vllm-dspark-runtime:dspark-nvfp4-stage-c`
image), which in turn wraps **aidendle94**'s compiled GB10 (`sm_121a`) kernels. **They are
already baked into the stage-c image.** This repo's job is to *enable* them via env/flags and
*document* them — not to reimplement them.

## Summary

| Feature | What it does | Enabled by (env / flag) | Maturity | Biggest risk |
|---|---|---|---|---|
| **1. Keys scalable-concurrency patch** | Lets DSpark speculative decoding serve real concurrent streams instead of serializing to one | `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` + `--max-num-seqs` > 1 (`MAX_NUM_SEQS`) | Validated on this model; ragged steps run eager | Without the mask=1 ragged path, concurrent draft-KV can be misattributed → acceptance collapse |
| **2. "dual cache"** | Split-K scratch **sizing** fix in the sparse-MLA kernel so it's correct when SWA + global top-k index sets are attended together | Correctness fix compiled into the stage-c kernel (0.24.0 port gates it on `VLLM_USE_FLASHINFER_SPARSE_MLA=1`, **not present** in this compose) | Fix is sound; the 0.24.0 drop-in caps acceptance ~40%, so production uses the compiled kernel | Misreading the name as a two-tier/offload cache (it is not) |
| **3. `sm121a_DSA` with DeepGEMM enabled** | Runs the DeepSeek Sparse Attention indexer on GB10 with a genuine GB10 DeepGEMM kernel | `DG_JIT_USE_NVRTC`, `DG_JIT_NVCC_COMPILER`, `TORCH_CUDA_ARCH_LIST=12.1a`, `FLASHINFER_CUDA_ARCH_LIST=12.1a` | High-risk / experimental; transplanted `.so`, no public source | DeepGEMM is wanted for the **indexer**, *not* MoE — MoE's fast path is B12X; needs a CUDA-13.0 image |
| **4. `sm121a_sparse_MLA` enabled** | The top-k-indexed MLA that DSA drives, running on GB10 where native backends are absent | `VLLM_TRITON_MLA_SPARSE=1`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256`; leave `--attention-backend` on AUTO | Compiled flashinfer path ships in stage-c; Triton fallbacks are portable but 2–8× slower | Default Triton decode is **wrong** for `compress_ratio≥4` layers unless `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` |

---

## Feature 1 — Keys scalable-concurrency patch

**What it does.** Stock DSpark speculative decoding effectively pins `--max-num-seqs 1`: it
serializes, so you can't serve concurrent streams under continuous batching. The patch removes
that serialization so speculative decoding works with real concurrency.

**The bug it fixes.** Stock DSpark keyed its persistent per-request draft KV cache by
**batch-row position**. That is unstable under vLLM-v1, which *compacts the running set* as
requests finish — a finished row gets reused by a new request, so the draft KV silently belongs
to the wrong request → **acceptance collapse or output corruption**. The patch keys draft KV by
a stable **req-id → slot** map, and adds a **ragged-context path** (per-request
`query_start_loc` offsets instead of a single rectangular reshape).

**Source.** <https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash>
(Apache-2.0).

**How it's enabled here.** The ragged path is **only implemented for the GPU rejected-context
mask mode**, so the hard requirement is:

```yaml
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK: "1"   # present in docker-compose.dspark.yml
```

…then set your target concurrency with `--max-num-seqs` (driven by `MAX_NUM_SEQS`, default `12`
in `cluster.env`, `6` as the compose fallback). With the mask at `1` and `--max-num-seqs 1` you
get the old serialized behavior; raise it to fan out.

**Maturity.** Validated **on DeepSeek-V4-Flash-DSpark specifically**: byte-identical
single-stream output under churn, quality-neutral on GSM8K / MATH / HumanEval. Caveat: **ragged
/ mixed steps run eager** (not cudagraph-captured), so they're correct but not captured-fast.

**NVFP4 + MTP compatibility.** Designed for exactly this substrate — compatible with
`nvfp4_ds_mla` KV and MTP speculative decoding.

**Related guard.** This patch is what makes the draft-KV slot mapping request-stable; the
`DSPARK_SLOT_CLAMP=1` guard in [`docs/LONG_CONTEXT_CRASH_FIX.md`](./LONG_CONTEXT_CRASH_FIX.md)
is the belt-and-suspenders pair that clamps any stale slot id that still survives into a
long-context step. Keep both on.

## Feature 2 — "dual cache"

**What it does — and what it is NOT.** Despite the name, this is **not** a two-tier or
offloading KV hierarchy. It is a **split-K scratch-sizing correctness fix** in the sparse-MLA
kernel. DeepSeek-V4 sparse MLA attends over **two index sets at once** — the sliding-window
(SWA) set **and** the global top-k set — so the split-K scratch buffer must be sized to the
**sum** of each set's tiles, not the max:

```
nsplit = cdiv(prim_topk, 64) + cdiv(extra_topk, 64)
```

For the single-cache case (`extra_topk == 0`) this collapses back to the reference
`cdiv(width, 64)`, so it's a strict generalization. It also ships a paired `out_lse`
contiguity fix.

**Source.** drowzeys `keys-vLLm-0.24.0` repo:
`vllm-0.24-port/flashinfer_sparse_mla.py` + `PORT_NOTES.md`
(<https://github.com/drowzeys/keys-vLLm-0.24.0-Optimized-DeepSeekV4-Flash-DSpark-NVFP4-KV-1.5M-CTX-3M-Pool-C-12-on-2-DGX-Spark>).

**How it's enabled here.** In the 0.24.0 drop-in, this path is gated by
`VLLM_USE_FLASHINFER_SPARSE_MLA=1` — **that variable is not in this repo's compose.** In the
0.24.0 drop-in the draft path caps acceptance at ~40%, so the **production path (this repo's
stage-c image) uses aidendle94's compiled kernel instead**. Read "dual cache" as **the
correctness fix that makes the compiled verify kernel produce right answers at real batch sizes
on GB10** — it's baked into the kernel, not a runtime toggle you flip here.

**Maturity.** The sizing fix itself is sound and simple; the concern is the 0.24.0 drop-in's
acceptance cap, which is why production doesn't use that path.

**NVFP4 + MTP compatibility.** It's the sparse-MLA correctness fix underneath the whole NVFP4-KV
+ MTP path, so yes — it's a prerequisite for them behaving at batch size > 1.

## Feature 3 — custom `sm121a_DSA` with DeepGEMM enabled

**What it does.** **DSA = DeepSeek Sparse Attention** — token-level sparse attention plus a
"Lightning Indexer," introduced in V3.2 and carried into V4. The problem on GB10: the DSA
indexer **hardcodes DeepGEMM calls**, and upstream DeepGEMM only targets **SM90 (Hopper)** — it
hard-asserts `sm_90a`. So "`sm121a_DSA` with DeepGEMM enabled" means **getting the DSA indexer
to run on GB10 against an actual GB10 DeepGEMM kernel**: aidendle94 compiled a DeepGEMM
`sm_121a` `.so` (a genuine GB10 build) and transplanted it into the stage-c image.

**How it's enabled here.** The relevant env in the compose:

```yaml
DG_JIT_USE_NVRTC: "0"
DG_JIT_NVCC_COMPILER: "/opt/env/bin/nvcc"       # DeepGEMM JIT toolchain
TORCH_CUDA_ARCH_LIST: "12.1a"                    # the 'a' suffix turns on hardware FP4
FLASHINFER_CUDA_ARCH_LIST: "12.1a"
```

The `a` / `f` arch suffix on `12.1a` is what enables the hardware FP4 path.

**Important nuance — DeepGEMM is for the INDEXER, not MoE.** "DeepGEMM enabled" is desirable for
the **DSA indexer**. For **MoE**, the fast path is **B12X** — DeepGEMM-MoE is the *slow*
fallback, and `VLLM_USE_B12X_MOE=0` silently falls back and **tanks decode**. Don't conflate the
two.

**Provenance / alternatives.** Many other recipes instead **bypass DeepGEMM on GB10** with
Triton fallbacks (hazyumps, CosmicRaisins); upstream vLLM PR **#38476** adds a
`TRITON_MLA_SPARSE` backend. This repo takes the native-DeepGEMM route via the transplanted
`.so`.

**Maturity — high-risk / experimental.** The native DeepGEMM `sm121a` kernel is a **transplanted
`.so`, not public source**. It requires a **CUDA-13.0-based image** — driver `580.159.03` is
CUDA 13.0, and a `_C` built for CUDA 13.2 **won't JIT**, which is a documented wall.

**NVFP4 + MTP compatibility.** Compatible — the `12.1a` FP4 arch is exactly what the NVFP4-KV
path and the sparse attention chain are built against.

## Feature 4 — `sm121a_sparse_MLA` enabled

**What it does.** **MLA = Multi-head Latent Attention**; **sparse-MLA** is the top-k-indexed MLA
that the DSA indexer drives. On GB10 the native backends are simply **unavailable**:
`FLASHMLA_SPARSE` reports "compute capability not supported," and DeepSeek's own FlashMLA only
ships SM90 / SM100 kernels. Three flavors exist in the wild:

1. **Compiled flashinfer sparse-MLA (aidendle94)** — `sparse_mla_sm120_decode_dsv4`, the
   fastest, and **the one that ships in the stage-c image**. Conceptually gated by the
   sparse-MLA path rather than a single on/off flag.
2. **Portable Triton sparse-MLA (CosmicRaisins)** — public but **~2–8× slower**.
3. **Upstream `TRITON_MLA_SPARSE`** — vLLM PRs **#38476 / #47629**.

**How it's enabled here.** The compose exposes:

```yaml
VLLM_TRITON_MLA_SPARSE: "1"
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: "256"
```

…and deliberately **leaves `--attention-backend` on AUTO** on the stage-c image. Do **not** pass
`--attention-backend FLASHINFER_MLA_SPARSE_DSV4` — it throws "unknown backend."

> ⚠️ **Correctness trap.** On SM12x the **default Triton decode kernel produces WRONG output**
> for layers with `compress_ratio ≥ 4`. The known-good path needs
> **`VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0`**. (This repo's stage-c image relies on the compiled
> flashinfer path plus the image's own defaults rather than exposing this as a compose knob —
> if you swap to a Triton-decode path, this is the first thing to set.)

**Cache envelope.** The stage-c NVFP4 path deliberately keeps DeepSeek-V4's known-good
**584-byte padded** sparse-MLA cache envelope. It does **not** use the unresolved "true-layout
416-byte NVFP4 kernel," which failed past ~411 real tokens.

**Maturity.** The compiled path is fast but closed-source; the Triton alternatives are auditable
but slow; the "true-layout" variant is a known dead end past a few hundred tokens.

**NVFP4 + MTP compatibility.** This is the attention path underneath `nvfp4_ds_mla` KV, and it's
what the DSA indexer + MTP draft/verify loop run against.

---

## The garble fix (2026-07-03) — do not revert

Cold-start **tool-call / Chinese-character garble under concurrency** was traced to a **greedy**
draft of length 5. The fix, which is the current default, is:

- `MTP_NUM_TOKENS=3` (down from a greedy 5), and
- **probabilistic** draft (`"draft_sample_method":"probabilistic"` in the speculative config), and
- the **FlashInfer sampler**: `VLLM_USE_FLASHINFER_SAMPLER=1`.

**Do NOT revert to greedy 5.** This narrows the same mismatch window that `DSPARK_SLOT_CLAMP`
guards — see [`docs/LONG_CONTEXT_CRASH_FIX.md`](./LONG_CONTEXT_CRASH_FIX.md). If you still see
garbled output, walk the ladder in [`docs/05-troubleshooting.md`](./05-troubleshooting.md).

## The B12X MoE path — the single biggest speed lever

`VLLM_USE_B12X_MOE=1` selects the fast MoE kernel and is the **single biggest speed lever** in
the whole stack. The DeepGEMM MoE path is the **slow fallback**; setting `VLLM_USE_B12X_MOE=0`
silently falls back and tanks decode. Leave it at `1`.

## Do NOT set `VLLM_DSV4_B12X_COMPRESSED_MLA=1`

The compose ships `VLLM_DSV4_B12X_COMPRESSED_MLA=0` on purpose. Setting it to `1` **wedged
unified memory** in community testing. Leave it at `0`.

## A driver caveat worth knowing

Driver **`580.159.03`** has a reported **~3.5× decode regression vs `580.142`** on GB10. It's
the CUDA-13.0 driver the DeepGEMM `sm121a` path requires (Feature 3), so this is a genuine
tradeoff, not a free upgrade — if you're chasing decode throughput, this is a known variable.

---

## Provenance, alternatives, and upstream convergence

This repo **pins the currently-proven tonyd2wild recipe + `dspark-nvfp4-stage-c` image on
purpose** (see [`CREDITS.md`](../CREDITS.md)). Known alternatives that this kit does **not**
ship:

- **drowzeys' vLLM 0.24.0 port** — `git-apply` patches rather than a prebuilt image.
- **r0b0tlab's native-DeepGEMM benchmark image** — a different prebuilt.

**Watch upstream convergence.** The fork/patch/transplant stack documented here is a snapshot of
a moving target:

- **DSpark merged into vLLM** in PR **#46995**.
- **Sparse-MLA-on-SM12x is going mainline** via PRs **#38476 / #47629** (plus **#43477** for the
  FP8-KV path).

A rebase onto **vLLM ≥ 0.24.1** could retire much of this fork/patch/transplant stack. **Track
those PRs before deep-pinning any one prebuilt image** — the right long-term move is to follow
this upstream rather than to over-invest in the current stage-c image.
