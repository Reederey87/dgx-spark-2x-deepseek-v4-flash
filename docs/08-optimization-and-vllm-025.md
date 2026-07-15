# Optimization ledger and the vLLM 0.25 promotion

This page records the experiments behind the shipped profile. As of **2026-07-15 it documents a
lane change**: the default serving image moved from the vLLM 0.21.x recipe to a vLLM **0.25.1**
base. Numbers are directional measurements from one 2× DGX Spark pair; re-run the gates on your own
hardware and model revision. This is a ledger, not a spec sheet — it keeps the losers and the open
questions next to the wins on purpose.

## Two different vLLM 0.25 candidates — do not conflate them

This file previously concluded that "vLLM 0.25 is not ready to ship." That conclusion was about a
**different candidate** than the one that shipped. The distinction matters for anyone trying to
reproduce this, so state it plainly:

| | **`v0250-cand4`** (rejected) | **`dspark-vllm-gx10`** (shipped 2026-07-15) |
|---|---|---|
| What it was | A vLLM **0.25.0 rebase of the same tonyd2wild/drowzeys recipe lineage** the 0.21.x lane already used | A **separately-maintained community GB10 port** by GitHub user `anemll`, built from a different lineage entirely, pinning vLLM **0.25.1** |
| Verdict | Not promoted: −9.2% concurrency-8 throughput, plus a follow-up ladder that never closed the gap | **Promoted to default** after a whole-image correctness gate + a hardening pass |

Today's promotion is **not** "the `v0250-cand4` problems got fixed." It is a different, independent
0.25.1 port that was evaluated, hardened, and promoted **instead**. The rejected `v0250-cand4`
findings are preserved below (see *The earlier `v0250-cand4` rejection*) unchanged — they are still
true about that candidate.

## Current production profile — vLLM 0.25.1 (`dspark-vllm-gx10`)

The supported default lane is the digest-pinned `anemll/dspark-vllm-gx10` base plus one thin
upstream-fix layer (vLLM PR #47356). Pins:

| Component | Pin |
|---|---|
| Base image | `ghcr.io/anemll/dspark-vllm-gx10@sha256:d0a0c050252dd0b64a3213c728d1c15db9eb37602593ba37534bc708dd223ae7` (tag `0.1.0`) |
| vLLM | `v0.25.1` @ `752a3a504485790a2e8491cacbb35c137339ad34` (self-reports `0.25.2.dev0+g752a3a504.d20260714`) |
| FlashInfer | `0.6.15-dev` @ `0472b9b3f2fba11b463f8526f390297d52a8aad7` (unreleased dev commit; `FLASHINFER_DISABLE_VERSION_CHECK=1` suppresses the version-check mismatch) |
| b12x (MXFP4 MoE) | `0.15.3` @ `7dc6fb8fcc6446ea093537d1657df81985fa5f43` |
| Patched serving tag | `vllm-dspark-runtime:vgx10-cand1-pr47356`, image ID `sha256:99b05acd7e23c447e549122ed8a932d6ca959aa8671c32b6ee7f85f821222a22` (byte-identical across both nodes) |

The serve knobs that are *unchanged* from the prior lane still hold: context 1,048,576 (calibrated
YaRN ceiling), MTP draft length 3 (probabilistic), max sequences 12, batch-token cap 8,192,
long-prefill threshold 4,096, `nvfp4_ds_mla` KV, and `GPU_MEMORY_UTILIZATION=0.85` for a dedicated
pair. The single applied patch and the four config deltas below are what is *new*.

### The one applied patch: vLLM PR #47356

The base's vLLM pin (dated 2026-07-12) just misses upstream **PR #47356** (merged 2026-07-07), which
excludes `kv_cache_memory_bytes` from `CacheConfig.compute_hash()`. That field is a runtime-measured
KV-sizing knob, not a compiled-graph-shape factor, but pre-patch it leaks into the hash — so setting
it perturbs the `torch.compile` cache key and needlessly invalidates the warm FlashInfer/Triton
autotune caches every boot. The fix is a single-line addition to a Python set literal, identical to
the literal upstream diff. Applied as a thin source-overlay layer, same extract-then-patch pattern as
the prior lane's FlashInfer overlay. Full kit + reproducibility notes: `patches/vllm-pr47356-vgx10/`.

## Config changes shipped vs. the prior lane

### CUDA graph capture: 48 → 96 (the formula changed in 0.25.1)

The prior lane documented the capture ceiling as `MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1)` =
`12 × (3 + 1)` = 48. **vLLM 0.25.1 changed its own internal default formula** to
`min(MAX_NUM_SEQS × (1 + num_speculative_tokens) × 2, 512)` — note the added `× 2`. Carrying the old
48 onto 0.25.1 silently ran the engine at **half** its intended capture ceiling. Root-caused by a
direct read of 0.25.1's `config/vllm.py`, then verified against the deployed config. Setting it to
96 (`12 × (3 + 1) × 2`) moved concurrency-8 throughput from ~82.9 to ~84.55 tok/s and lifted the KV
pool from 2,137,107 to a **2,279,532–2,376,103** token range (real boot-to-boot variance from
ambient memory conditions — not one fixed number). **New rule if you change `MTP_NUM_TOKENS`:**
`MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1) × 2`, capped at 512 (so MTP 5 → `min(12×6×2, 512)` = 144).

### Breakable CUDA graph: keep `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (new in 0.25)

DeepSeek-V4's model classes (`DeepseekV4ForCausalLM`, `DeepSeekV4MTPModel`) lack the
`@support_torch_compile` decorator that vLLM 0.25's standard compilation pipeline expects, so 0.25.1
**auto-enables** a separate "breakable CUDA graph" path for this model — eager per-layer scratch
tensors instead of one fused `torch.compile`d graph. We explicitly tested disabling it (`=0`, which
re-enables the standard `torch.compile` pipeline) to see if it would close the KV/throughput gap
vs. the prior lane. **It made things worse:**

- KV pool dropped (2,279,532 → 2,250,634).
- Spec-decode draft-acceptance collapsed from the normal ~0.637–0.640 to ~0.458.
- Concurrency-8 latency became erratic — one batch spiked to 221 s wall-clock vs. ~73 s for sibling
  batches in the same run.

The boot log itself warns `torch.compile is turned on, but the model … does not support it` when
forced off. Root cause: DeepSeek-V4 genuinely isn't compatible with the standard `torch.compile`
pipeline on this GB10/0.25.1 combination. Reverted immediately. **Do not disable this on this model.**

### FlashInfer sampler: keep `VLLM_USE_FLASHINFER_SAMPLER=1` — verified hang-safe here

There is a real, known concern that generic vLLM/GB10 guidance raises: vLLM issue #43885 plus an
open, unmerged PR #44405 propose to **default-disable** the FlashInfer top-k/top-p sampler on
GB10/consumer-Blackwell (SM120/121) because of a CUDA-stream-hang bug in FlashInfer's radix top-k
kernel (`RadixTopKMaskLogitsMultiCTA` — an end-of-kernel barrier-reset race where the leading CTA
cleared shared state without syncing a still-spinning peer CTA, which on desktop GPUs with context
time-slicing can spin forever).

This kit runs the FlashInfer sampler **on** deliberately (originally for a *different*, already-fixed
garble bug — FlashInfer PR #3615, merged 2026-06-16). Direct verification this session:
**PR #3615 is the fix for the hang bug too** — its own description says so, citing the same kernel
and root cause, and its fix (last-CTA-to-exit resets state, not the leading CTA) is exactly what
stops the hang. `gh api` confirms the base's pinned FlashInfer commit (`0472b9b3`) is **128 commits
ahead of PR #3615's merge commit with zero behind** — the fix is unambiguously included. So **no
tradeoff exists for this setup**: the sampler stays on, covering both the garble fix and the
hang-avoidance the newer guidance worries about, because they are the same fix.

One residual, unrelated FlashInfer item is on the watch-list only: **#3618 / PR #3625** — concurrent
CUDA-stream top-k calls sharing a buffer can produce silently *wrong* (not hung) output under
specific multi-stream conditions. Not confirmed to affect this deployment; flagged, not acted on.

### `TRITON_CACHE_DIR`: pin it to the persistent bind (a hardening lesson)

Under `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `torch.compile` is fully disabled
(`compilation_config.mode=NONE`), so vLLM's normal cache-redirect
(`InductorAdaptor.initialize_cache()`, which would route `TRITON_CACHE_DIR` under the persistent
`VLLM_CACHE_ROOT`/HF-cache mount) **never runs** — it's part of the torch.compile-integrated path.
Without an explicit override, Triton's hand-written kernel cache (used regardless of torch.compile
status, e.g. for DSpark's own custom kernels) silently fell through to the container-ephemeral
`/root/.triton/cache`, which is wiped on **every container recreate**. This kit's systemd units
always do a full `docker rm -f` + recreate (every watchdog self-heal, every redeploy, every reboot),
never an in-place `docker restart` — so every restart was cold-recompiling Triton kernels, exactly
the 10–30-minute post-boot JIT-storm the warm-up and observability work (see docs/07) exist to
avoid. Fix: set `TRITON_CACHE_DIR` explicitly under the persistent HF-cache mount (same pattern as
`TILELANG_CACHE_DIR`). Verified live: 1400+ compiled-kernel files landed in the new persistent path
on the next boot, and a full correctness gate afterward showed zero regression.

### `GPU_MEMORY_UTILIZATION`: unchanged at 0.85

Still the sweet spot for a dedicated pair.

## Hardening pass (2026-07-15, before promotion)

Five items investigated as due diligence before flipping the default:

1. **FlashInfer sampler hang risk** — resolved safe; the pinned FlashInfer already carries PR #3615
   (see config change above). No action beyond confirming the sampler stays on.
2. **Unified-memory OOM during weight load** *(recommended for anyone running vLLM on DGX Spark's
   unified memory, not 0.25-specific)*. During the ~155 GiB checkpoint load, NVMe reads can fill the
   Linux page cache faster than the unified-memory manager drains it; by KV-allocation time no free
   pages remain and the kernel OOM-killer can fire **before** vLLM's own `gpu_memory_utilization`
   check ever runs — a host/page-cache race, invisible to and unfixable by tuning
   `GPU_MEMORY_UTILIZATION`. Recommended host-level mitigation via `/etc/sysctl.d/` (persists across
   reboots): `vm.min_free_kbytes` ≈ 3–5 GiB depending on node RAM (reserves an always-free floor so
   the kernel reclaims page cache proactively — reported the single most impactful lever),
   `vm.dirty_ratio=5`, `vm.dirty_background_ratio=2`, `vm.vfs_cache_pressure=200`. This needs real
   root on the host — **not** the unprivileged docker-group deploy user — so it is a one-time manual
   operator step the automated kit scripts cannot apply themselves.
3. **Triton kernel cache staleness** *(a narrower issue than the persistence gap above)*. vLLM issue
   #41871 on SM121: an incomplete Triton compile-cache key can occasionally let a cubin compiled
   under one transient toolchain/fallback state get reused under a different one, producing silently
   garbled output with no crash. Investigated: this kit's cache-key hashing already folds in enough
   factors (torch content hash, device/system fingerprint, Triton's own arch-in-key backend hash) to
   protect the common case — an image/toolchain upgrade automatically gets a fresh cache dir. The
   residual risk is a narrow transient-fallback edge case; a diagnostic comparing the recorded arch
   in cached kernel metadata against the live GPU's compute capability found no mismatch. **Reader
   signal:** mixed-language tokens or malformed tool-call output *with no crash* is the signature —
   the mitigation is a *targeted* cache wipe (not a full reset), which shouldn't be done
   automatically because it costs real JIT-recompile time.
4. **Streaming special-token leakage under MTP + tool-calling** — a shared-code-path bug class
   (community report against vLLM's DeepSeek-V4 tool-call parser, not GB10-specific): under
   speculative decoding the draft model could in principle sample a raw special token that a stricter
   streaming-buffer guard doesn't catch, leaking it into visible output. Verified clean: 7 live
   streaming requests against this exact deployment (including 3 that genuinely triggered tool calls,
   the specific vector of concern), zero leaked tokens in either the parsed response fields or a raw
   text sweep of the full stream. Not proof of absence — the bug is described as rare — but a
   reasonably thorough negative result.
5. **`TRITON_CACHE_DIR` persistence** — fixed; see the config change above.

## Residual gaps (open, honest)

The promotion does not close everything. These are real and bounded:

- **KV-cache pool capacity.** The new profile measures **93–97%** of the prior lane's 2,446,083-token
  pool (boot-to-boot variance in that range, not one number) at the identical
  `GPU_MEMORY_UTILIZATION=0.85`. Root cause **not** fully identified. Ruled out: the V1 model runner
  as an ablation (it crashes — DeepSeek-V4's spec-decode config hard-requires the V2 runner on this
  checkpoint); and `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (tested live — made the gap slightly *worse*).
  Remaining unruled-out candidates are GB10/unified-memory memory-accounting differences between vLLM
  0.21.x and 0.25.x that weren't isolated in the time available. Genuinely open work.
- **Concurrency-8 throughput.** ~84.55 tok/s vs. the prior profile's ~91.72 tok/s baseline (~−7.8%),
  improved from an initial ~82.9 tok/s (~−9.6%) after the capture-size fix, but not fully closed.
  **Single-request (C1) throughput is at parity** with the prior baseline — this is specifically a
  concurrency-scaling gap, not a general slowdown.
- **vLLM issue #42948** (external, not this kit's bug). DeepSeek-V4's hybrid-attention-group
  prefix-cache implementation can lose cache keys on request reassignment under speculative decoding,
  so the prefix-cache hit rate can be near-zero in practice even though prefix caching is enabled and
  appears to work. Still open upstream as of the last check (2026-06-22); no merged fix, and no known
  workaround short of disabling prefix caching entirely (which trades the benefit away rather than
  fixing the bug). Relevant to workloads with repeated/shared prefixes.

## Why promote despite the open gaps

The whole-image correctness gate passed cleanly: composite eval **100/100**, with correctness,
garble-clean, latency-SLO, and spec-decode sub-scores all at **1.00** (this kit's own
`eval-cluster.sh`-equivalent harness). The residual gaps are measured, bounded, and understood in
shape if not in root cause; they are a capacity/throughput deficit, not a correctness or stability
risk. Shipping a genuine vLLM version bump has independent value — staying current with upstream, the
newer FlashInfer/b12x lineage — and the prior lane stays fully documented and rollback-able, so the
throughput-tested lane remains one config swap away.

## Rollback to the prior 0.21.x lane

Set `DSPARK_VLLM_IMAGE` back to the prior lane's tag (or the digest-pinned base for the unpatched
image), swap `cluster.env` section 4 to its ROLLBACK block, re-render both nodes, and restart the
ordered pair. The prior lane's build path is preserved as `bringup/05-build-image.prior-0.21.sh`, and
its FlashInfer overlay kit is intact at `patches/flashinfer-pr3615/`. The prior serve profile is
recorded below and in git history at the pre-promotion commit.

---

## Prior production profile — vLLM 0.21.x (the rollback lane)

Everything from here down documents the **previously-shipped default**, kept for rollback and
historical record. It remains reproducible via the pins in the `cluster.env` ROLLBACK block plus
`bringup/05-build-image.prior-0.21.sh`.

The prior lane is the pinned stage-c runtime plus the FlashInfer #3615 safety overlay, with vLLM
0.21.x inside the image:

| Knob | Shipped value | Why it stayed |
|---|---:|---|
| Context | 1,048,576 | The model's calibrated YaRN ceiling. |
| MTP draft length | 3, probabilistic | Clean output with useful speculative acceptance. |
| Max sequences | 12 | Best tested concurrency/capacity balance. |
| CUDA graph capture | 48 | Covers `12 × (3 + 1)` decode tokens (the 0.21.x formula) and avoids eager fallback. |
| Batch-token cap | 8,192 | Best tested prefill/decode balance. |
| Long-prefill threshold | 4,096 | Prevents a long prompt from monopolizing the scheduler. |
| GPU memory utilization | 0.85 dedicated / 0.80 co-located | 0.85 produced 2,446,083 KV tokens after the vision sidecar was removed; 0.80 produced 2,022,645 with extra headroom. |
| KV cache | `nvfp4_ds_mla` | The capacity win that makes 1M context practical on two Sparks. |

Keep `GLOO_SOCKET_IFNAME` pinned alongside `NCCL_SOCKET_IFNAME`. NCCL can still use both RDMA rails
for payload traffic while Gloo uses the stable control rail for CPU-side coordination.

### Prior-lane A/B tests that changed production

**CUDA graph capture: 12 → 48.** The original capture ceiling covered request count, but not
speculative decode width. Raising it to `MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1)` kept the hot decode
path graphed; at concurrency 8, measured throughput moved from 92.3 to 111 output tokens/s (about
+20%). *(On the 0.25.1 lane this formula gained a `× 2` factor — see the current-profile section
above.)*

**Long-prefill scheduling: disabled → 4,096.** Without a threshold, one long prefill could block
short interactive requests. Enabling the 4,096-token chunk threshold reduced short-request TTFT from
roughly 58.7 s to 5.9 s during the mixed-workload test. Carried forward unchanged to the 0.25.1 lane.

**FlashInfer top-k safety overlay.** The FlashInfer sampler was required for the DSpark path, but its
multi-CTA radix top-k kernel could leave the global counter in a stale state on SM120/121. The prior
lane applies the pinned source-only fix from FlashInfer PR #3615 as a thin image layer. *(The 0.25.1
lane gets the same fix from its base's FlashInfer 0.6.15-dev, so no overlay is needed there.)*

**Graceful termination and hardware-fault evidence.** `--shutdown-timeout 30` gives normal SIGTERM
shutdowns time to finish in-flight work; three drain tests completed without truncating the active
response, and a 29-request streaming run completed 29/29. The optional Xid monitor is deliberately
alert-only. Carried forward unchanged.

### Prior-lane tests that were rolled back or kept as fallbacks

| Experiment | Result | Decision |
|---|---|---|
| MTP 3 → 5 | Output stayed clean, but the KV pool fell from 1.716M to 1.578M tokens and positions 3/4 accepted only 16.6%/8.8%. | Rolled back to 3. |
| Confidence scheduler (`-17`, then `-15`) | No stable end-to-end win across C1 and C8. | Off by default. |
| Batch-token cap 8,192 → 16,384 | Higher prefill burst, but about 43% less KV capacity in the tested lane. | Keep 8,192. |
| FP8 KV cache | About 19% less KV capacity than NVFP4 MLA while still supporting the 1M gate. | Diagnostic fallback only. |
| Full-decode-only mode | No useful KV-capacity gain. | Do not enable. |
| Decode fusion prototype | Correctness/performance trade-off did not clear the acceptance bar. | No-go. |

These are not universal laws. Image commits, kernels, model weights, and vLLM scheduling all move;
the point of the ledger is to preserve the decision boundary and the rollback.

### The earlier `v0250-cand4` rejection (a different candidate — see the top of this page)

vLLM 0.25 added substantial DeepSeek-V4 and disaggregated-serving work. The `v0250-cand4` candidate —
a 0.25.0 **rebase of the same recipe lineage** as the 0.21.x lane — needed several corrections before
it could even be evaluated (repair the container entrypoint/startup contract; restore the sampler/NCCL
integration; pass correctness gates before comparing speed; add the continuous-prefill boundary safety
fix). After those, it passed the hard gates (evaluation 8/8, garble 1.0, a 200K needle, 95.1% KV
occupancy) and improved the single-request lane by ~4.3%, with speculative acceptance rising from
~0.32 to 0.42–0.43. But the representative concurrency-8 run was still ~9.2% slower than the prior
baseline, so it was not promoted. The follow-up ladder did not change that call:

| `v0250-cand4` experiment | Observed result | Call |
|---|---:|---|
| MTP 3 → 5 | ~16.9% slower | Reject. |
| Expert parallelism | ~14.8% slower | Reject. |
| DeepGEMM variant | ~0.2% throughput change (noise) and ~15% less KV | Reject. |
| Partial-prefill path | Unsupported in the tested DSpark combination | Defer. |
| Acceptance-kernel backport | Depended on newer MRv2-only interfaces | Do not force into production. |

That result stands **for `v0250-cand4`**. The `dspark-vllm-gx10` port that shipped is a separate
lineage and was evaluated on its own merits — do not read the two as the same experiment.

## Promotion gates

A candidate must pass all of these before becoming the default:

- Eval and garble gates with reasoning both enabled and disabled.
- 200K and 1M-context retrieval/needle checks appropriate to available time.
- Mixed long-prefill + short-request latency, not only isolated throughput.
- C1 and representative C8 throughput with the same sampling and output budget.
- KV-capacity comparison from boot logs, plus peak unified-memory headroom.
- Streaming completion and graceful-drain tests.
- Two-node restart, watchdog recovery, and listener exposure audit.

**Status (2026-07-15):** `dspark-vllm-gx10` (vLLM 0.25.1) **passed the correctness gates**
(composite 100/100; correctness/garble/latency-SLO/spec-decode sub-scores all 1.00) and was
promoted. It passed on correctness and stability; it did **not** match the prior lane on two
throughput/capacity axes (C8 ~−7.8%, KV pool 93–97% — see *Residual gaps*), which are documented as
open work rather than blockers, because a real version bump carries independent value and the prior
lane stays rollback-able. When a future candidate wins outright, update the pinned image/digest, this
ledger, and the measured capacity in `runtime/cluster.env.example` in the same pull request.

## Upstream references

- [anemll/dspark-vllm-gx10](https://github.com/anemll/dspark-vllm-gx10) — the vLLM 0.25.1 GB10 base image
- [vLLM PR #47356 — exclude kv_cache_memory_bytes from CacheConfig.compute_hash](https://github.com/vllm-project/vllm/pull/47356)
- [vLLM issue #43885](https://github.com/vllm-project/vllm/issues/43885) / [PR #44405](https://github.com/vllm-project/vllm/pull/44405) — proposed default-disable of the FlashInfer sampler on GB10 (the hang concern)
- [FlashInfer PR #3615 — multi-CTA radix top-k reset-race fix](https://github.com/flashinfer-ai/flashinfer/pull/3615) (fixes both the garble and the hang)
- [FlashInfer issue #3618](https://github.com/flashinfer-ai/flashinfer/issues/3618) / [PR #3625](https://github.com/flashinfer-ai/flashinfer/pull/3625) — concurrent-stream top-k buffer sharing (watch-item)
- [vLLM issue #41871](https://github.com/vllm-project/vllm/issues/41871) — SM121 Triton compile-cache-key staleness
- [vLLM issue #42948](https://github.com/vllm-project/vllm/issues/42948) — DeepSeek-V4 prefix-cache key loss under speculative decoding
- [vLLM v0.25.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- [NVIDIA DGX Spark local AI overview](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
