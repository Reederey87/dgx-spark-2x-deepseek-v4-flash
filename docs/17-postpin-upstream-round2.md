# Post-pin upstream round 2 (2026-08-18/19) — ixfix + c128arev + smpcache promoted

The second upstream survey of the vLLM `main` delta on top of the production pin,
and the first one whose candidates all landed. Four lanes were A/B-tested on the
production pair, one payload per lane, under the standing gate (warm paired
non-inferiority, needles on both arms, T2 logprobs, functional battery — cold arms
discarded and re-armed warm). **Three lanes promoted, one on HOLD.**

- **Survey window:** `c794754062..69d3335066` — 89 commits (the delta after the
  2026-08-14 survey), 14 shortlisted and deep-dived (diff + PR body + prod-path
  check + port cost), plus one community scan (NVIDIA DGX Spark forums, Aug 2026).
- **Method note:** every promoted layer is a one-file derivative image with its own
  cache roots, gated `qualify-lane`-style against the then-current production, then
  folded into production with the prior image kept resident as the rollback rung.

## Production now

| Knob | Value |
|---|---|
| Image | `vllm-dspark-runtime:v0261-main-c8r-tbfix-ixfix-c128arev-smpcache` (`sha256:603bb93273b9…`) |
| Upstream pin | vLLM `main` @ `48bada6ea4` + overlay0261 + #49731 revert + four derivative layers |
| Weights | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Cache roots | the `-smpcache` set (`vllm-cache-smpcache`, `flashinfer-smpcache`, `tilelang-smpcache`, `triton-cache-smpcache`) — roots move with the image, always |
| KV | `nvfp4_ds_mla`, `KV_CACHE_MEMORY_BYTES=21316272128` → **3,027,217** tokens (unchanged, admission-verified after each promotion) |

Rollback ladder (all images resident on both nodes):
`v0261-main-c8r-tbfix-ixfix-c128arev` + `-c128arev` roots (immediate) →
`v0261-main-c8r-tbfix-ixfix` + `-ixfix` roots →
`v0261-main-c8r-tbfix` + `-tbfix` roots →
`v0261-main-c8r` + `-c8r` roots → cand7 → cand4 → 0.25.1 → 0.21.x.

## Lane A — #52492 "keep indexer scoring in breakable graphs" → **PROMOTED 2026-08-18**

The short-context shortcut in `DeepseekV4Indexer.forward` was missing
`and not torch.cuda.is_current_stream_capturing()`. Breakable PIECEWISE capture
runs with short dummy metadata, so the unguarded branch is **baked into the
captured graph**: on replay, any request whose cached prefix exceeds 2048 tokens
skips learned indexer scoring and attends only to candidates 0..511 — silently.
This deployment (MRV2 + breakable graphs + 1M ctx + prefix caching) meets every
activation condition, and the pre-fix base had never been re-needled.

Kit: [patches/vllm-0261-main-ixfix/](../patches/vllm-0261-main-ixfix/) (3-line edit,
one file). Correctness-only by construction — the promotion bar was non-inferiority,
not a win. Gate: needles **3/3 at 200K/944K on both arms** (the first long-context
revalidation on this base), long-generation battery clean, T2 flat, acceptance
0.805/0.828. The first C8 read was inconclusive on cold fresh roots (one-sided
lower-95 −3.91% vs the −3.0% margin, candidate trending upward as autotune warmed);
the warm re-arm passed cleanly — **C1 −0.19%, C8 +0.34%, energy NI both**.

## Lane B — #51318 "revert adaptive C128A metadata packing" → **PROMOTED 2026-08-18**

#50004 (in the pinned base, shipped verbatim by the overlay) made the eager C128A
metadata builder write packed rows of *batch-dependent* width while the
graph-captured sparse-decode consumer keeps its capture-time row stride — rows ≥1
read stale offsets → wrong top-k indices, silently (upstream issue #52448:
reasoning loops to `max_tokens` at concurrency ≥10 on exactly this model+graph
mode). The revert restores fixed capacity-width rows so writer and reader always
agree.

Kit: [patches/vllm-0261-main-c128arev/](../patches/vllm-0261-main-c128arev/)
(~10-line revert, one file), stacked on `-ixfix`. Gate + warm re-arm (the gate's
bookend guard tripped on baseline drift, so the warm re-arm decided): battery
arm-neutral both windows, **C1 +2.60% / +2.10% — a repeatable small win** (on GB10
the fixed-width rows beat the adaptive packing; #50004's ~1% claim was unstated
hardware — GB300≠GB10 again), **C8 tie** (+0.41%/−0.43% across two windows), energy
NI. Side result: the documented python30 length-cap flake persisted identically on
both fixed images — #52492 and #51318 are now both ruled out as causes; the trait
stands as the #50580 effort-prefix + checkpoint behavior.

## Lane C — #52329 "MRV2: cache logits-processing request state" → **PROMOTED 2026-08-19**

Upstream now computes a per-slot cached `needs_logits_processing` bool once in
`add_request()` — **including the thinking-budget term** — and deletes the per-step
`_requires_logits_processing()` scan. That term is exactly what the tbfix layer
added, so adopting #52329 **upstream-subsumes and retires the tbfix patch**: one
less overlay file to re-port on every rebase. Hygiene, not speed.

Kit: [patches/vllm-0261-main-smpcache/](../patches/vllm-0261-main-smpcache/) (one
file), stacked on `-c128arev`. The decision variable was the thinking-budget smoke:
**12/12** (budgets none/4096/8192 honored exactly like tbfix — the cached predicate
is behaviorally identical). Battery arm-neutral (nprobe 0.819/0.831, T2 flat). Warm
re-arm: **C1 +0.75% with energy/token upper −0.81%** (the payload removes ~40µs/step
of host NumPy work — slightly less energy per token, as designed), C8 −0.61% NI,
energy NI.

## Lane P — `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` → **HOLD**

The config-only candidate (community report: retention fixes allocation-driven
prefix-cache invalidation in parallel long-context agentic workloads on this exact
model/hardware; upstream #52216 is meanwhile making `0` the default). Probed
fresh-boot A/B: prime fixed branches (23.6K / 73.7K), churn 3.49M tokens (>1.15× the
pool) to force eviction, resume and measure survival.

**Both arms returned 0% survival on both branches** (resume-TTFT ≈ prime-TTFT, hit
deltas 0). The invalidation reproduces on this stack — the community failure mode
is real here — but retention=4096 did not prevent it at this regime: the free-block
queue is FIFO oldest-first, so a completed branch's densely-cached blocks are
popped from token 0 regardless of SWA checkpoint spacing. The benefit most likely
lives in a long-running steady-state regime a window probe cannot demonstrate.
Full qualification not spent. Retention=0 arrives free as the upstream default on
the next rebase (#52216); re-evaluate then.

## Also checked this round — evaluated and rejected / deferred

| PR | Verdict | Reason |
|---|---|---|
| #52502 GB10 fused-MoE fp8 tuning | SKIP | Only consumed by the Triton `fused_experts` fp8_w8a8 path; prod runs B12X MXFP4 experts. |
| #51967 DSV4 top-k index constexpr | next rebase | On-path but ~0.03% of step time; below noise here. |
| #52084 sparse top-k metadata workers | SKIP | FlashMLA/ROCm/XPU-only path; never launched under `FLASHINFER_MLA_SPARSE_DSV4`. |
| #49793 MTP all-reduce fusion | SKIP | deepseek_v32/GLM-5.2 MTP module, not deepseek_v4; rejects the probabilistic draft prod uses. |
| #51538 DSV4 sparse MLA fixes 6+7 | next rebase | Close a latent MTP n=2 drain hang (never observed here); `.cuh` changes need a full rebuild, not a derivative layer. |
| #52401 DSV4 eager cudagraph per runner | SKIP | MRV1-only fix; prod is config-forced MRV2. |
| #51395 SM120 dense-prefill disable | SKIP | Touches the `_SM120` backend; prod pins `_DSV4`. |
| #49613 thinking-budget SWAP fix | SKIP | MRV1-only file; MRV2 budget state has no such bookkeeping. |
| #52550 indexer_kv_dtype unify | next rebase | No behavior change on the fp8-default path; costs an overlay port + cache-key change. |
| #52216 prefix-retention CLI/default 0 | next rebase | Subsumed by the Lane P probe; adopt the arg when it arrives. |
| #52368 B12X linear refactor | SKIP | Refactors code newer than the pinned base; nothing to port. |
| UCX rcache leak (community) | N/A | No UCX env vars or libraries in the running container; the fabric is NCCL NET=IB directly. Recorded so it is not re-proposed. |

Still standing from the 2026-08-15 round ([docs/16](16-post-pin-qualification.md)):
#51739 NOT adopted (kernel win, TTFT regression), `reasoning_effort=high` kept,
#47808 adaptive verify HOLD (cherry-pick conflicts + SM100-gated win).

## What you should copy if you deployed an older revision of this kit

1. Re-pull and rebuild: `bringup/05-build-image.sh --distribute` now runs the full
   c8r → tbfix → ixfix → c128arev → smpcache chain and tags each rung.
2. Merge your `runtime/cluster.env` against the new example: `DSPARK_VLLM_IMAGE` /
   `DSPARK_VLLM_ROLLBACK_IMAGE` and the four `-smpcache` cache roots all moved
   together. Never point a new image at the old roots.
3. If you only want the correctness fixes without the full chain, each derivative
   kit builds standalone over its parent (`BASE=`/`TAG=` env-overridable).
4. Nothing about client behavior changed: thinking stays on/high,
   `thinking_token_budget` is still enforced at the production sampling point (now
   by upstream's cached predicate), and the KV pool is byte-identical.
