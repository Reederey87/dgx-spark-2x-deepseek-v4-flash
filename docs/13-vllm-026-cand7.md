# The 0.26.0 cand7 update — post-release backports round 2 (2026-08-10)

> **Status (2026-08-11):** cand7 is the **first image rollback rung**, not the live
> production default. Production is the full-source c8r lane — see
> [docs/14](14-vllm-027-c8r.md). This page remains the cand7 selection and gate record.

The serving image moved from `v026-gx10-cand4-backports` to
**`v026-gx10-cand7-backports`** on 2026-08-10. This page records how the picks were
selected, what was rejected and why, the gate that qualified them, and the operational
rules that come with a source-patch image. Numbers are directional measurements from one
2× DGX Spark pair; re-run the gates on your own hardware.

## Why a second backport round

cand4 ([docs/10](10-vllm-026-rebase.md)) froze the lane at v0.26.0 + two perf backports.
Upstream vLLM main kept moving — by 2026-08-10 it was **836 commits past the v0.26.0
tag**, so the lane was re-surveyed for optimizations that touch this deployment's actual
path: DeepSeek-V4-Flash on GB10 (SM121), TP=2, `nvfp4_ds_mla` KV, DSpark speculative
decoding, `FLASHINFER_MLA_SPARSE_DSV4` attention.

## Selection method (on-path analysis, not changelog shopping)

Every candidate PR was read against the serving path before being shortlisted:

- **IN** only if the touched code executes in this deployment (DSv4 model files, the
  sparse-MLA/flashinfer bridge, the DSpark spec-decode path, the cudagraph runner) and
  the change is a thin-layer-applicable Python delta (no `csrc` — the base image's `.so`
  is precompiled).
- **OUT** when path analysis shows the code is never exercised here. Examples from this
  round, rejected *on the path, not by guessing*: **#50365** (sparse_utils — unused by
  the DSV4 model-side backends), **#50911** (TokenSpeed MLA — not on the draft path),
  **#51458** (V1 runner only — this deployment runs the V2 model runner, required for
  DSpark on this checkpoint), **#49236** (+3.9% TTFT claim, but changes `csrc` — not
  deliverable by a Python overlay), **#47574** (KV-block zeroing — **boot-breaking** on
  0.26.0 without its companion warmup refactor #49903; the build smoke asserts its
  absence).

## The cand7 picks (branched from cand4, NOT cand6)

| Pick | What it does | On this path |
|---|---|---|
| **#48957** | Skips the empty c128 kernel launch | Prefill-side win; gated `cudagraph_runtime_mode != FULL` |
| **#48047** | q-head padding helper for the sparse-MLA kernel | **Alignment-only** — the overlay already carried the SM120 {8,16,32,64,128} head ladder, so behavior is unchanged; it keeps the file close to upstream for the next rebase |
| **#50330** | 4-line DSpark `get_draft_quant_config` override (hand-extracted from a 39-file CI-reorg commit) | **Correctness fix** for draft-quant handling — the payload of this round |

A hard lesson from the round in between: **cand6** (2026-08-01) bundled four plausible
backports (#49731, #48407, #50298, #50312) and was **rejected at bench** — C8 throughput
−10.7%, with #49731/#48407 as the suspects. cand7 therefore branches from cand4 and adds
only the three low-risk picks. Bundling "harmless" upstream perf PRs without per-pick
isolation is how a lane regresses; the gate below is what catches it.

## Gate (same-day arms vs cand4, 2026-08-10)

| Metric | cand4 (prod) | cand7 | Verdict |
|---|---|---|---|
| Composite eval | 100/100 | **100/100** (warm) | tie |
| KV pool (pinned) | 2,948,751 tok / 2.81× @1M | **byte-identical** | tie |
| Acceptance — eval workload | 0.663 | 0.665 | flat |
| Acceptance — bench workload | 0.490 | 0.489 | flat |
| C1 single-stream | 34.14 tok/s | 33.84 (−0.9%, Welch95 [−0.90, +0.30]) | tie |
| C8 aggregate (6 warm batches) | 88.15 tok/s | 86.81 (−1.5%, Welch95 [−12.25, +9.55]) | tie |

→ **Throughput-neutral.** The value is the #50330 correctness fix + #48047 upstream
alignment at zero measured cost. Promoted the same day; rollback = the cand4 image + its
cache roots.

Cold-run artifacts worth recording: the first eval rep on **fresh cache roots** scored
98.0 and the first C8 batch 79.26 — both warmup effects (cold prefix cache / fresh
compile roots), gone on warm re-runs. **Never gate on a cold boot.**

One blind spot this gate does not cover: every C1/C8 number above is short-context.
Decode rate at 500K+ is unmeasured on this lane — see the caveat in the
[README](../README.md#performance) and `runtime/bench-decode-depth.py`.

## The cache-root rule (why cand7 has its own roots)

vLLM's compile cache (`VllmConfig.compute_hash()`) keys on version + config — **never on
model source**. A source-patch image that shares cache roots with another variant
silently reuses stale compiled graphs and *masks the patch* (this exact trap also
matters for any A/B that changes model code — it is Finding-B methodology in upstream
issue vllm-project/vllm#49927). Rule: **every source-patch image variant gets its own
`VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` / `TILELANG_CACHE_DIR` triple.** cand7 ships
`/cache/huggingface/{vllm-cache,triton-cache,tilelang}-cand7` in
`runtime/docker-compose.dspark.yml` and `runtime/cluster.env.example`. Rolling back to
cand4 means pointing all three back — see the rollback block in
`runtime/cluster.env.example`.

## Re-auditing VLLM_* envs after an image bump

An env var that had a consumer in one image can silently become a no-op in the next (the
`VLLM_DSPARK_*` family and `DSPARK_SLOT_CLAMP` both went this way — the compose keeps a
graveyard comment for the former). After any image bump, re-audit what the installed
package actually reads:

```bash
docker exec vllm-dsv4 python3 -c "import vllm.envs as e; print('\n'.join(sorted(e.environment_variables)))"
```

Diff that list against the `VLLM_*` keys in the compose. A compose key missing from the
package's registry is a warn-only no-op: vLLM logs it as an unknown environment variable
at boot and otherwise ignores it. Keeping the compose's known-dead set out of the
environment (or clearly commented) is what keeps that boot-log unknown-var count a
meaningful drift signal.

## CUTE_DSL_ARCH — checked, N/A on this lane (2026-08-10)

Community advice (MiaAI's #7) sets `CUTE_DSL_ARCH=sm_121a` for a claimed +30% — that
measurement came from their **0.25 Anemll image's W4A16 MoE path**, a different lane.
Audited against the cand7 image before adopting:

- The actual consumers in the installed package are `quack/cache/async_compile.py` and
  `nvidia_cutlass_dsl` internals only; the two vllm files that name the variable mention
  it solely in a docstring.
- Boot logs show `Skipping CuTeDSL warmup because no compile units were requested`, and
  no quack/cute compile-cache dirs exist on either node — cute-dsl is **not on the DSv4
  0.26 serving path**, so the env would be inert here.

Not adopted. If a future image moves a serving-path kernel onto cute-dsl, re-run the
audit above (consumers first, claims second).
