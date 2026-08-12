# The vLLM 0.26.0 rebase and promotion (2026-07-28)

This page documents the **0.26.0 rebase**, promoted 2026-07-28 (replacing the 0.25.1 lane —
[docs/08](08-optimization-and-vllm-025.md)). **Live production since 2026-08-11 is the
full-source c8r lane** ([docs/14](14-vllm-027-c8r.md)); 0.26.0 cand7 is the first image
rollback. Numbers are directional measurements from one 2× DGX Spark pair; re-run the
gates on your own hardware.

> **2026-08-11 update:** production moved to **`v0261-main-c8r`** (main @48bada6ea4 +
> overlay0261 + #49731 revert). See [docs/14](14-vllm-027-c8r.md).
>
> **2026-08-10 update:** the 0.26.0 image moved to **`v026-gx10-cand7-backports`** (cand4 +
> backports #48957/#48047/#50330, own `-cand7` cache roots) after a same-day gate measured it
> throughput-**neutral** — it buys the #50330 draft-quant correctness fix at zero measured
> cost. Full cand7 record: [docs/13](13-vllm-026-cand7.md).

## Lane definition

| Component | Pin |
|---|---|
| Serving image | `vllm-dspark-runtime:v026-gx10-cand7-backports` (built by [patches/vllm-026-rebase/](../patches/vllm-026-rebase/)) |
| Base image | `vllm/vllm-openai:v0.26.0` (official **linux/arm64**, digest-pinned `@sha256:ffb2d59b…abf52`) |
| Overlay | `gx10-overlay-026.patch` — 15 files: the `dspark-vllm-gx10` GB10 overlay (Anemll, see [NOTICE](../NOTICE)) cherry-picked onto v0.26.0 with **zero conflicts**, plus two guards and five backports (below + [docs/13](13-vllm-026-cand7.md)) |
| Extra pins | FlashInfer `0.6.15-dev` @ `0472b9b3…`, b12x `0.15.3` @ `7dc6fb8f…` (unchanged from the 0.25.1 lane; `FLASHINFER_DISABLE_VERSION_CHECK=1`) |
| Serving config | **unchanged** from the 0.25.1 lane except one flag: 1M ctx, `nvfp4_ds_mla`, DSpark n=3 probabilistic, util 0.85, `KV_CACHE_MEMORY_BYTES=21316272128`, **`--no-async-scheduling`** (see "Open items" — restores the reported KV pool at zero measured cost). **2026-07-29 update:** DSpark **n=2** + explicit `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4"}'` — see [11](11-v026-feature-qualification.md) |

No CUDA rebuild: the official arm64 image supplies every 0.26 dependency (its
`TORCH_CUDA_ARCH_LIST` includes 12.0 — family-compatible with GB10 sm_121a), so the lane is a
thin Python layer (see the patch-kit README for the extract-then-patch build).

## The two guards the lane needs (both shipped in the overlay)

1. **Warn-only `dspark_block_size`.** v0.26.0 hard-rejects `num_speculative_tokens=3 <
   dspark_block_size: 5` claiming garbled output. On this stack n=3 is measured garble-clean
   (needle HITs to 944,471 tokens, reasoning on and off) and complying (n=5) costs −12%
   throughput, so the overlay logs a warning and proceeds. Upstream: vllm-project/vllm#50012.
2. **Zero-token-prefill-chunk skip.** 0.26's capture/warmup batches (padded to
   `max_num_seqs`) emit zero-token prefill spans that crash flashinfer's sparse-MLA segment
   normalize (`cannot reshape tensor of 0 elements into shape [0, -1]`) — an undocumented
   boot crash any DSpark/0.26 deployment with this batch shape hits. The overlay skips empty
   chunks (same precedent as upstream's #48957 empty-launch skip).

Dependency note: inherit the base image's `apache-tvm-ffi` (0.1.10) — pinning 0.1.9 breaks
cutlass-dsl 4.6.0's `tvm_ffi_provider` at first dummy run.

## The acceptance-regression hunt (why the backports are in)

Stock 0.26.0 measured **−7% eval-workload spec-decode acceptance** vs the 0.25.1 lane
(0.638 → 0.592), concentrated at draft positions 2–3; bench-workload acceptance stayed flat
(~0.43). Reverting the release's own perf kernels (#48137, #48660 — measured as acceptance
costs on 0.25.1) made it **monotonically worse** (0.561 / 0.548): on the 0.26 base those
kernels *help* drafter-target agreement — the 0.25.1 finding inverts. The residual was closed
by backporting two post-0.26.0 upstream DSv4 perf commits:

- **#49486** — skip topk/router when not needed (also +3.4% claimed decode TTFT);
- **#50004** — adaptive topk width (+1.0% claimed E2E throughput).

With them, acceptance **beats** the 0.25.1 lane (0.713/0.700) and throughput is up +4.3%.
Attribution between the two was not isolated (both apply cleanly; a #49486-only arm is cheap
to rebuild if ever needed). Related dead end recorded for completeness: running the drafter
on BF16 KV (upstream #48787's escape hatch) is **structurally unavailable** — the DSv4 model
code asserts fp8-family KV for target *and* drafter (`_resolve_dsv4_kv_cache_dtype`).

## Evidence (same-day A/B vs the 0.25.1 lane)

_Measured on the cand4 image. The cand7 increment re-gated throughput-neutral on 2026-08-10
(C1/C8 Welch95 CIs both cross zero, acceptance flat, KV pool byte-identical), so every number
below still stands — see [docs/13](13-vllm-026-cand7.md)._

| Metric | 0.25.1 lane | **0.26.0 lane (cand4)** |
|---|---|---|
| Composite eval | 100/100 | **100/100 ×2** |
| Throughput — aggregate @ concurrency 8 | 84.29 tok/s (same window) | **87.94 tok/s (+4.3%)** |
| Spec-decode acceptance (eval workload) | 0.638 | **0.713 / 0.700** |
| Spec-decode acceptance (bench workload) | ~0.42–0.43 | ~0.42–0.44 (flat) |
| Deep-context retrieval | (E17: clean) | **3/3 needle HITs @ 944,471 tokens**, 6/6 finish=stop, battery acceptance 0.784 |
| KV cache pool | 2,948,751 tokens | **2,948,751 tokens** (same 19.85 GiB pin; restored — the 2,075,155 first read was #47728's admission reservation under async scheduling, not a layout change; see "Open items") |
| Output distribution (T2 bit-control) | (control) | sequence-equal 827/827; logprob shift expected from the engine change, benign by evidence (correctness 1.00, acceptance up) |

## Rollback

First rung (0.26.0, same base): point `DSPARK_VLLM_IMAGE` back at
`vllm-dspark-runtime:v026-gx10-cand4-backports` **and** the three cache roots
(`VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` / `TILELANG_CACHE_DIR`) back at the pre-cand7 paths —
a source-patch image must not share compile-cache roots across variants (see
[docs/13](13-vllm-026-cand7.md)). Deeper rollback: point `DSPARK_VLLM_IMAGE` at
`vllm-dspark-runtime:vgx10-011-pr47356` (the 0.25.1 lane image, kept on both nodes) and
restart the pair. The 0.25.1 lane's build kit (`patches/vllm-pr47356-vgx10/`) and ledger
([docs/08](08-optimization-and-vllm-025.md)) are preserved.

## Open items

- **KV pool reported-shrink (#47728) — RESOLVED 2026-07-28 (async-off, zero cost).** The
  2,948,751 → 2,075,155-token move at the unchanged byte-pin was **not a physical layout
  change** — both lanes allocate the identical seven caches (20,000 blocks × 1,065,792 B;
  the 584B nvfp4 envelope and the #47493 alignment values exist identically in the 0.25.1
  lane). The delta was upstream **#47728**: the sliding-window recycling specs (SWA KV +
  C4/C128 compressor states) reserve `window−1 + max_in_flight_tokens` per request, and
  `max_in_flight_tokens` = `max_concurrent_batches × max_num_batched_tokens` = 2 × 8,192
  under `--async-scheduling` (0.25.1 reserved ~8,192). The doubled reservation was the
  entire +42% *reported* per-token charge; physical steady-state cost never changed.
  **Fix shipped in this repo's compose: `--no-async-scheduling`.** Same-window A/B:
  reported pool back to **2,948,751 / 2.81×** with C8 86.12 vs 86.92 tok/s (95% CI of the
  paired delta [−4.39, +2.78] — within noise), bench/eval acceptance flat, eval 100/100 ×2.
  Two traps recorded for anyone touching the flag: (1) **omitting** `--async-scheduling`
  does NOT disable it on 0.26 — the tri-state default resolves to enabled
  (`vllm/config/vllm.py:1059-1107`); the only off spelling is `--no-async-scheduling`.
  (2) `#` comments inside the compose command scalar comment out every arg after them
  (folded shell string) — keep notes out of the argv. Reverting #47728 itself is **not**
  recommended — it is a correctness fix (out-of-window block freeing under async
  scheduling).
- **Upstream watch.** #50012 (the block-size guard) and #49927 (the numerics findings) are
  open; if upstream lands a warn-only guard or the acceptance fixes, the corresponding
  overlay pieces get dropped on the next rebase. The drafter-attention fix #48167 is already
  in 0.26.0; `thinking_token_budget` remains unsupported on the V2 runner (same as 0.25.1).
