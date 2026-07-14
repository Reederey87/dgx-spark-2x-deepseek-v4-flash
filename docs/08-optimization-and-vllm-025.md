# Optimization ledger and the vLLM 0.25 lane

This page records the experiments behind the shipped profile. It separates changes that improved
the production lane from promising vLLM 0.25 work that is **not yet the default**. Numbers are
directional measurements from one 2× DGX Spark pair; re-run the gates on your own hardware and
model revision.

## Current production profile

The supported lane remains the pinned stage-c runtime plus the FlashInfer #3615 safety overlay,
with vLLM 0.21.x inside the image. The default profile is:

| Knob | Shipped value | Why it stayed |
|---|---:|---|
| Context | 1,048,576 | The model's calibrated YaRN ceiling. |
| MTP draft length | 3, probabilistic | Clean output with useful speculative acceptance. |
| Max sequences | 12 | Best tested concurrency/capacity balance. |
| CUDA graph capture | 48 | Covers `12 × (3 + 1)` decode tokens and avoids eager fallback. |
| Batch-token cap | 8,192 | Best tested prefill/decode balance. |
| Long-prefill threshold | 4,096 | Prevents a long prompt from monopolizing the scheduler. |
| GPU memory utilization | 0.85 dedicated / 0.80 co-located | 0.85 produced 2,446,083 KV tokens after the sidecar was removed; 0.80 produced 2,022,645 with extra headroom. |
| KV cache | `nvfp4_ds_mla` | The capacity win that makes 1M context practical on two Sparks. |

Keep `GLOO_SOCKET_IFNAME` pinned alongside `NCCL_SOCKET_IFNAME`. NCCL can still use both RDMA
rails for payload traffic while Gloo uses the stable control rail for CPU-side coordination.

## A/B tests that changed production

### CUDA graph capture: 12 → 48

The original capture ceiling covered request count, but not speculative decode width. Raising it to
`MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1)` kept the hot decode path graphed. At concurrency 8, measured
throughput moved from 92.3 to 111 output tokens/s (about +20%). This is now a preflight invariant;
if MTP becomes 5, the matching capture ceiling is 72.

### Long-prefill scheduling: disabled → 4,096

Without a threshold, one long prefill could block short interactive requests. Enabling the 4,096
token chunk threshold reduced short-request time to first token from roughly 58.7 seconds to 5.9
seconds during the mixed-workload test. The threshold is wired into the serve command and checked
after startup.

### FlashInfer top-k safety overlay

The FlashInfer sampler was required for the DSpark path, but its multi-CTA radix top-k kernel could
leave the global counter in a stale state on SM120/121. The kit now applies the pinned source-only
fix from FlashInfer PR #3615 as a thin image layer. This keeps the proven base image available for
rollback while making the patched tag the deployment default.

### Graceful termination and hardware-fault evidence

`--shutdown-timeout 30` gives normal SIGTERM shutdowns time to finish in-flight work. Three drain
tests completed without truncating the active response, and a 29-request streaming run completed
29/29. The optional Xid monitor is deliberately alert-only: catastrophic hardware codes capture
logs but never trigger a blind service bounce.

## Tests that were rolled back or kept as fallbacks

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

## vLLM 0.25 investigation: promising, not ready to ship

vLLM 0.25 added substantial DeepSeek V4 and disaggregated-serving work. On this exact ARM64/GB10
lane, however, the candidate image needed several corrections before it could even be evaluated:

1. Repair the container entrypoint and startup contract.
2. Restore the sampler/NCCL integration expected by the DSpark runtime.
3. Pass correctness gates before comparing speed.
4. Add the continuous-prefill boundary safety fix used by the candidate.

After those fixes, the candidate passed the hard gates: evaluation 8/8, garble score 1.0, a 200K
needle test, and 95.1% KV occupancy. It also improved the single-request lane by about 4.3%, with
speculative acceptance rising from roughly 0.32 to 0.42–0.43. But the representative concurrency-8
run was still about 9.2% slower than the production baseline, so the lane was not promoted.

The follow-up ladder did not change that call:

| v0.25 experiment | Observed result | Call |
|---|---:|---|
| MTP 3 → 5 | ~16.9% slower | Reject. |
| Expert parallelism | ~14.8% slower | Reject. |
| DeepGEMM variant | ~0.2% throughput change (noise) and ~15% less KV | Reject. |
| Partial-prefill path | Unsupported in the tested DSpark combination | Defer. |
| Acceptance-kernel backport | Depended on newer MRv2-only interfaces | Do not force into production. |

The correct conclusion is narrower than "vLLM 0.25 is slow": it is **not ready for this pinned
2×Spark DSpark lane yet**. Re-test when the DSpark runtime rebases onto a newer vLLM point release,
when FlashInfer/DSpark kernels land upstream, or when the C8 scheduler and KV regressions have an
identified fix.

## Promotion gates

A future candidate must pass all of these before becoming the default:

- Eval and garble gates with reasoning both enabled and disabled.
- 200K and 1M-context retrieval/needle checks appropriate to available time.
- Mixed long-prefill + short-request latency, not only isolated throughput.
- C1 and representative C8 throughput with the same sampling and output budget.
- KV-capacity comparison from boot logs, plus peak unified-memory headroom.
- Streaming completion and graceful-drain tests.
- Two-node restart, watchdog recovery, and listener exposure audit.

When a candidate wins, update the pinned image/digest, this ledger, and the measured capacity in
`runtime/cluster.env.example` in the same pull request.

## Upstream references

- [vLLM v0.25.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- [vLLM DeepSeek-V4 DSpark work](https://github.com/vllm-project/vllm/pull/46995)
- [FlashInfer multi-CTA radix top-k fix](https://github.com/flashinfer-ai/flashinfer/pull/3615)
- [NVIDIA DGX Spark local AI overview](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
