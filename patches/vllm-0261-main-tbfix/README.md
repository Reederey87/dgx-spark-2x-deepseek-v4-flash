# c8r thinking-budget fast-path fix

> **Superseded 2026-08-19** as the top layer: upstream PR #52329 (carried by
> [patches/vllm-0261-main-smpcache/](../vllm-0261-main-smpcache/)) folds the
> thinking-budget term into upstream's cached `needs_logits_processing`
> predicate, so current production no longer applies this patch. The kit stays —
> it is rollback rung 0c and a middle layer of the full build chain
> (`bringup/05-build-image.sh`).


This is a small, reproducible runtime layer over
`vllm-dspark-runtime:v0261-main-c8r`. It changes only
`vllm/v1/worker/gpu/sample/sampler.py`.

An active `thinking_token_budget` must run `ThinkingBudgetState.apply()` so the
reasoning-end marker can be forced. On the pinned vLLM main revision, default
sampling (`temperature=1`, `top_p=1`) took a no-op fast path and skipped that
processing. The included patcher asserts the exact pinned source block before it
changes anything, then the image build compiles and checks the result.

Build on the head node without changing services:

```bash
bash patches/vllm-0261-main-tbfix/build-and-distribute.sh
```

Add `--distribute` to send the resulting image over rail 1 and assert identical
image IDs on both nodes. The production tag is
`vllm-dspark-runtime:v0261-main-c8r-tbfix`; immediate rollback is the unchanged
`vllm-dspark-runtime:v0261-main-c8r` image plus its `-c8r` cache roots.

The fix was promoted on 2026-08-12 after unset/4K/8K budget probes, ordinary
C1/C8 non-inferiority, Python/coding/tool correctness, speculative acceptance,
and both-node image parity passed. It does not set a server-default budget.
