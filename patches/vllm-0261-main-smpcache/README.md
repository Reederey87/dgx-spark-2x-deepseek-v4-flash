# MRV2 cached logits-processing state (upstream PR #52329) — retires tbfix

A small, reproducible runtime layer over
`vllm-dspark-runtime:v0261-main-c8r-tbfix-ixfix-c128arev`. It changes only
`vllm/v1/worker/gpu/sample/sampler.py`: the tbfix-patched copy plus upstream
`6664d397bf` ("[MRV2] Cache logits-processing request state").

This is an **overlay-retirement** play, not a throughput play. The tbfix layer
exists because the per-step `_requires_logits_processing()` scan did not include
the thinking-budget term, so default-sampling requests with an active
`thinking_token_budget` silently took the no-op fast path. Upstream's rewrite
computes a per-slot cached `needs_logits_processing` bool once in `add_request()`
— **including the thinking-budget term** — and deletes the scan. Adopting it means
the tbfix patch is fully subsumed by upstream: one less file to re-port on every
rebase. Upstream's numbers: host-side gate 56.7µs → 16.8µs/call; GPU-bound serving
neutral (+0.07%).

Build on the head node without changing services:

```bash
bash patches/vllm-0261-main-smpcache/build-and-distribute.sh
```

Add `--distribute` for head→worker transfer with image-ID parity. Tag
`vllm-dspark-runtime:v0261-main-c8r-tbfix-ixfix-c128arev-smpcache`; own `-smpcache`
cache roots (see `runtime/cluster.env.example` §4).

Promoted 2026-08-19. The decision variable was the thinking-budget smoke: **12/12**
on the candidate (budgets none/4096/8192 honored exactly like tbfix — the cached
predicate is behaviorally identical). Needles/T2/functional arm-neutral; warm
re-arm C1 +0.75% with energy/token −0.81% upper (the payload removes ~40µs/step of
host NumPy work), C8 −0.61% NI, energy NI. Receipt: docs/17. Rollback is the
resident `-c128arev` image plus its `-c128arev` roots.
