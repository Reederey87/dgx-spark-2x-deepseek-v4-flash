# Long-context crash fix — `DSPARK_SLOT_CLAMP` (legacy no-op on the 0.26 lane)

## TL;DR

**On the current 0.26 lane `DSPARK_SLOT_CLAMP` does nothing.** Verified 2026-08-10: a
grep of the installed vllm package inside the running `v026-gx10-cand7-backports` image
finds **zero readers** of the variable — it is a dead env here, a legacy of the retired
0.25.1 image lane where it was live. The crash class it guarded is handled by the gx10
overlay itself (below). The compose still injects it (`${DSPARK_SLOT_CLAMP:-1}`) purely
for harmlessness and **0.25.1-rollback compatibility** — leave the line alone; there is
nothing to tune.

## Background

DSpark speculative decoding keeps a **persistent per-request draft KV cache**. In
vLLM v1 the running set of requests is compacted as requests finish — a finished
request's slot can be reused by a different request. The community "Keys" concurrency
patch (see `docs/03-model-and-features.md`) makes the draft-KV slot
mapping **request-stable** so a reused slot never silently serves the wrong request.

On the retired 0.25.1 lane, `DSPARK_SLOT_CLAMP` was the belt-and-suspenders guard that
paired with that patch: if a **stale draft-KV slot id** survived into a step at high
sequence length (e.g. after churn near the top of a 1M-token context), an unclamped id
could index out of the allocated KV range and trigger a CUDA illegal-memory-access —
which kills the engine, not just the request.

On the 0.26 lane that mismatch window is closed at the source: the overlay's own
request-stable **req-id → slot** draft-KV mapping means no stale slot id survives into a
step, and the installed package contains no code that consults the clamp env at all.

| Lane | Status |
|------|--------|
| **0.26 (`v026-gx10-*` images, current)** | **Legacy no-op** — zero readers in the installed package (verified 2026-08-10). Value is irrelevant. |
| **0.25.1 (`vgx10-011-pr47356`, rollback)** | **Live** — `1` (default) clamps stale/out-of-range draft-KV slot ids; `0` is detect+log only (expect crashes if it fires). |

## When you'd touch it

- **0.26 lane:** never — it is inert. If you see engine deaths at high context under
  load on 0.26, this knob is not the cause and not the fix; walk the OOM/stability
  ladder in `docs/05-troubleshooting.md` (drop `MAX_MODEL_LEN`, then `MAX_NUM_SEQS`,
  then `GPU_MEMORY_UTILIZATION`, then the speculative config).
- **0.25.1 rollback lane:** keep `1`. Set `0` only to reproduce and log the raw
  condition for debugging, then set it back.

## How to override (0.25.1 lane only)

It's a plain env var read by the compose (`${DSPARK_SLOT_CLAMP:-1}`). For a one-off
run, export it before `docker compose up`, or add it to the rendered `.env.dspark`.
Do **not** bake `0` into `cluster.env` for production.

## Related

- The mismatch window that this guarded is the same one the DSpark garble fix narrows
  via probabilistic drafting (originally `MTP_NUM_TOKENS=3`; **n=2** since the 2026-07-29
  K re-tune — see `docs/03-model-and-features.md` and `docs/11-v026-feature-qualification.md`).
- If crashes persist, walk the OOM/stability ladder in
  `docs/05-troubleshooting.md` (drop `MAX_MODEL_LEN`, then `MAX_NUM_SEQS`, then
  `GPU_MEMORY_UTILIZATION`, then the speculative config).
