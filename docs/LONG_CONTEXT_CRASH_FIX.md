# Long-context crash fix — `DSPARK_SLOT_CLAMP`

## TL;DR

`docker-compose.dspark.yml` sets `DSPARK_SLOT_CLAMP=1` by default. **Leave it at `1`.**
It is a safety clamp that prevents a class of crash (illegal-memory-access / engine
death) that can appear at long context under concurrency with DSpark speculative
decoding. Set it to `0` only when you are deliberately trying to *reproduce and log*
the underlying condition for debugging.

## Background

DSpark speculative decoding keeps a **persistent per-request draft KV cache**. In
vLLM v1 the running set of requests is compacted as requests finish — a finished
request's slot can be reused by a different request. The community "Keys" concurrency
patch (see `CREDITS.md` and `docs/03-model-and-features.md`) makes the draft-KV slot
mapping **request-stable** so a reused slot never silently serves the wrong request.

`DSPARK_SLOT_CLAMP` is the belt-and-suspenders guard that pairs with that patch: if a
**stale draft-KV slot id** survives into a step at high sequence length (e.g. after
churn near the top of a 1M-token context), an unclamped id can index out of the
allocated KV range and trigger a CUDA illegal-memory-access — which kills the engine,
not just the request.

| Value | Behavior |
|-------|----------|
| `1` (default) | **Clamp** stale/out-of-range DSpark draft-KV slot ids into the valid range. Safe; the request continues. |
| `0` | **Detect + log only** — do not clamp. Use to surface the raw condition when diagnosing; expect crashes if it fires. |

## When you'd touch it

- **Normal operation:** never — keep `1`.
- **You see engine deaths at high context under load** and want to confirm the cause:
  temporarily run one node with `DSPARK_SLOT_CLAMP=0`, reproduce, and read the log
  line it emits; then set it back to `1`.

## How to override

It's a plain env var read by the compose (`${DSPARK_SLOT_CLAMP:-1}`). For a one-off
run, export it before `docker compose up`, or add it to the rendered `.env.dspark`.
Do **not** bake `0` into `cluster.env` for production.

## Related

- The mismatch window that this guards is the same one the DSpark garble fix narrows
  by using `MTP_NUM_TOKENS=3` + probabilistic draft (see `docs/03-model-and-features.md`).
- If crashes persist even with the clamp on, walk the OOM/stability ladder in
  `docs/05-troubleshooting.md` (drop `MAX_MODEL_LEN`, then `MAX_NUM_SEQS`, then
  `GPU_MEMORY_UTILIZATION`, then the speculative config).
