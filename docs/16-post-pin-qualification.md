# Post-pin qualification (2026-08-15) — production stays c8r-tbfix

This is a **negative result that belongs in the recipe**. After the 2026-08-12
c8r-tbfix promotion, three tempting next steps were measured on the same 2× GB10
pair. **None of them became the default.** The serving image, knobs, and
`reasoning_effort=high` in this kit are still the production stack.

If you are cloning the kit today, do not “catch up” to later upstream `main`,
do not flip effort to `low`, and do not enable DSpark adaptive verification.
Those were the three things we tried. Receipts below.

## Production (unchanged)

| Knob | Value |
|---|---|
| Image | `vllm-dspark-runtime:v0261-main-c8r-tbfix` |
| Upstream pin | vLLM `main` @ `48bada6ea4` + overlay0261 + #49731 revert + tbfix sampler |
| Weights | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Spec decode | DSpark probabilistic, `MTP_NUM_TOKENS=2` |
| Scheduling | `--no-async-scheduling` (omitting the flag **enables** async) |
| Thinking | `DSPARK_REASONING=on`, `DSPARK_REASONING_EFFORT=high` |
| KV | `nvfp4_ds_mla`, `KV_CACHE_MEMORY_BYTES=21316272128` → **3,027,217** tokens |

Rollback rungs are unchanged: `v0261-main-c8r` → cand7 → cand4 → 0.25.1. See
[docs/15](15-tbfix-and-async-safety.md) and [docs/14](14-vllm-027-c8r.md).

## What we measured and left off the default

### 1. Later `main` + #51739 long-context MLA gathers — **not the default**

Upstream #51739 (“Optimize long-context MLA cache gathers”) landed 3 h 52 m
**after** the production pin. A derivative image was built from `cb30f6f8dd`
(the last commit before the KV-layout refactor #51612/#51704).

On GB10 the kernel **did** move: `cp_gather_cache` was **5.41×** at 60K,
**5.79×** at 300K, **3.00×** on a mixed batch-8 versus the production image.
That was enough to spend a serving window.

End-to-end long-prefill TTFT (nonce-prefixed 44K / 94K / 188K, so prefix
cache cannot fake the number) **did not improve**. The candidate was slower
than both production bookends at every size (+17% / +3% / +14%). A 188K
chunked prefill is ~100 s of compute; the gather save is tens of milliseconds.

Do not rebase this kit onto current `main` to “pick up #51739.” Crossing
#51612/#51704 rewrites the overlay’s `nvfp4_ds_mla` layout files. The pin
stops at `48bada6ea4` on purpose.

### 2. `reasoning_effort=low` — **keep `high`**

vLLM #50580 (already **inside** the production pin) changed what `"high"`
means: it now emits the old *maximum-effort* prefix (only `"max"` used to).
There is no distinct medium rung — `low` / `minimal` / `medium` all collapse
to an empty prefix.

A full A→B→A′ of `high` vs `low`:

| Probe | `high` (production) | `low` |
|---|---|---|
| python30 (30-line function) | 8192 tokens, `finish=length`, `content: null` | 3267 tokens, **pass** |
| “short story” eval | 293 tokens, **pass** | 2048-token cap, **fail** |
| 200K / ~944K needles | 3/3 HIT | 3/3 HIT |
| T2 prompt logprobs | bit-exact | bit-exact |
| C1 / C8 throughput | reference | non-inferior, not a win |

`low` fixes the length-cap flake and makes planned short-form worse. It is
not a quality upgrade. This kit keeps `DSPARK_REASONING_EFFORT=high`.

For a numeric length constraint or a 500+-line single-shot ask, **do not
flip the server default.** Send a request-level `thinking_token_budget`
(see [docs/06](06-reasoning-mode.md)). That is why tbfix exists: on this
image the budget is actually enforced at the production sampling point
(`temperature=1`, `top_p=1`). The server still injects **no** default
budget.

### 3. #47808 DSpark adaptive verification — **HOLD**

A different design from the old (and dead) `VLLM_DSPARK_CONFIDENCE_SCHEDULER`
env path that measured −17%. The new path sizes the batch budget on CPU from
stale confidences — no per-step GPU→CPU sync. The 0731 checkpoint already
has the confidence head.

It is **not** in the production pin, cherry-picking it onto that pin
conflicts in `input_batch.py` / `model_runner.py` and needs a second overlay
re-port, and the varlen-FULL CUDA-graph win is SM100-gated (GB10/sm_121
falls back to PIECEWISE). Output logprobs are rejected when it is on; this
kit’s T2 probe uses prompt logprobs and would be safe, but no adaptive-on
image was served.

Leave it off. There is no `enable_adaptive_verification` in the shipped
compose.

## What you should copy if you already deployed an older revision of this kit

1. Stay on **c8r-tbfix** if you are already there. That *is* current production.
2. Teach clients `thinking_token_budget` for long-form / tight `max_tokens`.
3. Do not adopt community “MTP=5 / breakable graphs off / effort=low” defaults
   from other 2×Spark writeups. Those were measured here and are not this
   recipe.
4. Keep `--no-async-scheduling` explicit. See [docs/15](15-tbfix-and-async-safety.md).
