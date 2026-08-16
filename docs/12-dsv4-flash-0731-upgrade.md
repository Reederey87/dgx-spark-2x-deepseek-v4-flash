# DeepSeek-V4-Flash-0731 upgrade (2026-07-31)

The day-3 model upgrade: the cluster moved from the `DeepSeek-V4-Flash-DSpark` **preview**
checkpoint to **`DeepSeek-V4-Flash-0731`**, the official V4-Flash release — with zero
serving-stack changes. This page is the evidence record: why the swap is safe, the two
offline-cache traps it surfaced, the A/B gate numbers, and one 0731 behavior worth knowing.

## What changed and why

0731 is DeepSeek's official release of V4-Flash, superseding the preview. Same
architecture and parameter count; the gains are training/post-training — DeepSeek's
published agentic results put it **ahead of its own V4-Pro preview** (Terminal-Bench 2.1
82.7 vs 72.1; DeepSWE 54.4 vs 12.8; see the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
and the benchmark table in the [README](../README.md#performance)). The repo
`DeepSeek-V4-Flash-0731` ships the DSpark speculative-decoding module attached, exactly
like the preview's `DeepSeek-V4-Flash-DSpark` repo.

## Compatibility proof (verified before touching the cluster)

| Check | Result |
|---|---|
| `config.json` | **field-identical** to the preview — same arch (`DeepseekV4ForCausalLM`), vocab 129280, YaRN 65536×16, fp8 quant config, `expert_dtype: fp4`, and every `dspark_*` field (`dspark_block_size: 5`, noise token, target layers, markov rank) |
| `tokenizer.json` / `tokenizer_config.json` | **byte-identical** — both repos' snapshots link the same blob (`628e3364…`), sha256-verified |
| Weight footprint | `model.safetensors.index.json` `total_size` **166,878,536,440 B (~155.4 GiB)** — within 8 MB of the preview. HF's "304B params / I8" widget is logical-parameter accounting (FP4 experts pack 2 per byte); it is not a storage change |
| Serving stack | unchanged — same image (`v026-gx10-cand4-backports` at the time; **cand7** since 2026-08-10, throughput-neutral — see [13](13-vllm-026-cand7.md)), same knobs, same served name `deepseek-v4-flash-dspark`, same KV pin |

Conclusion: a **pure weights swap**. `DSPARK_MODEL` (+ an optional `DSPARK_REVISION` pin)
is the only configuration delta; the pinned revision here is `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.

## The two offline-cache traps (both patched in `bringup/07`)

The upgrade failed its first two boots on a cache-layout subtlety, not on the model:

1. **`hf download --revision <sha>` never writes `refs/main`.** The kit serves fully
   offline (`HF_HUB_OFFLINE=1`); vLLM resolves the default revision `main` through that
   ref at startup, and without it huggingface_hub raises
   `LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder … and
   outgoing traffic has been disabled`. Prior deployments never saw this because
   unpinned downloads create the ref implicitly.
2. **A hand-written ref must not end in a newline.** The first manual fix wrote
   `refs/main` with `echo` (41 bytes); huggingface_hub 1.24's offline resolver rejects
   anything but the exact 40-byte sha (`printf '%s'`). The symptom is the same
   `LocalEntryNotFoundError`, one layer later.

`bringup/07-download-weights.sh` now recreates `refs/main` (newline-free) after every
download and validates the snapshot, and both `07`/`08` derive the cache path from
`$DSPARK_MODEL` instead of hardcoding the repo name. Verification trick that saved a
boot cycle: run vLLM's exact lookup inside the serving image before restarting —
`docker run --rm --entrypoint python3 -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/huggingface
-v <hf-cache>:/cache/huggingface <image> -c "from huggingface_hub import snapshot_download;
print(snapshot_download('<repo>'))"`.

## Never overlay the checkpoint's own encoder

The checkpoint ships its own copy of the chat encoder at `encoding/encoding_dsv4.py`
(deepseek-ai/DeepSeek-V4-Flash-0731 @ `9e165c30`). **Do not install it over vLLM's vendored
encoder.** The checkpoint's copy corrupts tool-call arguments when `arguments` arrives as a
dict instead of a JSON string — it wraps them under a spurious `arguments` key, and the model
then *imitates the corruption* on every later turn (reported as
[MiaAI-Lab issue #21](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/21),
open). This kit is not exposed: it never installs the checkpoint encoder, and the vendored
0.26 copy (`--tokenizer-mode deepseek_v4`) carries the isinstance fix. The warning matters
only if you're tempted to "sync" the checkpoint's trust-remote-code files into the image —
don't; the vendored copy is the corrected one.

## A/B gate (same pair, same image, same-day preview baseline)

| Metric | Preview (DSpark) | 0731 | Verdict |
|---|---|---|---|
| Composite eval | 100/100 | 100/100 | tie |
| C8 aggregate tok/s (6 batches) | 94.2 best / ~89.3 mean | 93.0 best / ~89.2 mean | **tie (noise)** |
| C1 single-stream tok/s (6 batches) | 36.1 best / ~35.8 mean | 34.2 best / ~32.7 mean | −5…−8% |
| Acceptance — eval workload | 0.788 | 0.796–0.823 | **up** |
| Acceptance — nprobe battery | 0.842 | **0.875** | **up** |
| Needles @944,471 tok + 200K | 3/3 HIT | 3/3 HIT | tie |
| Garble / markup leaks | 1.00 clean | 1.00 clean | tie |
| KV pool (pinned bytes) | 2,948,751 tok | 2,948,751 tok | unchanged |
| Smoke: thinking in `message.reasoning`, tool-call parse | ✓ | ✓ | tie |

Promotion decision: throughput parity at concurrency, a single-digit single-stream dip,
measurably better draft acceptance, and a large agentic-quality gain → **promoted**.

## One 0731 behavior worth knowing

0731 **deliberates much longer on length-constrained or very long single-shot prompts**:
observed on the eval's "200-word story" concurrency probe and a "write a 500+-line Raft
module" stress prompt — it word-counts sentences and iteratively re-drafts *inside its
thinking* (coherently; 16K+ chars, six full module rewrites in one completion) until a
tight `max_tokens` truncates the visible answer to empty. This is not garble (no
repetition loops, no markup leaks; garble score 1.00) and it does not affect
tool-use/agentic traffic — but:

- give requests a generous `max_tokens` (the same trap class as `docs/06`'s thinking
  budget note),
- on **c8r-tbfix**, send a request-level `thinking_token_budget` when you need a hard
  think cap without dropping `reasoning_effort` (see [docs/06](06-reasoning-mode.md)), and
- for long-form outputs, prefer chunking or explicit structure over one giant ask.

A 2026-08-15 A/B showed this trait is **amplified** by vLLM #50580 (already in the
production pin): `"high"` now emits the old maximum-effort prefix. Switching the
server to `low` unblocks python30 and fails the constrained short-story probe.
Keep `DSPARK_REASONING_EFFORT=high`. Details: [docs/16](16-post-pin-qualification.md).

The eval's concurrency3 probe dropped its numeric length constraint for exactly this
reason — the probe exists to test concurrency + garble, not constraint compliance.

## Rollback

Weights-only, instant: flip `DSPARK_MODEL` back to `deepseek-ai/DeepSeek-V4-Flash-DSpark`
(and `DSPARK_REVISION` to the preview sha, or empty) and re-run `bringup/07`+`08` if the
preview isn't still resident. The served name doesn't change, so clients, the eval, and
the bench harness need nothing.
