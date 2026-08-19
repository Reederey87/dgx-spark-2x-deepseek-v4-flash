# c8r-tbfix indexer-scoring fix (upstream PR #52492)

A small, reproducible runtime layer over `vllm-dspark-runtime:v0261-main-c8r-tbfix`.
It changes only `vllm/models/deepseek_v4/attention.py`: the overlay0261 copy plus
upstream `292187dd8c` ("Keep indexer scoring in breakable graphs"). The short-context
shortcut in `DeepseekV4Indexer.forward`

```python
if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
```

gains `and not torch.cuda.is_current_stream_capturing()`.

Why it matters here: the shortcut is correct eagerly but not under breakable
PIECEWISE capture — capture runs with short dummy metadata, so the unguarded branch
is **baked into the captured graph**. On replay, any request whose cached prefix
exceeds 2048 tokens (512 candidates at compress_ratio 4) skips learned indexer
scoring and attends only to candidates 0..511 — silently, with no error. This kit's
exact config (MRV2, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, 1M ctx, prefix caching) meets
every activation condition. Upstream validated the fix on 2× B200 TP2+EP DSpark-7
with FULL_AND_PIECEWISE: the >2K-prefix wrong-token repro and the #52448 length-cap
runaway signature both clear.

Build on the head node without changing services:

```bash
bash patches/vllm-0261-main-ixfix/build-and-distribute.sh
```

Add `--distribute` to send the resulting image over rail 1 and assert identical
image IDs on both nodes. Tag `vllm-dspark-runtime:v0261-main-c8r-tbfix-ixfix`; give
it its own `-ixfix` cache roots (a source-file change is invisible to the
compile-cache key — see `runtime/cluster.env.example` §4).

Promoted 2026-08-18. Correctness-only by construction (a host-side `if`), so the bar
was non-inferiority: needles 3/3 at 200K/944K on both arms (the first long-context
revalidation on this base), T2 prompt logprobs flat, and warm paired C1/C8
non-inferior (C1 −0.19%, C8 +0.34%, energy NI). Receipt: docs/17. Rollback is the
unchanged `v0261-main-c8r-tbfix` image plus its `-tbfix` cache roots.
