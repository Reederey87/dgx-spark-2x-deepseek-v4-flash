# C128A adaptive-packing revert (upstream PR #51318)

A small, reproducible runtime layer over
`vllm-dspark-runtime:v0261-main-c8r-tbfix-ixfix`. It changes only
`vllm/models/deepseek_v4/sparse_mla.py`: the overlay0261 copy plus upstream
`edd4c8176c` ("Revert adaptive C128A metadata packing" — reverts #50004).

Why it matters here: #50004 is in the pinned base and the overlay shipped it
verbatim. The C128A metadata builder runs eagerly *before* FULL-graph replay and,
with the adaptive width, wrote packed rows of batch-dependent width
(`next_power_of_2(max_seq_len // 128)` clamped to [128, 8192]) while the
graph-captured sparse decode consumer keeps its capture-time row stride. Rows ≥1
then read stale offsets → **wrong top-k indices, silently** (upstream issue #52448:
DSpark + FULL_AND_PIECEWISE breakable graphs + concurrency ≥10 → requests loop
inside reasoning to `max_tokens`). The revert restores fixed capacity-width rows
(`buffer[:n]` slices + real `stride(0)`), so the eager writer and the captured
reader always agree.

Build on the head node without changing services:

```bash
bash patches/vllm-0261-main-c128arev/build-and-distribute.sh
```

Add `--distribute` for head→worker transfer with image-ID parity. Tag
`vllm-dspark-runtime:v0261-main-c8r-tbfix-ixfix-c128arev`; own `-c128arev` cache
roots (see `runtime/cluster.env.example` §4).

Promoted 2026-08-18. Gate + warm re-arm: needles 3/3 both arms, long-generation
battery clean, T2 flat, **C1 +2.60%/+2.10% (repeatable small win — on GB10 the
fixed-capacity-width rows beat #50004's adaptive packing, whose ~1% claim was
unstated hardware)**, C8 tie (+0.41%/−0.43% across two windows), energy NI.
Receipt: docs/17. Rollback is the resident `-ixfix` image plus its `-ixfix` roots.
