# Thinking-budget fix and the guarded async lane (2026-08-12/13)

This update has one production promotion and one deliberately unpromoted
experiment.

## Promoted: `c8r-tbfix`

The pinned sampler correctly contains `ThinkingBudgetState.apply()`, but its
fast-path decision did not count an active `thinking_token_budget` as logits
processing. Default sampling could therefore bypass the call and ignore the
budget.

The new runtime derivative adds that missing predicate. It is intentionally a
small layer over c8r, and the patcher refuses to operate unless it sees the
exact expected source block. No default budget is configured by the server.

The lane passed budget-unset, 4K, and 8K request probes; C1/C8
non-inferiority; Python, coding, and tool correctness; speculative-acceptance;
and cross-node image-ID parity. It was promoted on 2026-08-12.

All generated caches move with the derivative:

```text
vllm-cache-tbfix
triton-cache-tbfix
tilelang-tbfix
flashinfer-tbfix
```

Rollback is the unchanged `vllm-dspark-runtime:v0261-main-c8r` image and its
`-c8r` cache roots. Do not mix the roots across these images.

## Not promoted: async scheduling

The serve command still includes `--no-async-scheduling`. Omitting the flag is
not an off spelling on this vLLM line: the tri-state default enables async.

Async remains an interesting performance candidate, so the kit now includes a
deterministic W6 harness under `qualification/async-scheduling/`. Its first
production-shaped exercise exposed the reason this needs a stronger gate:
request completion alone looked acceptable, but the post-request safety tail
saw swap activity on the unified-memory nodes. The candidate is therefore on
HOLD, not rejected and not production-ready.

The next valid test is a rollback-ready A/B/A maintenance window with identical
manifests, clean memory admission, both W6 workload profiles, both-node power,
and a clean 60-second tail after each arm. The test must stop immediately on
swap, low memory, service loss, fatal GPU logs, or missing telemetry.

The harness also requires 60 continuous quiet seconds before W6 begins. This
prevents delayed swap writeback from an opening evaluation, boot, or prior arm
from being misattributed to the next workload; a critical spike stops the run
before any W6 request is sent.

## Operations hardening

`runtime/metrics-watch.sh` now separates harmless parked swap from active
thrash. It logs parked pages without paging operators, warns after three
consecutive samples above 256 moved pages, and warns immediately above 4,096
pages. This matters on GB10 because CPU and GPU share the same memory budget.
