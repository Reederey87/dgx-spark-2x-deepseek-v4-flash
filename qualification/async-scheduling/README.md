# Async-scheduling qualification lane

Production deliberately remains on `--no-async-scheduling`. This directory is
the reusable W6 load-and-safety harness for testing an async derivative; it is
not permission to remove the production flag.

Why the lane is still on HOLD: the first production-shaped run completed its
requests, but the post-request safety tail observed swap pressure on the
unified-memory hosts. A faster request path is not a win if it leaves the next
workload with a poisoned memory state.

## Workloads

- `decode-heavy`: 48 requests, 512 input tokens and 2,048 output tokens,
  concurrency 8.
- `production-mix`: 24 short, 12 medium, and 4 long requests on a frozen seeded
  arrival trace, concurrency 8.

`w6-async.py` creates an exact-token, cache-resistant manifest through the
server's `/tokenize` endpoint. `power-sample.py` records both-node energy.
`run-w6.sh` adds the operational safety envelope:

- at least 2.25 GiB `MemAvailable` on each node before admission;
- immediate abort below 1 GiB;
- abort on a 4,096-page swap spike or three samples above 256 pages;
- abort on service loss, fatal GPU/engine logs, or unreadable telemetry;
- a 60-second post-request tail so delayed swap is charged to the arm that
  caused it.

## Run it safely

Open a real maintenance window, stop normal traffic, make rollback ready, and
disable the watchdog and metrics timers so they cannot hide a failure. Then:

```bash
W6_TEST_WINDOW=1 bash qualification/async-scheduling/run-w6.sh \
  async-candidate decode-heavy 1
W6_TEST_WINDOW=1 bash qualification/async-scheduling/run-w6.sh \
  async-candidate production-mix 1
```

Repeat A/B/A with the same manifests and accept only if both workload profiles
are non-inferior, all requests finish, energy per output token does not regress,
and every safety tail stays clean. Re-enable the timers and normal traffic only
after restoring the chosen profile and verifying both services.
