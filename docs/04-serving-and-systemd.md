# 04 — Serving & systemd

This is how the two nodes become one engine and stay one engine unattended: the config flow,
the TP=2 rendezvous, the systemd user units, and the two guards that make cross-node
coordination survive reboots and lost peers.

## Config flow — one source, one rendered file

```
cluster.env  ──(render-env.sh <head|worker>)──►  .env.dspark  ──►  docker-compose.dspark.yml
(you edit this)      per-node, git-ignored, chmod 600            (the full vLLM serve argv)
```

- **`cluster.env`** is the single source of truth. Every script sources it.
- **`render-env.sh <head|worker>`** bakes it into a node-local **`.env.dspark`** (git-ignored,
  `chmod 600`) that `docker compose` reads. It sets the per-role bits: `NODE_RANK` 0/1,
  `HEADLESS` for the worker, and `VLLM_HOST_IP` to that node's rail-1 IP.
- **The full vLLM serve argv lives ONLY in `docker-compose.dspark.yml`.** Don't reinvent it in
  the units — they just run compose.

## The serve profile (sweet spot)

Set in `cluster.env`; see the tunables table in the [README](../README.md). The defaults are
tuned for a **dedicated** 2× GB10 pair:

| Knob | Value | Note |
|---|---|---|
| `MAX_MODEL_LEN` | `1048576` | True YaRN ceiling (65536×16). First rung of the OOM ladder. |
| `MAX_NUM_SEQS` | `12` | Concurrent streams. |
| `GPU_MEMORY_UTILIZATION` | `0.85` | ~2.8M-token KV pool on one pair; `0.80` ≈ 2.0M when co-locating. |
| `MTP_NUM_TOKENS` | `3` | DSpark speculative draft length — the garble fix (see [03](03-model-and-features.md)). |

Key serve argv facts baked into the compose: `--tensor-parallel-size 2`,
`--kv-cache-dtype nvfp4_ds_mla`, `--block-size 256`, `--speculative-config` (method `dspark`,
`draft_sample_method: probabilistic`), and the multi-node flags below.

## TP=2 rendezvous — native `mp`, no Ray

vLLM's native multiprocessing backend is used (`--distributed-executor-backend mp`) — **no
Ray**. The two ranks find each other over the QSFP fabric:

- Both run `vllm serve … --nnodes 2 --node-rank N --master-addr $MASTER_ADDR
  --master-port $MASTER_PORT`.
- `MASTER_ADDR` **must equal `HEAD_R1`** (`192.168.177.10`) and `MASTER_PORT` is `25000` — the
  rendezvous rides rail 1.
- The **worker (rank 1) starts BEFORE the head (rank 0).** The headless worker waits for a
  master to appear; the head rendezvouses to it. Start ordering is enforced by the scripts
  and units, not left to chance.

## systemd user units

The units run as **user** services under the `nvidia` cluster user (with linger, so they
start at boot without a login):

| Unit | Node | Role |
|---|---|---|
| `vllm-dsv4-head.service` | head | rank 0 — runs compose, serves the API, `Restart=always`. |
| `vllm-dsv4-worker.service` | worker | rank 1 (`--headless`), `Restart=always`. |
| `vllm-dsv4-watchdog.service` + `.timer` | head | inference-level self-heal, every 5 min. |

The units reference **`%h/dgx-cluster`** — so the last path component of `KIT_DIR` **must be
`dgx-cluster`** for the units to find `preflight.sh`, `render-env.sh`, and the compose file.
Each unit's `ExecStartPre` runs `preflight.sh <role>` then `render-env.sh <role>`, then
`ExecStart` is `docker compose … up --exit-code-from vllm-dspark`.

`install-services.sh` (from the control host) rsyncs the runtime kit to `$KIT_DIR` on both
nodes (excluding docs/license/example config), installs the units into
`~/.config/systemd/user/`, and verifies linger — but it does **not** enable or start them.
`cluster-enable.sh` does that.

### Why cross-node coordination is script-level

systemd dependencies **cannot cross user managers or nodes** — the head unit can't declare
"After the worker unit on the other box." So all cross-node/cross-user ordering lives in
`preflight.sh` and the control-host scripts, not in `[Unit]` `After=`/`Requires=`.

## `preflight.sh` — bounded boot dependencies

Runs as `ExecStartPre` on the node itself. Every wait is **bounded** (the head unit's
`TimeoutStartSec` is set to exceed the worst-case sum). On the **head**, in order:

1. rail-1 QSFP IP assigned → 2. Docker answering → 3. `/dev/infiniband` present →
   4. image present → 5. weights present in the token-free cache → 6. peer reachable over QSFP
   → 7. **worker unit active** — and if the worker is `failed`/`inactive`, preflight
   **resets and starts it over QSFP** (drill-proven: a flapping window can exhaust the
   worker's `StartLimit`, otherwise leaving the head waiting forever).

It also removes any stale `vllm-dsv4` container before compose runs, and (optionally) enforces
the memory-pool mutual-exclusion below. Its non-fatal `check_hol_threshold()` guard, the
runtime observability watcher, the readiness warm-up (`ExecStartPost` on the head), and optional
Telegram alerting all live in [07-observability-and-warmup.md](07-observability-and-warmup.md).

## `watchdog.sh` — inference-level self-heal

`/health` reports that the API process is alive — but **a lost TP peer makes `/health` lie
while inference hangs** (the API survived, the engine didn't). So the watchdog (5-min timer)
tests *real* inference:

- If `/health` is **down**, exit — startup isn't finished, not our problem.
- If `/health` is **up** but a **1-token completion times out**, it first **triages saturation
  vs wedge** before doing anything destructive (added after two production incidents where the
  bounce killed 3–4 in-flight long-context generations that were merely KV-starved, not hung):
  - Scrape `/metrics`. If `kv_cache_usage_perc < 0.95` (or the scrape fails) → treat as a
    **wedge** → bounce.
  - If KV is **saturated (≥ 0.95)** → **retry the 1-token canary with a 30 s cap**. Success
    means admission recovered → no bounce. (Admission recovery is the real health signal — an
    aggregate token counter can't prove the canary path works while other requests drain.)
  - If the retry also fails → compare `generation_tokens_total` across the 30 s window. Still
    advancing = **saturation**: log it, count it in `.watchdog-satn.state`, and **don't bounce**
    — vLLM's V1 scheduler never preempts a running request to admit a capacity-blocked waiter,
    so a full pool starves new requests while old ones finish; killing everything only makes it
    worse. After **3 consecutive** saturation verdicts (~15 min) it escalates to a bounce anyway
    (livelock backstop). Counter frozen = **wedge** → bounce.
- The **ordered pair-bounce** itself is unchanged:
  1. **Stop the head FIRST** — its stale rendezvous store on `:25000` must be gone before the
     worker restarts, or the fresh worker joins the dead group and zombifies.
  2. **Restart the worker** (headless; waits for a master).
  3. **Start the head** — preflight re-checks the worker, then re-rendezvouses.

It `reset-failed`s throughout, and the 5-min timer period is itself the rate limiter. The unit
carries an explicit `TimeoutStartSec=600` (oneshot disables the default; worst-case probe +
triage + bounce is ~4–5 min). Each cycle records exactly **one** probe outcome to
`.watchdog-probe.state` (the metrics watcher's canary logic depends on that invariant).

## Daily ops

| Command | What it does |
|---|---|
| `start-cluster.sh` | Start worker unit, then head unit, then poll `/health` (worker-first). |
| `stop-cluster.sh` | Stop head first, then worker; remove containers; confirm the API port is released. |
| `cluster-enable.sh` | Enable units for boot + start + poll `/health`. Also handles `CONFLICTING_SERVICE` (below). |
| `cluster-disable.sh` | Stop + disable units; re-enable/restart `CONFLICTING_SERVICE` if configured. |
| `eval-cluster.sh` | Correctness + throughput + a long-context needle probe through the loopback API. |
| `metrics.sh` | `nvidia-smi`, `free -g`, and vLLM `/metrics` (running/waiting/KV usage) from both nodes. |

### Optional memory-pool mutual exclusion

If a node also runs a **single-node** model service that would fight the cluster for the
shared unified pool, set that service's systemd user unit name in `CONFLICTING_SERVICE`.
Then `preflight.sh` **refuses to start the head** while it's active (override with `FORCE=1`),
`cluster-enable.sh` **stops + disables** it before cutover, and `cluster-disable.sh`
**re-enables + restarts** it on rollback. Leave `CONFLICTING_SERVICE` empty to disable the check.

## Reference behavior (observed on one pair)

- **Startup:** ~5–10 min (cold ~9.5 min, warm ~5 min).
- **KV pool:** ~2.8M tokens @ util 0.85; ~2.0M @ 0.80.
- **Eval:** 5/5 correctness; a **147K-token needle retrieved in ~86 s**; **~48 tok/s**
  aggregate at concurrency 3; single-stream **~22–34 tok/s**.
- **Resilience:** unattended self-heal from a killed worker in **~11 min**; cold
  reboot-to-serving in **~6 min**.

These are observations on one 2× GB10 pair, not guarantees — yours will vary.
