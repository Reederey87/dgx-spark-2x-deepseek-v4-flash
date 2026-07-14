# 07 — Observability & warm-up

Everything in [04](04-serving-and-systemd.md) keeps the two nodes *serving* unattended. This doc
covers the layer above that: a specific latency fix that no other gate can see, a lightweight
watcher that catches it drifting at runtime, optional alerting, a post-restart warm-up, and the
enriched evaluator with a single composite score.

---

## The prefill head-of-line (HoL) fix

Long prefills and short chats share one running batch. Without a cap, a single multi-hundred-K
prefill monopolizes the prefill budget for the whole time it chunks through, and a short request
that arrives behind it waits for the *entire* prefill before it gets its first token — its TTFT
regresses from sub-second to tens of seconds. `--long-prefill-token-threshold` **caps how many
tokens of a long prefill run in any one step**, so short requests interleave and keep near-idle
TTFT even while a big prefill is in flight.

It rides the same config chain as every other knob:

```
cluster.env  ──(render-env.sh)──►  .env.dspark  ──►  docker-compose.dspark.yml  ──►  vllm serve …
LONG_PREFILL_TOKEN_THRESHOLD=4096         (per-node, 600)      --long-prefill-token-threshold ${…:-0}
```

Default is **`4096`**; see the tunables table in the [README](../README.md).

### Caveat A — it reverts to *off* silently

The compose passes `--long-prefill-token-threshold ${LONG_PREFILL_TOKEN_THRESHOLD:-0}`. The
fallback is **`0`**, which means *disabled*. So any drift that drops the value — an empty/unset
line in `cluster.env`, a `render-env.sh` that stops emitting it, a compose edit that removes the
flag — doesn't error. It **silently reverts to `0`** and short-request TTFT quietly regresses.

The reason this is dangerous is that **no correctness gate catches it**: the eval's
correctness/needle/garble/throughput probes and the inference watchdog are all **TTFT-blind**.
The tokens are still correct; they just arrive late for short requests stuck behind a long
prefill. Two guards below close that hole.

### `preflight.sh` — the static plumbing assert

`preflight.sh`'s `check_hol_threshold()` runs as `ExecStartPre`, **before** `render-env.sh` and
before the container exists, so it can't inspect the live serve command. Instead it asserts the
fix survives the whole plumbing chain, statically, in three points:

1. `LONG_PREFILL_TOKEN_THRESHOLD` is a **positive integer** in `cluster.env` (unset / `0` /
   negative / non-numeric all fail — they'd make the fix inert or break `vllm serve` at start).
2. `render-env.sh` still **emits** `LONG_PREFILL_TOKEN_THRESHOLD` into the rendered `.env.dspark`.
3. `docker-compose.dspark.yml` still **passes** `--long-prefill-token-threshold` (comment-aware —
   a commented-out line doesn't count).

It is **non-fatal**: any broken point prints a `preflight WARN: …` to the journal and boot
continues (a latency regression must never keep the cluster down). When all three hold it prints:

```
preflight ok: prefill HoL threshold wired (LONG_PREFILL_TOKEN_THRESHOLD=4096)
```

---

## The observability watcher — `metrics-watch.sh`

A lightweight loopback watcher for the head, run by a **systemd user timer (~45 s)** on the
**head node only**. It is **read-only** and **loopback-only**: it scrapes `127.0.0.1:8000/metrics`
and inspects the running container. It **never fails hard** — a watcher has to survive transient
scrape misses. It does two jobs.

**1. A live Caveat-A check.** It reads the **running** container's serve command
(`docker inspect … .Config.Cmd`) and WARNs if it no longer carries a positive
`--long-prefill-token-threshold`. This is the check `preflight.sh` structurally *cannot* do,
because preflight runs before the container exists — here we assert the fix on the process that
is actually serving.

**2. Interval-delta metrics.** It scrapes `/metrics` and hands the text to
`metrics-watch-analyze.py`, which, against a small state file from the previous tick, computes
**interval** deltas (not cumulative-since-boot): mean TTFT, mean e2e latency, and spec-decode
acceptance for the interval, plus current waiting / KV util / preemptions, the interval
**prefix-cache hit rate** (`prefix_hit_iv`, log-only — worth watching because of the documented
MTP × prefix-cache interaction, vLLM #38182), and the number of **long prefills (>30 s)** that
landed in the interval (`long_prefills_iv`). It WARNs on any threshold breach, and additionally
WARNs when requests are waiting specifically on **KV capacity**
(`num_requests_waiting_by_reason{reason="capacity"}` — the direct saturation signal). A HoL
revert shows up here too — interval-mean TTFT spikes if short requests start queueing behind
long prefills mid-run. When **≥ 2 long prefills overlap one interval** it emits a **log-only
note** (`capacity-b`): that's the one HoL case the threshold lever can't close (two long
prefills together refill the whole token budget), so the note counts how often it actually
happens before anyone builds client-side request shaping for it.

The analyzer also correlates against the watchdog's `.watchdog-probe.state`: intervals whose
only traffic was the watchdog's own 1-token canary are marked **canary-only** (`ttft_iv_ms=n/a`
+ `ttft_probe_ms=…`) instead of being reported as user latency — otherwise an idle box looks
permanently "busy" at canary latency and TTFT alerts fire on the canary itself.

Thresholds are env-tunable:

| Env | Default | WARNs when |
|---|---|---|
| `TTFT_WARN_MS` | `3000` | interval-mean TTFT exceeds it (a HoL revert sends short-req TTFT to tens of seconds) |
| `WAIT_WARN` | `5` | sustained requests waiting (saturation / head-of-line) |
| `KVUTIL_WARN` | `0.95` | KV pool pressure (OOM risk) |
| `ACCEPT_WARN` | `0.30` | spec-decode interval acceptance floor (MTP health) |
| — (fixed) | `>0` | any request waiting with `reason="capacity"` (KV-pool saturation, the pre-cursor to the watchdog's saturation triage) |
| `NOTIFY_COOLDOWN` | `900` | seconds between reminders for a persistent condition (see alerting) |

It deliberately **avoids a full Prometheus/Grafana TSDB** — a time-series database, scraper, and
dashboard stack is overkill for a single loopback box. WARNs go to journald and a rotating log
(`logs/metrics-watch.log`); the interval-delta math replaces what you'd otherwise use `rate()`
for. To roll the watcher back:

```bash
systemctl --user disable --now vllm-metrics-watch.timer
```

### Telegram alerting (optional)

WARNs route to Telegram **only** if a git-ignored `notify.env` (`chmod 600`) supplies
`TG_BOT_TOKEN` + `TG_CHAT_ID` (and, for forum-topic supergroups, an optional `TG_THREAD_ID`).
**Absent ⇒ strict no-op** — exactly the prior journald-plus-log behavior, nothing sent.

Alerting is deduped so it doesn't spam a persistent condition:

- Alerts key on the **set** of active WARN categories. A *new* set fires immediately.
- The **same** set re-fires only every `NOTIFY_COOLDOWN` (default `900 s` = 15 min).
- A one-shot **"recovered — all clear"** note is sent once all WARNs clear.
- Notification state advances **only on a delivered message** — if Telegram is briefly
  unreachable (or creds are absent) the run retries next tick rather than losing the alert or
  starting a premature cooldown.

Setup:

```bash
# On the head node, in the kit's runtime/ dir (~/dgx-cluster/runtime):
# 1. Create a bot with @BotFather and copy its token.
# 2. Get the destination chat id (e.g. message the bot, then read getUpdates,
#    or use a chat-id helper bot). Group/supergroup ids are negative.
cp notify.env.example notify.env
$EDITOR notify.env          # TG_BOT_TOKEN, TG_CHAT_ID (+ optional TG_THREAD_ID)
chmod 600 notify.env
```

The repo ships `notify.env.example`; the real `notify.env` is git-ignored and never committed.

### Optional Xid hardware-fault monitor

`vllm-dsv4-xid-monitor.service` is installed disabled on both nodes. When enabled it follows
the kernel journal continuously and classifies NVIDIA Xid lines. Catastrophic codes
48/79/94/95/119/140/154 trigger immediate current/previous-boot kernel-log capture and an
optional Telegram alert. Other Xids are log-only.
Repeated catastrophic events are debounced per Xid code for 300 seconds by
default (`XID_NOTIFY_COOLDOWN_SEC`), preventing one fault from filling the
incident directory or rate-limiting alerts while still allowing a different
catastrophic code through immediately.

This monitor has a strict safety boundary: it **never** starts, stops, or restarts a vLLM unit.
A GPU that has fallen off the bus or suffered a GSP timeout is a hardware incident, not the
software wedge handled by `watchdog.sh`; automatic pair-bouncing would only consume the units'
start limit. Enable it on each node after reviewing the behavior:

```bash
systemctl --user enable --now vllm-dsv4-xid-monitor.service
bash ~/dgx-cluster/runtime/xid-monitor.sh --test \
  'NVRM: Xid (PCI:0000:0f:00): 119, synthetic classification test'
```

`--test` suppresses log capture and outbound notification, so it is safe in CI and on a live node.

---

## The readiness warm-up — `warmup.sh`

After the head (re)starts, the first real request pays a cold-path tax. `warmup.sh` absorbs it.
It is wired as a **non-fatal, backgrounded `ExecStartPost`** on the head unit — backgrounded so
it never delays the unit reaching `active` or eats into `TimeoutStartSec`, and purely additive so
a failure here can **never** affect serving. Once `/health` returns 200 (bounded wait) it fires
four probes against the loopback API:

1. **One plain chat** — warms the decode / first-token path.
2. **One `tool_choice=auto` call** — warms the model's tool-**parser** path (what the downstream
   client actually hits after a self-heal).
3. **One >4096-token prefill** — compiles the **long-prefill chunk-metadata Triton kernels**
   (`_pack_topk_routes_*`, `_build/_compute_prefill_*_metadata`). Without it, vLLM's JIT monitor
   shows these compiling **during the first real long prompt** 10–30 min post-boot — a latency
   spike on exactly the request you care about.
4. **One `temperature=1.0` sampling call** — the spec-decode **rejection-sampling kernels**
   (`sample_recovered_tokens`, `rejection_random_sample`) are only reachable with temperature>0;
   the greedy probes above never touch them.

The Triton JIT cache is **persistent** (`TRITON_CACHE_DIR` lives under the hf-cache bind), so
these compiles are per-new-*shape*, not per-boot — the warm-up just guarantees the common shapes
are compiled before real traffic hits them.

It's a **parser** warm-up, not an FSM warm-up, on purpose: with `tool_choice=auto` vLLM does
**not** compile an xgrammar FSM — the parser extracts tool calls from the model's free text, so
that's the path worth priming. (A schema-constrained `tool_choice` that compiled a grammar would
be a different code path.) Everything logs to `logs/warmup.log`.

---

## The eval composite score — `eval-cluster.sh`

Beyond the correctness / long-context needle / concurrency probes it always ran, `eval-cluster.sh`
now adds a **tool-call correctness** probe (asserts a well-formed `get_weather` call with the
right argument) and collates a single **0–100 composite score**. The composite weights four
components (renormalized over whichever are available):

| Component | Weight | What it measures |
|---|---|---|
| correctness | 0.50 | pass-rate across the functional probes |
| garble | 0.15 | clean/garbled ratio over every captured response |
| latency_slo | 0.25 | TTFT-idle and p95 latency against fixed SLOs |
| spec_decode | 0.10 | draft-acceptance rate vs a floor |

It also reports perf metrics that continuously re-validate the deployment — most importantly
**streaming TTFT idle vs under a long prefill**, which is the live, ongoing check that the HoL fix
is doing its job (idle and under-load TTFT should stay close; a big gap means the cap regressed).
Alongside it: draft-acceptance rate and mean accepted length, the **prefill/decode tok-per-s
split**, queue depth sampled under concurrency, and **p50/p95/p99** latency over a small burst.

The two slow probes are opt-out via env guards so a quick correctness-only run stays fast:

```bash
SKIP_TTFT=1 SKIP_LATENCY=1 bash runtime/eval-cluster.sh   # skip the streaming-TTFT and latency-burst probes
```

Reference composite on one pair is high-90s with the HoL fix wired; the number is a health
signal, not a guarantee — yours will vary.
