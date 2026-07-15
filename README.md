# dgx-spark-2x-deepseek-v4-flash

Reproducible kit to serve **DeepSeek-V4-Flash-DSpark** (284B MoE / ~13B active, NVFP4-KV,
up to 1M context) on a **2× NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64, CUDA 13)
cluster with **vLLM tensor-parallel = 2** over a single **QSFP 200GbE** cable. The two
Sparks act as one inference engine; the OpenAI-compatible API is served on the head node's
loopback (`127.0.0.1:8000`).

This repo is **orchestration and documentation only**. It vendors no upstream source: the
serving image is *built from* a pinned community base and the weights are *pulled from*
Hugging Face at deploy time. As of **2026-07-15 the default lane is vLLM 0.25.1** — a
digest-pinned `anemll/dspark-vllm-gx10` GB10 base plus one thin upstream-fix layer (vLLM
PR #47356). The prior vLLM 0.21.x lane stays documented and rollback-able. The full lane-change
rationale, config deltas, hardening pass, and open gaps are in
[docs/08](docs/08-optimization-and-vllm-025.md). See [NOTICE](NOTICE) for upstream attribution.

> ⚠️ **Experimental.** The DSpark / GB10 serving stack is fast-moving, largely
> single-author, and partly dependent on prebuilt (non-source-buildable) kernels and
> images. Treat this as experimental and validate on your own hardware behind each
> upstream's own smoke/sanity tests. All performance numbers here are **observations on
> one 2× GB10 pair — not guarantees. Yours will vary.**

---

## Architecture

```
                       control host (your laptop/workstation)
                       runs the numbered scripts over SSH; not in the data path
                                     |
                 ssh $HEAD_HOST      |      ssh $WORKER_HOST   (mDNS / SSH-alias names)
              ┌──────────────────────┴──────────────────────┐
              |                                              |
   ┌──────────────────────┐                      ┌──────────────────────┐
   │  HEAD  (rank 0)       │                      │  WORKER (rank 1)      │
   │  DGX Spark · GB10     │                      │  DGX Spark · GB10     │
   │  ~121 GiB unified mem │                      │  ~121 GiB unified mem │
   │                       │                      │                       │
   │  vLLM serve           │   QSFP 200GbE cable  │  vLLM serve --headless│
   │  --node-rank 0        │◄════════════════════►│  --node-rank 1        │
   │                       │  rail 1: 192.168.177 │                       │
   │  OpenAI API           │  rail 2: 192.168.178 │  (no API listener)    │
   │  127.0.0.1:8000 ◄──┐  │  MTU 9000, dual-twin │                       │
   └────────────────────┼──┘   RoCEv2 / NCCL      └──────────────────────┘
                        │        TP=2, mp backend, rendezvous on HEAD_R1:25000
              your clients (loopback only by default)
```

One physical QSFP port enumerates as **two** PCIe "twin" netdevs (~100G each); using both
twins on two subnets gets the full ~200G. NCCL runs RDMA over both; the control/bootstrap
plane rides rail 1. TP=2 uses vLLM's native `mp` backend — **no Ray**. The **worker starts
before the head**; the head rendezvouses to `MASTER_ADDR:MASTER_PORT` (= `HEAD_R1:25000`).

---

## Quickstart

Everything except step 3 runs from a **control host** (any machine with SSH to both nodes).
Step 3 runs **on** each node. The repo is organized into `bringup/` (one-time, control-host
setup), `runtime/` (everything a node runs + the lifecycle/ops scripts), and `docs/`. The whole
tree is rsynced to each node *preserving that structure* — the units reference
`%h/dgx-cluster/runtime/…`.

```bash
# 1. Configure — this is the single source of truth for the whole kit.
cp runtime/cluster.env.example runtime/cluster.env
$EDITOR runtime/cluster.env                # set HEAD_HOST / WORKER_HOST (identity block)
export CONTROL_HOST_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"   # authorized on the nodes

# 2. Copy the kit to each node as your normal login user (the cluster user does not
#    exist yet). The on-node dir MUST be named dgx-cluster; keep bringup/ + runtime/ intact.
rsync -a ./ "$HEAD_HOST:~/dgx-cluster/"
rsync -a ./ "$WORKER_HOST:~/dgx-cluster/"

# 3. One-time node prep — runs ON each node with its role (the only sudo step).
#    Add --firmware FIRST if the two nodes' firmware differ (then reboot, re-run without it).
ssh -t "$HEAD_HOST"   'cd ~/dgx-cluster && bash bringup/00-node-prep.sh head'
ssh -t "$WORKER_HOST" 'cd ~/dgx-cluster && bash bringup/00-node-prep.sh worker'

# 4. Bring up the fabric + build + serve — all from the control host, in order.
bash bringup/01-verify-fabric.sh     # QSFP addressing, MTU 9000, RoCE up, jumbo ping both ways
bash bringup/02-setup-cluster-ssh.sh # node-to-node SSH over the QSFP rail IPs
bash bringup/03-build-nccl-tests.sh  # NCCL v2.30u1 + nccl-tests at sm_121, both nodes
bash bringup/04-run-nccl-bench.sh    # A/B the RDMA arms; put the winner in cluster.env (gate ≥15 GB/s)
bash bringup/05-build-image.sh       # build stage-c + the pinned FlashInfer #3615 safety layer
bash bringup/06-distribute-image.sh  # copy the image head → worker over QSFP; verify IDs match
bash bringup/07-download-weights.sh  # pull public weights to the head's token-free HF cache
bash bringup/08-distribute-weights.sh # rsync weights head → worker; verify file/byte parity
bash bringup/09-smoke-serve.sh       # foreground bring-up via compose; /health + a chat completion

# 5. Install the systemd user units, make the cluster primary, evaluate.
bash bringup/install-services.sh     # sync kit + install units on both nodes (does not start them)
bash runtime/cluster-enable.sh       # enable for boot + start (worker-first) + poll /health
bash runtime/eval-cluster.sh         # correctness + throughput + long-context needle probes
```

Call it (from the head node, loopback):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "deepseek-v4-flash-dspark",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 1024, "temperature": 1.0
      }'
```

Daily ops (all in `runtime/`): `start-cluster.sh` / `stop-cluster.sh` (bring the running cluster
up/down), `cluster-enable.sh` / `cluster-disable.sh` (toggle boot-persistence too), `eval-cluster.sh`,
`metrics.sh`. A `vllm-metrics-watch` user timer on the head runs a read-only observability
watcher (with optional Telegram alerts), and a non-fatal readiness warm-up primes the decode and
tool-parser paths after each head restart — see [docs/07](docs/07-observability-and-warmup.md).
An optional Xid monitor is installed disabled on both nodes; it captures hardware-fault evidence
and alerts but categorically never restarts vLLM.

---

## Tunables (the "sweet spot")

All live in `runtime/cluster.env`; `render-env.sh` bakes them into a node-local `.env.dspark`
that compose reads. The full vLLM serve argv lives only in `runtime/docker-compose.dspark.yml`.

| Knob | Default | Meaning |
|---|---|---|
| `MAX_MODEL_LEN` | `1048576` | Context ceiling. `1048576` is the model's true YaRN ceiling (65536×16). Higher boots but extrapolates past calibration. First rung of the OOM ladder. |
| `MAX_NUM_SEQS` | `12` | Concurrent streams. Drop toward `4` → `1` under memory pressure. |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill batch budget. |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Share of the ~121 GiB **unified** pool (~2.28–2.38M measured KV tokens on the 0.25.1 lane; ~2.45M on the prior 0.21.x lane). Drop to `0.80` if you co-locate other GPU processes on the head. Never exceed ~0.86. |
| `MTP_NUM_TOKENS` | `3` | DSpark speculative draft length. Probabilistic `5` was garble-clean but lost KV headroom and throughput; greedy `5` is unsafe. Keep `3`. See `docs/03`. |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | `96` | Keeps the spec-decode decode path graphed at concurrency. vLLM 0.25.1's formula is `MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1) × 2` (cap 512) — note the `× 2` vs the prior lane's `48`. If `MTP_NUM_TOKENS→5`, raise to `144`. See `docs/08`. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `1` | 0.25.1 auto-enables the breakable CUDA-graph path for DeepSeek-V4 (no `@support_torch_compile`). **Keep `1`** — `0` forces an unsupported `torch.compile` path that regressed spec-decode acceptance and KV here. See `docs/08`. |
| `TRITON_CACHE_DIR` | `/cache/huggingface/triton` | Triton kernel cache **must** live on the persistent HF-cache bind — under breakable-cudagraph vLLM's cache-redirect never runs, so an unset value falls to container-ephemeral `/root/.triton/cache` and cold-recompiles on every recreate. See `docs/07`, `docs/08`. |
| `SHUTDOWN_TIMEOUT` | `30` | vLLM engine grace period for in-flight requests after SIGTERM. The systemd units provide 90 s total stop headroom. |
| `GLOO_SOCKET_IFNAME` | `enp1s0f1np1` | Pins the CPU-side Gloo coordination group to the stable QSFP control rail; normally matches `NCCL_SOCKET_IFNAME`. |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `4096` | Caps each running long-prefill chunk so short requests interleave — the prefill head-of-line fix. `0`/unset disables it (short-request TTFT regresses under long prefills). See `docs/07`. |
| `DSPARK_REASONING` | `on` | Thinking mode (**production default** — what the Performance numbers were measured at). `on` = server-default thinking + `temp/top_p 1.0`; read the CoT from **`message.reasoning`** (not `reasoning_content`). `off` = non-think greedy (`temp 0`), fastest first token. **With `on`, give requests a generous `max_tokens` (≥1024)** or `content` comes back empty (the max_tokens trap). See `docs/06`. |
| `NCCL_IB_HCA` | `rocep1s0f1,roceP2p1s0f1` | RDMA data path. Default = both RoCE twins (~200G). `bringup/04-run-nccl-bench.sh` A/B-tests this. |

---

## Performance

Measured on **one** 2× GB10 pair. These are **observations, not guarantees** — yours will vary with silicon, thermals, firmware, and context.

### vLLM 0.25.1 lane (current default, `GPU_MEMORY_UTILIZATION=0.85`, `DSPARK_REASONING=on`)

Promotion-gate figures from 2026-07-15. A **full** `eval-cluster.sh` sweep (TTFT/latency/prefill/startup breakdowns) has **not** been re-run on this lane yet — the rows below are what was measured during promotion; the rest are marked TBD, not carried over from the prior lane.

| Metric | Result |
|---|---|
| **Composite eval score** | **100 / 100** — correctness 1.00 · garble-clean 1.00 · latency-SLO 1.00 · spec-decode 1.00 |
| Spec-decode acceptance | ~0.637–0.640 (draft len 3) |
| Throughput — single stream (C1) | at parity with the prior lane's baseline |
| Throughput — aggregate @ concurrency 8 | ~84.55 tok/s (vs. the prior lane's ~91.72 baseline, ~−7.8%; see [docs/08](docs/08-optimization-and-vllm-025.md)) |
| KV cache pool | **~2,279,532–2,376,103 tokens** @ util 0.85 (boot-to-boot variance; 93–97% of the prior lane's 2.45M — an open gap, see [docs/08](docs/08-optimization-and-vllm-025.md)) |
| TTFT (idle / under long prefill), latency percentiles, prefill throughput, startup | **TBD** — pending a fresh full `eval-cluster.sh` sweep on this lane |

### Prior 0.21.x lane (for reference, `GPU_MEMORY_UTILIZATION=0.80`, 2026-07-08)

| Metric | Result |
|---|---|
| **Composite eval score** | **98.2 / 100** — correctness 1.00 · garble-clean 1.00 · latency-SLO 1.00 · spec-decode 0.82 |
| Correctness | 7/7 functional probes + a **147K-token needle** (retrieved in ~83 s), zero garble |
| Throughput — single stream | ~31–47 tok/s (prose ~37, code ~47) |
| Throughput — aggregate @ concurrency 3 | ~55 tok/s |
| Prefill throughput | ~900 tok/s |
| Spec-decode acceptance | ~0.49 (per-position 0.71 / 0.48 / 0.29, draft len 3) |
| **TTFT — idle** | **~150 ms** |
| **TTFT — during a live ~130K-token prefill** | **~5.9 s with the head-of-line fix, vs ~59 s without** (≈10×; see [docs/07](docs/07-observability-and-warmup.md)) |
| Latency p50 / p95 / p99 (short-request burst) | ~620 / 630 / 630 ms |
| KV cache pool | ~2.0M tokens @ util 0.80; **2.45M measured** @ the 0.85 dedicated-pair default |
| Startup (worker → head, to `/health` 200) | ~5–10 min (warm ~6) |

`eval-cluster.sh` prints the composite plus the throughput/latency probes in one run; `SKIP_TTFT=1 SKIP_LATENCY=1` skips the two slow streaming probes.

---

## Documentation

| Doc | Covers |
|---|---|
| [docs/01-hardware-and-firmware.md](docs/01-hardware-and-firmware.md) | GB10 / ARM64 / unified memory, CUDA 13, and why **firmware parity** across the two nodes matters. |
| [docs/02-networking-nccl.md](docs/02-networking-nccl.md) | QSFP dual-twin fabric, RoCEv2/GID, MTU 9000, the NCCL A/B benchmark and its gate. |
| [docs/03-model-and-features.md](docs/03-model-and-features.md) | The model, NVFP4-KV, DSpark spec-decode, the garble fix, and image provenance. |
| [docs/04-serving-and-systemd.md](docs/04-serving-and-systemd.md) | The serve profile, TP=2 rendezvous, systemd user units, preflight, and the inference watchdog. |
| [docs/05-troubleshooting.md](docs/05-troubleshooting.md) | OOM ladder, NCCL bandwidth, garbled output, restart deadlocks, and the security/listener audit. |
| [docs/06-reasoning-mode.md](docs/06-reasoning-mode.md) | Turning on thinking mode, the `message.reasoning` field (not `reasoning_content`), the sampling profile, the `max_tokens` trap, tool-call behavior, and client integration. |
| [docs/07-observability-and-warmup.md](docs/07-observability-and-warmup.md) | Observability watcher, the prefill-HoL guard, Telegram alerts, readiness warm-up, and the eval composite score. |
| [docs/08-optimization-and-vllm-025.md](docs/08-optimization-and-vllm-025.md) | The A/B decision ledger and the **vLLM 0.25.1 promotion** (2026-07-15): the two-candidate distinction, config deltas, hardening pass, residual gaps, and the preserved prior 0.21.x lane + rollback. |
| [docs/LONG_CONTEXT_CRASH_FIX.md](docs/LONG_CONTEXT_CRASH_FIX.md) | The `DSPARK_SLOT_CLAMP` long-context crash guard. |

---

## License & attribution

Licensed under Apache-2.0 (see [LICENSE](LICENSE)). This kit is orchestration and documentation
only — it vendors no upstream source; the serving image is built from a pinned community recipe and
the weights are pulled from Hugging Face at deploy time. Upstream components (vLLM, the model weights,
the recipe/image, and the GB10 kernels) each ship under their own licenses; see [NOTICE](NOTICE) for
the attribution required by those licenses. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
