# dgx-spark-2x-deepseek-v4-flash

Two desk-side **NVIDIA DGX Sparks**, one QSFP cable, a control host with SSH to both, and
this kit — that's what you need to serve **DeepSeek-V4-Flash-0731** (a 284B-parameter MoE,
~13B active per token) at up to **1M tokens of context**. vLLM splits the model across both
boxes (tensor-parallel 2), and the head node exposes a plain OpenAI-compatible API on its
loopback (`127.0.0.1:8000`). This repo is the full production recipe: configs, scripts,
systemd units, image build, and docs that explain *why* every knob is set the way it is.
In a hurry? → [Quickstart](#quickstart).

> ⚠️ **Experimental / n=1.** Everything here was validated on **one** 2× GB10 pair — every
> number is an observation, not a guarantee; yours will vary. The full-source image build is
> multi-hour nvcc with serving stopped. Throughput tables are short-context; decode rate at
> 500K+ context is unmeasured on this lane (`runtime/bench-decode-depth.py` closes that).
> The DSpark/GB10 stack is fast-moving and largely single-author. The current c8r lane
> builds from source (the FlashInfer wheel and the PyTorch builder base image are the
> prebuilt exceptions); the older rollback lanes (0.25.1, 0.21.x) additionally depend on
> prebuilt, non-source-buildable kernels and images.

Name decoder (this kit's jargon, used throughout):

- **gx10** — the community GB10 port lineage this kit builds on (anemll's `dspark-vllm-gx10` overlay).
- **DSpark** — DeepSeek's speculative-decoding draft-head family used by V4-Flash (also the name of the preview checkpoint).
- **c8r** — the current image lane: full-source vLLM `main` @48bada6ea4, i.e. "0.27-content". One thing, three spellings: image tag `v0261-main-c8r`, prose "0.27-content", receipt [docs/14](docs/14-vllm-027-c8r.md).
- **cand7 / cand4** — the 0.26.0 thin-overlay image lanes, now the first rollback rungs.

**Current production stack (2026-08-11):** full-source vLLM **main @48bada6ea4**
(0.27-content) + the gx10 GB10 overlay + a measured-neutral #49731 revert → image
`vllm-dspark-runtime:v0261-main-c8r`, built by
[patches/vllm-0261-main-c8r/](patches/vllm-0261-main-c8r/). Weights:
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
at revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. KV pool **3,027,217 tokens** at
the pinned 19.85 GiB budget. Full receipt: [docs/14](docs/14-vllm-027-c8r.md).

**0731 is the official V4-Flash release** (2026-07-31), superseding the
`DeepSeek-V4-Flash-DSpark` preview. Same checkpoint family, same ~155.4 GiB footprint,
byte-identical config and tokenizer — large jump in agentic capability (see
[Performance](#performance)). Preview stays a one-variable weights rollback.

Nothing here is a black box. The serving image is *built* from a pinned upstream vLLM
commit plus a reviewable overlay (no prebuilt third-party image you cannot rebuild), and
the weights are pulled from Hugging Face at deploy time. No accounts, no tokens. Prior
lanes (0.26 cand7, 0.25.1, 0.21.x) are one config + image swap away. See [NOTICE](NOTICE)
for upstream attribution.

## Why use this kit (benefits for other users)

Most dual-Spark DeepSeek writeups are a blog post and a hope. This kit is the **operating
system for the cluster** — the same recipe that runs in production on a real pair, cut
down so someone else can reproduce it without rediscovering the traps.

| Benefit | What you actually get |
|---|---|
| **End-to-end reproducibility** | Numbered `bringup/00–09` path from bare nodes → fabric → image → weights → smoke → systemd. Hostnames are the only edit needed before bring-up (`runtime/cluster.env.example`); the NCCL bench adds one more (its winning RDMA arm). |
| **Hardware / host prerequisites** | 2× DGX Spark (GB10) with matched firmware, one QSFP 200GbE cable, Docker, a control host with SSH to both — plus **~155.4 GiB of weights on *each* node** and headroom for the stage-1 build. |
| **Time to first `/health`** | Half a day to a long day on a clean matched-firmware pair: fabric + NCCL gates, multi-hour full-source stage-1 (serving stopped), ~155 GiB weight pull + head→worker rsync, ~12 min cold compile on first boot. The cand7 thin-image rollback is the faster rebuild path. |
| **Production-shaped, not demo-shaped** | Boot-persistent user units, inference watchdog, metrics timer, preflight invariant checks, ordered worker-before-head restarts, loopback-only API by default. |
| **Evidence-gated knobs** | Every non-obvious default (MTP n=2, `--no-async-scheduling`, KV byte pin, attention backend, cache-root isolation) has a measured A/B and a doc page. You can re-run the gates on your hardware. |
| **Auditable trust surface** | Public HF weights + pinned upstream vLLM SHA + overlay you can read. No kit-side telemetry, no opaque registry image as the only option. |
| **Documented rollback rungs** | Weights (0731 ↔ preview), image (c8r → cand7 → cand4 → 0.25.1 → 0.21), and cache roots that *must* move with the image — spelled out so a bad upgrade is reversible. |
| **Fabric that is measured** | Dual-twin QSFP RoCEv2 A/B (`bringup/04`), NCCL gate ≥15 GB/s, proven ~23 GB/s dual-rail path — not "plug a cable and pray." |
| **Ops that survive Monday** | Warm-up after restart, swap/churn observability, optional Telegram alerts, Xid evidence capture that never restarts vLLM for you. |
| **Agent-ready API surface** | Thinking on by default (`message.reasoning`), DeepSeek-V4 tool parser, 1M context, prefix caching — the profile agents actually need. (Retrieval at ~944K is measured; decode *rate* at 500K+ depth is not — see the [Performance](#performance) callout.) |

What this is **not**: a guarantee that your pair will hit the same tok/s, a hosted image
registry product, or a substitute for reading the gotchas in `docs/05` and `docs/14`. It
is the shortest path from "I have two Sparks" to "I have a correct, boot-persistent,
1M-context DeepSeek endpoint" with the failure modes already written down.

## Why this deployment shape

Two desktop boxes and one cable get you:

- **A 284B MoE with a 1M context window, at home.** The two Sparks pool ~242 GiB of
  unified memory; NVFP4 KV + the c8r packing path stretches that into a
  **3,027,217-token KV pool** (~2.89× a full 1M window of KV capacity). Long-context
  agentic work stops being theoretical on this hardware.
- **Real speed, measured — not claimed.** ~34 tok/s single stream, ~88 tok/s aggregate at
  concurrency 8 on the warm c8r gate (throughput-tie vs the prior 0.26 cand7 lane), with
  DSpark speculative decode acceptance in the ~0.80 band at draft length 2. Every number
  ships with its workload and gate in `docs/08`–`docs/14`.
- **One cable, no cluster plumbing.** Tensor-parallel over a single QSFP 200GbE link: no
  Ray, no switch, no Ethernet fabric — NCCL RDMA at ~23 GB/s, vLLM's native `mp` backend,
  plain systemd user units (no root).
- **A stack you can rebuild.** Full-source stage-1 from upstream's own Dockerfile, then a
  thin runtime overlay — not an irreplaceable community binary as the only path.

## Reproducibility

Everything you need is **public — no accounts, no tokens, anywhere**:

- **Hardware:** 2× DGX Spark (GB10) + one QSFP cable between them + Docker. That's the
  non-negotiable part.
- **Weights:** [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  is a public Hugging Face repo — no token required. (Preview rollback:
  [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark).)
- **Serving image:** full-source build of upstream vLLM `main` @ `48bada6ea4` + the
  overlay in this repo ([patches/vllm-0261-main-c8r/](patches/vllm-0261-main-c8r/)).
  First rollback is the thinner official `vllm/vllm-openai:v0.26.0` + cand7 overlay
  ([patches/vllm-026-rebase/](patches/vllm-026-rebase/)).
- **Path:** clone → `bringup/00–09` → `runtime/cluster.env` from the example → systemd.

Two honest caveats: every number here was measured on one 2× pair — yours will vary —
and the full-source stage-1 is multi-hour nvcc (cluster stopped). The thinner 0.26 cand7
image remains a supported rollback if you need a faster rebuild path.

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

One physical QSFP port shows up as **two** PCIe "twin" netdevs (~100G each) — using both,
on separate subnets, gets the full ~200G. NCCL runs RDMA over both; the control plane
rides rail 1. TP=2 uses vLLM's native `mp` backend, so there's **no Ray** anywhere. The
**worker boots before the head**; the head then rendezvouses at `MASTER_ADDR:MASTER_PORT`
(= `HEAD_R1:25000`).

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

# Image build: full-source stage-1 is multi-HOUR nvcc — stop any serving containers first.
# One-time: clone vLLM on the control host, then pin a worktree at the c8r SHA:
#   git clone https://github.com/vllm-project/vllm.git ~/vllm
#   git -C ~/vllm worktree add ~/vllm-0261-main-wt 48bada6ea4
bash bringup/05-build-image.sh --distribute   # c8r build + worker push; or omit --distribute and run 06
# bash bringup/06-distribute-image.sh         # only if you skipped --distribute above
bash bringup/07-download-weights.sh  # pull public weights to the head's token-free HF cache
bash bringup/08-distribute-weights.sh # rsync weights head → worker; verify file/byte parity
bash bringup/09-smoke-serve.sh       # foreground bring-up via compose; /health + a chat completion

# 5. Install the systemd user units, make the cluster primary, evaluate.
bash bringup/install-services.sh     # sync kit + install units on both nodes (does not start them)
bash runtime/cluster-enable.sh       # enable for boot + start (worker-first) + poll /health
# First boot on empty -c8r cache roots is a full cold compile (~12 min). Warm before gating.
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

Running an agent client (Hermes Agent) against the deployment — config profile, switch script,
and the model-specific gotchas: [hermes/README.md](hermes/README.md).

---

## Tunables (the "sweet spot")

All live in `runtime/cluster.env`; `render-env.sh` bakes them into a node-local `.env.dspark`
that compose reads. The full vLLM serve argv lives only in `runtime/docker-compose.dspark.yml`.

| Knob | Default | Meaning |
|---|---|---|
| `MAX_MODEL_LEN` | `1048576` | Context ceiling — the model's true YaRN ceiling (65536×16). Higher boots but extrapolates past calibration. First rung of the OOM ladder. |
| `MAX_NUM_SEQS` | `12` | Concurrent streams. Drop toward `4` → `1` under memory pressure. |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill batch budget. |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Share of the ~121 GiB **unified** pool. Drop to `0.80` if you co-locate other GPU processes on the head. Never exceed ~0.86. |
| `KV_CACHE_MEMORY_BYTES` | `21316272128` | Pins the KV pool in bytes (**3,027,217 tokens** on c8r via #48993 packing; cand7 rollback reports 2,948,751 at the same pin). Set = vLLM skips profiling and ignores `GPU_MEMORY_UTILIZATION` for KV sizing. **No free-memory clamp** — an oversized pin OOMs the boot. Unset = profiler sizing returns. See `docs/08`, `docs/14`. |
| `MTP_NUM_TOKENS` | `2` | DSpark speculative draft length — the current sweet spot: **+3.8% single-stream decode vs `3`, concurrency-8 tie** (measured, see `docs/11`). `3` is a fine fallback; greedy `5` is unsafe (garble risk, see `docs/03`). |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | `72` | Keeps the spec-decode decode path graphed at concurrency. Derive as `MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1) × 2` (cap 512) — `72 = 12 × (2+1) × 2`. See `docs/08`. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `1` | The supported CUDA-graph route for this model. **Keep `1`.** See `docs/08`. |
| `TRITON_CACHE_DIR` | `/cache/huggingface/triton-cache-c8r` | Triton kernel cache **must** live on the persistent HF-cache bind — unset falls to container-ephemeral storage and cold-recompiles on every recreate (see `docs/07`). The `-c8r` root follows the full-source cache-root rule (own roots per image generation; see `docs/14`). |
| `SHUTDOWN_TIMEOUT` | `30` | vLLM engine grace period for in-flight requests after SIGTERM. The systemd units provide 90 s total stop headroom. |
| `GLOO_SOCKET_IFNAME` | `enp1s0f1np1` | Pins the CPU-side Gloo coordination group to the stable QSFP control rail; normally matches `NCCL_SOCKET_IFNAME`. |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `4096` | Caps each running long-prefill chunk so short requests interleave — the prefill head-of-line fix. `0`/unset disables it (short-request TTFT regresses under long prefills). See `docs/07`. |
| `DSPARK_REASONING` | `on` | Thinking mode (**production default** — what the Performance numbers were measured at). `on` = server-default thinking + `temp/top_p 1.0`; read the CoT from **`message.reasoning`** (not `reasoning_content`). `off` = non-think greedy (`temp 0`), fastest first token. **With `on`, give requests a generous `max_tokens` (≥1024)** or `content` comes back empty (the max_tokens trap). See `docs/06`. |
| `NCCL_IB_HCA` | `rocep1s0f1,roceP2p1s0f1` | RDMA data path. Default = both RoCE twins (~200G). `bringup/04-run-nccl-bench.sh` A/B-tests this. |

The serve argv also pins `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4"}'`
(the SM120 sparse-MLA route — explicit drift-guard; the default would resolve identically)
and `--no-async-scheduling` (on 0.26+ the async default would shrink the *reported* KV pool
via upstream #47728's doubled sliding-window reservation — measured throughput-neutral, so
the pool comes back for free; see `docs/10`, `docs/11`).

---

## Performance

All measured on our own 2× GB10 pair (`GPU_MEMORY_UTILIZATION=0.85`, `DSPARK_REASONING=on`).
Treat these as **observations, not guarantees** — yours will vary.

**DeepSeek-V4-Flash-0731 on c8r (current, 2026-08-11 warm gate):**

| Metric | Result |
|---|---|
| Throughput — single stream (C1) | **33.85 tok/s** mean (6 batches; −1.14% vs same-day cand7, Welch TIE) |
| Throughput — aggregate @ concurrency 8 (C8) | **87.78 tok/s** mean (6 batches; +0.11% vs same-day cand7, Welch TIE) |
| Spec-decode acceptance (eval) | **0.800** mean draft length 1.60 (draft len 2, probabilistic) |
| Bench acceptance | **0.495 / 0.490** (flat vs cand7) |
| Deep-context retrieval | **3/3 needle HITs @ ~944K** (prior 0731 gate; c8 arm reconfirmed) |
| KV cache pool | **3,027,217 tokens** (same 19.85 GiB pin as cand7; **+2.7%** via #48993 packing) — ~**2.89×** concurrent @ 1M |
| Serving image | `vllm-dspark-runtime:v0261-main-c8r` — full-source main @`48bada6ea4` + overlay0261 + #49731 revert, built by [patches/vllm-0261-main-c8r/](patches/vllm-0261-main-c8r/) — see [docs/14](docs/14-vllm-027-c8r.md) |
| Dependencies | main's pins at the SHA (FlashInfer **0.6.16.post3**, b12x lineage, torch/CUDA from the stage-1 Dockerfile) + DeepGEMM rebuild at the 0.26-class SM120 pin |
| Rollback | weights: flip `DSPARK_MODEL` to `DeepSeek-V4-Flash-DSpark` + its `DSPARK_REVISION` · image: `v026-gx10-cand7-backports` + `-cand7` roots (first rung), then cand4, then `vgx10-011-pr47356` (0.25.1) |

> ⚠️ **These are short-context numbers.** Every C1/C8 throughput figure above was measured
> at prompts of a few K tokens; **decode rate at 500K+ context is unmeasured on this
> lane.** The open datapoint that motivated measuring it is
> [MiaAI-Lab issue #22](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/22):
> on a different lane a reporter saw `nvfp4_ds_mla` decode collapse at huge context vs
> `fp8_ds_mla`. `runtime/bench-decode-depth.py` closes that blind spot (see
> [docs/07](docs/07-observability-and-warmup.md)).

**c8r vs the prior cand7 prod** (same pair, same-day warm A/B, 2026-08-11):

| Metric | cand7 | c8r | Delta |
|---|---|---|---|
| C8 aggregate tok/s (mean) | 87.68 | **87.78** | **tie (+0.11%)** |
| C1 single-stream tok/s (mean) | 34.24 | **33.85** | **tie (−1.14%)** |
| Bench acceptance | 0.502 / 0.488 | 0.495 / 0.490 | flat |
| KV pool (same byte pin) | 2,948,751 | **3,027,217** | **+2.7%** |

The short version: **same speed, larger KV pool**, plus native 0.27-content sparse-MLA /
packing work that is not deliverable as a thin 0.26 source patch.

**0731 agentic quality** (why the model upgrade mattered, independent of the image):

![DeepSeek-V4-Flash-0731 agentic benchmarks vs the preview, V4-Pro preview, GLM-5.2 and Opus-4.8 (source: DeepSeek model card)](docs/images/deepseek-v4-flash-0731-agentic-benchmarks.jpeg)

DeepSeek's published agentic results for 0731 vs the preview this kit previously deployed:
**Terminal-Bench 2.1 82.7 vs 61.8**, **DeepSWE 54.4 vs 7.3**, **NL2Repo 54.2 vs 39.4**,
Cybergym 76.7 vs 38.7, Toolathlon-Verified 70.3 vs 49.7 — ahead of even the V4-Pro preview
on every row (source: [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)).
Upgrade evidence: [docs/12](docs/12-dsv4-flash-0731-upgrade.md).

One 0731 quirk worth knowing: give it a prompt with a **numeric length constraint** or a
very long single-shot ask, and it will word-count and re-draft inside its thinking —
coherent, not garble — until the budget runs out. Tight `max_tokens` caps will truncate
the answer. Tool-use and agentic traffic never notice; for long-form asks, leave room or
chunk them. See `docs/12`.

`eval-cluster.sh` prints the composite plus the throughput/latency probes in one run;
`SKIP_TTFT=1 SKIP_LATENCY=1` skips the two slow streaming probes. History:
[docs/08](docs/08-optimization-and-vllm-025.md) (0.25.1) →
[docs/10](docs/10-vllm-026-rebase.md) / [docs/11](docs/11-v026-feature-qualification.md)
(0.26.0) → [docs/12](docs/12-dsv4-flash-0731-upgrade.md) (0731 weights) →
[docs/13](docs/13-vllm-026-cand7.md) (cand7) → [docs/14](docs/14-vllm-027-c8r.md) (c8r).

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
| [docs/07-observability-and-warmup.md](docs/07-observability-and-warmup.md) | Observability watcher, the prefill-HoL guard, per-request cached-token telemetry, Telegram alerts, readiness warm-up, the eval composite score, and the decode-at-depth bench (`runtime/bench-decode-depth.py`). |
| [docs/08-optimization-and-vllm-025.md](docs/08-optimization-and-vllm-025.md) | The A/B decision ledger and the **vLLM 0.25.1 promotion** (2026-07-15): the two-candidate distinction, config deltas, hardening pass, residual gaps, and the preserved prior 0.21.x lane + rollback. |
| [docs/09-upstream-backport-candidates.md](docs/09-upstream-backport-candidates.md) | Post-v0.25.1 upstream vLLM fixes verified **absent** in the gx10 image (in-container probes), ranked — the evidence-backed maintainer ask. |
| [docs/10-vllm-026-rebase.md](docs/10-vllm-026-rebase.md) | The **vLLM 0.26.0 promotion** (2026-07-28): the in-house rebase (official image + gx10 overlay + backports), the two guards it needs, the acceptance-regression hunt, evidence, and rollback. |
| [docs/11-v026-feature-qualification.md](docs/11-v026-feature-qualification.md) | The **0.26.0 feature-qualification round** (2026-07-29): the 12-row release-feature audit, the DSpark **n=2** K re-tune, the explicit backend pin, and the FlashInfer vendor-pin crash proof. |
| [docs/12-dsv4-flash-0731-upgrade.md](docs/12-dsv4-flash-0731-upgrade.md) | The **DeepSeek-V4-Flash-0731 model upgrade** (2026-07-31): compatibility proof, the A/B gate numbers vs the preview, the two offline-cache traps, and the 0731 long-form deliberation trait. |
| [docs/13-vllm-026-cand7.md](docs/13-vllm-026-cand7.md) | The **cand7 backport round** (2026-08-10): now the first image rollback rung; on-path pick selection, throughput-neutral gate, source-patch cache-root rule. |
| [docs/14-vllm-027-c8r.md](docs/14-vllm-027-c8r.md) | The **c8r full-source 0.27-content promotion** (2026-08-11): main @48bada6ea4, overlay0261, #49731 revert, warm-gate TIE + larger KV pool, cold-cache lesson. |
| [docs/LONG_CONTEXT_CRASH_FIX.md](docs/LONG_CONTEXT_CRASH_FIX.md) | The `DSPARK_SLOT_CLAMP` long-context crash guard — **legacy no-op on 0.26+/c8r** (zero readers in the installed package; the overlay handles the crash class itself), kept for 0.25.1-rollback compatibility. |

---

## License & attribution

Licensed under Apache-2.0 (see [LICENSE](LICENSE)). This kit is orchestration and documentation
plus one reviewable source overlay: `patches/vllm-0261-main-c8r/overlay0261/` carries modified
copies of 17 vLLM files (Apache-2.0, upstream SPDX headers retained) that the image build lays
over the pinned upstream tree; the weights are pulled from Hugging Face at deploy time. Upstream components (vLLM, the model weights,
the recipe/image, and the GB10 kernels) each ship under their own licenses; see [NOTICE](NOTICE) for
the attribution required by those licenses. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Thanks

- **anemll** — the dspark-vllm-gx10 GB10 port this kit's serving lane is built on.
- And everyone running this on their own pair — bug reports with logs are the fastest way to make the kit better.
