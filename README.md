# dgx-spark-2x-deepseek-v4-flash

Reproducible kit to serve **DeepSeek-V4-Flash-DSpark** (284B MoE / ~13B active, NVFP4-KV,
up to 1M context) on a **2× NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64, CUDA 13)
cluster with **vLLM tensor-parallel = 2** over a single **QSFP 200GbE** cable. The two
Sparks act as one inference engine; the OpenAI-compatible API is served on the head node's
loopback (`127.0.0.1:8000`).

This repo is **orchestration and documentation only**. It vendors no upstream source: the
serving image is *built from* a pinned community recipe and the weights are *pulled from*
Hugging Face at deploy time. See [CREDITS.md](CREDITS.md).

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
Step 3 runs **on** each node. The kit is flat at the repo root; only `docs/` is separate.

```bash
# 1. Configure — this is the single source of truth for the whole kit.
cp cluster.env.example cluster.env
$EDITOR cluster.env                       # set HEAD_HOST / WORKER_HOST (identity block)
export CONTROL_HOST_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"   # authorized on the nodes

# 2. Copy the kit to each node as your normal login user (the cluster user
#    does not exist yet). The on-node dir MUST be named dgx-cluster (units use %h/dgx-cluster).
rsync -a ./ "$HEAD_HOST:~/dgx-cluster/"
rsync -a ./ "$WORKER_HOST:~/dgx-cluster/"

# 3. One-time node prep — runs ON each node with its role (the only sudo step).
#    Add --firmware FIRST if the two nodes' firmware differ (then reboot, re-run without it).
ssh -t "$HEAD_HOST"   'cd ~/dgx-cluster && bash 00-node-prep.sh head'
ssh -t "$WORKER_HOST" 'cd ~/dgx-cluster && bash 00-node-prep.sh worker'

# 4. Bring up the fabric + build + serve — all from the control host, in order.
bash 01-verify-fabric.sh        # QSFP addressing, MTU 9000, RoCE up, jumbo ping both ways
bash 02-setup-cluster-ssh.sh    # node-to-node SSH over the QSFP rail IPs
bash 03-build-nccl-tests.sh     # NCCL v2.30u1 + nccl-tests at sm_121, both nodes
bash 04-run-nccl-bench.sh       # A/B the RDMA arms; put the winner in cluster.env (gate ≥15 GB/s)
bash 05-build-image.sh          # build the DSpark vLLM image on the head (pin BASE_IMAGE_DIGEST)
bash 06-distribute-image.sh     # copy the image head → worker over QSFP; verify IDs match
bash 07-download-weights.sh     # pull public weights to the head's token-free HF cache
bash 08-distribute-weights.sh   # rsync weights head → worker; verify file/byte parity
bash 09-smoke-serve.sh          # foreground bring-up via compose; /health + a chat completion

# 5. Install the systemd user units, make the cluster primary, evaluate.
bash install-services.sh        # sync kit + install units on both nodes (does not start them)
bash cluster-enable.sh          # enable for boot + start (worker-first) + poll /health
bash eval-cluster.sh            # correctness + throughput + long-context needle probes
```

Call it (from the head node, loopback):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "deepseek-v4-flash-dspark",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8, "temperature": 0
      }'
```

Daily ops: `start-cluster.sh` / `stop-cluster.sh` (bring the running cluster up/down),
`cluster-enable.sh` / `cluster-disable.sh` (toggle boot-persistence too), `eval-cluster.sh`,
`metrics.sh`.

---

## Tunables (the "sweet spot")

All live in `cluster.env`; `render-env.sh` bakes them into a node-local `.env.dspark` that
compose reads. The full vLLM serve argv lives only in `docker-compose.dspark.yml`.

| Knob | Default | Meaning |
|---|---|---|
| `MAX_MODEL_LEN` | `1048576` | Context ceiling. `1048576` is the model's true YaRN ceiling (65536×16). Higher boots but extrapolates past calibration. First rung of the OOM ladder. |
| `MAX_NUM_SEQS` | `12` | Concurrent streams. Drop toward `4` → `1` under memory pressure. |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill batch budget. |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Share of the ~121 GiB **unified** pool. Drop to `0.80` if you co-locate other GPU processes on the head. Never exceed ~0.86. |
| `MTP_NUM_TOKENS` | `3` | DSpark speculative draft length. `3` + probabilistic draft is the garble fix — **do not** revert to greedy `5`. See `docs/03`. |
| `NCCL_IB_HCA` | `rocep1s0f1,roceP2p1s0f1` | RDMA data path. Default = both RoCE twins (~200G). `04-run-nccl-bench.sh` A/B-tests this. |

Reference numbers on one pair (yours will vary): KV pool **~2.8M tokens** @ util 0.85
(~2.0M @ 0.80), **5/5** correctness eval, a **147K-token needle retrieved in ~86 s**,
**~48 tok/s** aggregate at concurrency 3, single-stream **~22–34 tok/s**. Startup ~5–10 min
(cold ~9.5, warm ~5).

---

## Documentation

| Doc | Covers |
|---|---|
| [docs/01-hardware-and-firmware.md](docs/01-hardware-and-firmware.md) | GB10 / ARM64 / unified memory, CUDA 13, and why **firmware parity** across the two nodes matters. |
| [docs/02-networking-nccl.md](docs/02-networking-nccl.md) | QSFP dual-twin fabric, RoCEv2/GID, MTU 9000, the NCCL A/B benchmark and its gate. |
| [docs/03-model-and-features.md](docs/03-model-and-features.md) | The model, NVFP4-KV, DSpark spec-decode, the garble fix, and image provenance. |
| [docs/04-serving-and-systemd.md](docs/04-serving-and-systemd.md) | The serve profile, TP=2 rendezvous, systemd user units, preflight, and the inference watchdog. |
| [docs/05-troubleshooting.md](docs/05-troubleshooting.md) | OOM ladder, NCCL bandwidth, garbled output, restart deadlocks, and the security/listener audit. |
| [docs/LONG_CONTEXT_CRASH_FIX.md](docs/LONG_CONTEXT_CRASH_FIX.md) | The `DSPARK_SLOT_CLAMP` long-context crash guard. |

---

## Credits

This kit stands entirely on the community authors below — see [CREDITS.md](CREDITS.md) and
[NOTICE](NOTICE) for full provenance, links, and licenses. If you use it, credit them; if you
improve on it, send fixes upstream.

- **tonyd2wild** — the 2× DGX Spark NVFP4-KV + DSpark serving recipe this kit pins and builds.
- **drowzeys ("Keys")** — the DSpark concurrency patch, the `nvfp4_ds_mla` KV path, and the sparse-MLA / dual-cache work.
- **aidendle94** — the compiled GB10 (`sm_121a`) DeepGEMM and sparse-MLA kernels shipped inside the image.
- **MiaAI-Lab** · **rafaelcaricio** — dual-Spark packaging / worker-first ordering and DSpark vLLM integration.
- **vLLM** (Apache-2.0) — the inference engine. **DeepSeek AI** — the model weights. **NVIDIA** — DGX Spark, GB10, and the "connect two Sparks" guidance.

Licensed under Apache-2.0 (see [LICENSE](LICENSE)). Contributions welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).
