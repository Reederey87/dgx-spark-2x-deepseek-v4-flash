# 01 — Hardware & firmware

This kit targets a matched **pair** of NVIDIA DGX Sparks. The single most important
preparation step that is *not* obvious is **firmware parity** across the two nodes — get
that wrong and the QSFP fabric quietly runs at a fraction of its bandwidth. Read to the end
before you build anything.

## The platform

Each node is a **DGX Spark** built on the **GB10 Grace Blackwell** superchip.

| Property | Value | Why it matters |
|---|---|---|
| Architecture | **aarch64 / ARM64** | All software must be ARM-native. x86 wheels and images fail — pull `aarch64`/`arm64` builds; run Docker with `--platform linux/arm64` where relevant. |
| GPU compute capability | **sm_121a** (GB10 = Blackwell) | CUDA/NCCL built from source must target this arch. `03-build-nccl-tests.sh` builds NCCL and nccl-tests with `-gencode=arch=compute_121,code=sm_121`; the serving image targets `12.1a`. |
| Memory | **~121 GiB unified** (CPU + GPU share one pool) | There is **no separate VRAM**. Every process — the model, KV cache, and anything else running on the node — draws from the same pool. Watch total footprint; see the unified-memory note below. |
| CUDA | **13.0** (`nvcc` V13.0.x) | On `PATH` at `/usr/local/cuda`. |
| GPU driver | **580.159.03** (as validated) | See the driver caveat below. |
| OS | DGX OS / **Ubuntu 24.04** | Kernel `6.17.0-nvidia` class. |

> All figures are what the source recipe was validated against. Treat them as a known-good
> baseline, not a hard requirement — but keep the two nodes **identical**.

## Unified memory — the recurring gotcha

The 121 GiB pool is shared between CPU and GPU and, crucially, between *every workload on
the node*. Two consequences:

- **Never build images or run heavy jobs on a node while it is serving.** A compile or a
  large `rsync` can starve the engine and trigger an OOM. Build first, serve second.
- **`GPU_MEMORY_UTILIZATION` budgets the whole pool.** `0.85` is the sweet spot (~2.8M-token
  KV pool, observed on one pair). Drop to `0.80` (~2.0M tokens) if you co-locate another GPU
  process on the head — that trades KV pool for headroom. Never exceed ~0.86. This is the
  third rung of the OOM ladder in [05-troubleshooting.md](05-troubleshooting.md).

## Firmware parity — do this FIRST

**Matched NIC/SoC firmware across the two nodes is worth a large fraction of your
performance.** In the source recipe, bringing firmware to parity was worth **~+140% prefill**,
and a community user's NCCL bus bandwidth jumped from **~3 GB/s to ~22 GB/s** after a
firmware update alone. **~3 GB/s NCCL bandwidth is the classic stale-firmware signature** —
if [02-networking-nccl.md](02-networking-nccl.md) shows it, come back here.

`00-node-prep.sh` has a dedicated firmware pass that runs `fwupdmgr`:

```bash
# Run this FIRST, on EACH node, if the two nodes' firmware differ.
ssh -t "$HEAD_HOST"   'cd ~/dgx-cluster && bash 00-node-prep.sh head   --firmware'
ssh -t "$WORKER_HOST" 'cd ~/dgx-cluster && bash 00-node-prep.sh worker --firmware'
# It runs `fwupdmgr refresh` + `fwupdmgr update`. If an update is applied:
sudo reboot                      # on each node
# then re-run 00-node-prep.sh WITHOUT --firmware to do the rest of node prep.
```

Bring **both** nodes to the same firmware level before you benchmark NCCL. If you only touch
one node, the mismatch itself can be what collapses the fabric.

## Driver caveat (know this; rollback is possible)

Driver **580.159.03** has been reported to carry a **~3.5× decode-throughput regression on
GB10** versus **580.142**. If your decode tok/s is well below the reference band in the
[README](../README.md) and firmware/NCCL/serving all check out, the driver is a plausible
culprit. Rolling back to 580.142 is possible. Keep both nodes on the **same** driver either
way — a driver mismatch is just another form of node asymmetry.

## What `00-node-prep.sh` does (the one sudo step)

Run interactively **on each node** with its role (`head` / `worker`). It is idempotent,
prints its change list, and asks one confirmation. It:

1. Creates the shared unprivileged `CLUSTER_USER` (`nvidia` by default) — **docker group
   only, never sudo** — and enables `loginctl` linger so its systemd user units survive logout
   and start at boot.
2. Authorizes your control host's public key (`CONTROL_HOST_PUBKEY`) for that user.
3. Writes the QSFP netplan (static rail IPs, MTU 9000) — see [02](02-networking-nccl.md).
4. Installs build/RDMA apt deps (`build-essential`, `openmpi`, `ibverbs-utils`,
   `infiniband-diags`, …).

Rollback instructions are printed by the script itself. After it succeeds on both nodes,
continue from the control host with `01-verify-fabric.sh`.
