# 02 — Networking & NCCL

TP=2 means the two GPUs exchange activations on **every token**. The QSFP fabric is the
inference bottleneck, so this step gets its own verification and its own benchmark gate. The
goal: RDMA (RoCEv2) over **both** PCIe twins of the single QSFP port, MTU 9000, with NCCL
proven to select the InfiniBand transport — not sockets.

## One cable, two "twin" links

A single DGX Spark QSFP port enumerates as **two** PCIe netdevs — a "twin" pair, ~100G each:

| Rail | Netdev (`QSFP_IF*`) | RoCE device (`NCCL_IB_HCA`) | Subnet |
|---|---|---|---|
| 1 (primary; bootstrap/control) | `enp1s0f1np1` | `rocep1s0f1` | `192.168.177.0/24` |
| 2 (PCIe twin of the same port) | `enP2p1s0f1np1` | `roceP2p1s0f1` | `192.168.178.0/24` |

Rail IPs (defaults): head `192.168.177.10` / `192.168.178.10`, worker `.11` / `.11`.

Two hard rules:

- **The twins MUST be on different subnets.** Same-subnet breaks routing. That is why rail 1
  is `177.x` and rail 2 is `178.x`.
- **MTU 9000 on both rails, both nodes.** A one-sided 9000 does not error — it *silently
  fragments*, which shows up later as collapsed NCCL bandwidth.

`00-node-prep.sh` writes this as a netplan (`/etc/netplan/40-cx7.yaml`) with static addresses
and `mtu: 9000`, and keeps the QSFP interfaces off the default route.

## Two transports, cleanly separated

NCCL uses two independent paths, and it helps to keep them straight:

- **Bootstrap / control plane → sockets**, pinned to rail 1 via `NCCL_SOCKET_IFNAME`
  (`enp1s0f1np1`). This is also where the TP rendezvous rides — `MASTER_ADDR` = `HEAD_R1`.
- **Data path → RDMA (IB verbs)**, selected via `NCCL_IB_HCA`. The default
  `rocep1s0f1,roceP2p1s0f1` uses **both** RoCE twins for the full ~200G.

The **GID** pins which RoCEv2 IPv4 endpoint NCCL uses. Auto-select usually works; if it picks
the wrong GID, set `NCCL_IB_GID_INDEX` to the RoCEv2 IPv4 index (**often 3**). Leave it empty
to let NCCL auto-select — the compose unsets an empty value so it never forces a bad index.
`01-verify-fabric.sh` prints the RoCEv2 IPv4 GID candidates per device so you know what to pin.

## Verify the fabric — `01-verify-fabric.sh`

Run from the control host after `00-node-prep.sh` on both nodes. It checks, on each node and
across the link:

- rail-1 address present, **MTU 9000 on both rails**;
- both RoCE devices `Up` (`ibdev2netdev`);
- Docker reachable as the cluster user;
- **no default route** via a QSFP interface;
- prints RoCEv2 IPv4 GID candidates;
- **jumbo ping both directions on both rails** — `ping -M do -s 8972` (8972 + 28 = 9000, DF
  set). This is the test that catches a one-sided MTU: it fails loudly instead of fragmenting.

## Benchmark the RDMA path — `03` then `04`

```bash
bash 03-build-nccl-tests.sh   # NCCL v2.30u1 + nccl-tests, sm_121, on BOTH nodes
bash 04-run-nccl-bench.sh     # runs 3 arms from the head, prints a summary + a gate
```

`04-run-nccl-bench.sh` runs `all_reduce_perf` (256M → 4G) across three arms and reports
bus bandwidth + the selected transport line:

| Arm | Config | Reference busBW (one pair) |
|---|---|---|
| **A — dual-twin** | both HCAs (`rocep1s0f1,roceP2p1s0f1`), GID auto | **23.1 GB/s — winner** |
| **B — single + GID3** | one HCA (`rocep1s0f1`), `NCCL_IB_GID_INDEX=3` | 13.6 GB/s |
| **C — socket control** | `NCCL_IB_DISABLE=1` (no RDMA) | 2.1 GB/s |

The dual-twin 23.1 GB/s beats a 20.4 GB/s *two-cable* community reference on one pair — one
QSFP cable, used as two twins, is enough. The socket arm at ~10× slower is the control: it
*proves RDMA is actually engaged* in the winning arm. **~3 GB/s on an RDMA arm is the
stale-firmware signature** — go back to [01](01-hardware-and-firmware.md) and fix firmware
parity. (Numbers are observations on one pair; yours will vary.)

### The gate

`04-run-nccl-bench.sh` fails unless **all** hold:

1. Best RDMA arm **busBW ≥ 15 GB/s**.
2. The winning arm's NCCL debug shows **`NET/IB`** (not `NET/Socket`).
3. The socket control arm is **slower** than the RDMA best (RDMA is real).

When it passes, it prints the exact `cluster.env` lines for the winning arm
(`NCCL_IB_HCA=` and `NCCL_IB_GID_INDEX=`). **Put the winner in `cluster.env`.**

> Insist on seeing the selected **HCA/GID** lines in `NCCL_DEBUG=INFO` output — not merely a
> generic `NET/IB`. A run that "works" on sockets will still serve tokens; it will just be
> ~10× slower and you will chase it in the throughput numbers later. Do not proceed on socket
> transport.

If the gate fails, the bandwidth section of [05-troubleshooting.md](05-troubleshooting.md)
walks the causes in order: confirm `NET/IB` → firmware parity → GID index (`show_gids`) →
MTU 9000 both rails both directions → dual-HCA vs single-HCA+GID3 A/B.
