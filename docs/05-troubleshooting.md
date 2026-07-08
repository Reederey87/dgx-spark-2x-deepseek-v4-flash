# 05 — Troubleshooting

Symptom-first. Each section is an **ordered** ladder — make **one** change at a time and
re-verify (re-run `eval-cluster.sh`, `04-run-nccl-bench.sh`, or the relevant probe) before the
next. Changing several knobs at once hides which one mattered.

---

## (a) OOM / instability at high context

The GB10 pool is **unified** (~121 GiB, CPU+GPU shared). Under memory pressure — engine
death, allocation failures, or crashes that appear only at long context under concurrency —
walk this ladder, **re-eval after each step**:

1. `MAX_MODEL_LEN` **1048576 → 524288** (halve the context ceiling first — cheapest win).
2. `MAX_NUM_SEQS` **12 → 4 → 1** (fewer concurrent streams).
3. `GPU_MEMORY_UTILIZATION` **0.85 → 0.80** (give the pool headroom; costs KV pool).
4. Drop `--speculative-config` entirely (removes the draft-KV footprint — last resort).

Two rules that prevent most of this:

- **Never build images or run heavy jobs on a node while it serves.** Unified memory means a
  compile or big `rsync` competes directly with the engine. Build first, serve second.
- If crashes appear specifically at **long context under load**, that's the
  `DSPARK_SLOT_CLAMP` territory — keep it at `1` (the default) and see
  [LONG_CONTEXT_CRASH_FIX.md](LONG_CONTEXT_CRASH_FIX.md).

---

## (b) NCCL low bandwidth

Symptom: `04-run-nccl-bench.sh` reports **busBW < 15 GB/s**, or the transport line shows
**`NET/Socket`** instead of `NET/IB`. Walk in order:

1. **Confirm the transport is `NET/IB`.** If it's `NET/Socket`, RDMA never engaged — fix that
   before chasing bandwidth. Insist on seeing the selected **HCA/GID** lines in
   `NCCL_DEBUG=INFO`, not just a generic `NET/IB`.
2. **Firmware parity** (`fwupdmgr`). **~3 GB/s is the stale-firmware signature** — bring both
   nodes to the same firmware ([01](01-hardware-and-firmware.md)), reboot, re-bench.
3. **GID index.** If auto-select picked the wrong endpoint, find the RoCEv2 IPv4 index with
   `show_gids` (often 3) and set `NCCL_IB_GID_INDEX`.
4. **MTU 9000 on both rails, both directions.** A one-sided 9000 silently fragments — re-run
   `01-verify-fabric.sh` (its `ping -M do -s 8972` both ways is the check).
5. **Dual-HCA vs single-HCA+GID3** A/B — arms A and B in `04-run-nccl-bench.sh`. Put the
   winner in `cluster.env`.

> **Do not proceed to serving on socket transport.** It will serve tokens ~10× slower and you
> will misread it as a model/serving problem later.

See [02-networking-nccl.md](02-networking-nccl.md) for the full fabric setup.

---

## (c) Garbled / incoherent output

The model answers but the text is nonsense or repetitive. Check in this order — cheapest and
most-common cause first:

1. **Firmware parity.** Mismatched NIC/SoC firmware corrupts the fabric and can produce
   garbage before it produces an obvious error. Rule it out first ([01](01-hardware-and-firmware.md)).
2. **KV dtype.** Before blaming the model, drop `--kv-cache-dtype` from `nvfp4_ds_mla` to
   `fp8` and re-test. If `fp8` is coherent, the issue is in the NVFP4-KV path, not the weights.
3. **`MTP_NUM_TOKENS`.** Confirm it is **`3`** (not `5`). `3` + probabilistic draft is the
   DSpark garble fix — greedy `5` reintroduces it. See the garble-fix section of
   [03-model-and-features.md](03-model-and-features.md).

---

## (d) Head/worker restart deadlock

Symptom: the head is stuck waiting for the worker, the worker is `failed`/start-limited, or a
stale rendezvous store leaves a fresh worker zombified. `preflight.sh` and `watchdog.sh` are
built to recover this unattended (they reset-failed and bounce the pair in the right order —
see [04-serving-and-systemd.md](04-serving-and-systemd.md)), but if you need to force it:

```bash
bash runtime/stop-cluster.sh && bash runtime/start-cluster.sh   # the recovery hammer: clean stop, then worker-first restart
```

`stop-cluster.sh` stops head-first and removes both containers; `start-cluster.sh` brings the
worker up before the head. That ordering is what clears the stale `:25000` rendezvous store.

---

## (e) Security / listener audit (hard gate)

The compose runs with **`--network host`**, so the container's listeners are the *node's*
listeners. **Audit both nodes** before you consider the deployment done:

```bash
ss -tlnp    # run on BOTH nodes
```

Only these may listen:

- the **API on loopback only** — `127.0.0.1:8000`;
- the **rendezvous / NCCL ports bound to the QSFP fabric** (`192.168.177.x` / `.178.x`).

**Nothing may listen on the node's LAN IP.** `09-smoke-serve.sh` prints exactly the listeners
that are *not* on loopback/QSFP so you can eyeball them.

- **Exposing the API to the LAN:** if you must, add auth (`VLLM_API_KEY`) **and** bind the
  **QSFP IP** — never `0.0.0.0`, never the LAN IP. A `0.0.0.0` bind under `--network host`
  puts an unauthenticated API on your LAN.
- **The rendezvous port** (`*:25000`) binds all interfaces. On a trusted point-to-point QSFP
  fabric that is acceptable. To close it explicitly:
  ```bash
  sudo ufw allow from 192.168.177.0/24 to any port 25000   # pin rendezvous to the fabric
  ```

---

### Still stuck?

- Model / feature flags, provenance, and the garble fix in depth → [03-model-and-features.md](03-model-and-features.md).
- Serving profile, rendezvous, units, watchdog → [04-serving-and-systemd.md](04-serving-and-systemd.md).
- Long-context engine deaths → [LONG_CONTEXT_CRASH_FIX.md](LONG_CONTEXT_CRASH_FIX.md).
