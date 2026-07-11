#!/usr/bin/env python3
"""Interval-delta analyzer for metrics-watch.sh (Recipe #11).

Reads Prometheus text from metrics_path, compares to previous state, applies
canary exclusion via watchdog-probe.state, prints notes/WARNs/SUMMARY lines.
"""
from __future__ import annotations

import json
import re
import sys

LINE = re.compile(r"^(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")
LE_RE = re.compile(r'le="([^"]*)"')

# Interval count of long prefills (>30s) at/above which we surface an informational
# note (HoL Caveat B: >=2 concurrent long prefills can re-starve short requests).
LONG_PREFILL_NOTE_THRESHOLD = 2


def main() -> None:
    state_path = sys.argv[1]
    probe_path = sys.argv[2]
    TTFT_WARN_MS = float(sys.argv[3])
    TTFT_HOL_MS = float(sys.argv[4])
    WAIT_WARN = float(sys.argv[5])
    KVUTIL_WARN = float(sys.argv[6])
    ACCEPT_WARN = float(sys.argv[7])
    MIN_DRAFT = float(sys.argv[8])
    hol_ok = sys.argv[9] == "1"
    metrics_path = sys.argv[10]

    fam: dict[str, float] = {}
    fam_seen: set[str] = set()
    # histogram buckets keyed by (name, le) — le must stay distinct, unlike other
    # labels (engine/model_name) which sum together like every other family here.
    fam_le: dict[tuple[str, str], float] = {}
    with open(metrics_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE.match(line)
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2), m.group(3)
            try:
                v = float(val)
            except ValueError:
                continue
            fam[name] = fam.get(name, 0.0) + v
            fam_seen.add(name)
            if labels:
                le_m = LE_RE.search(labels)
                if le_m:
                    key = (name, le_m.group(1))
                    fam_le[key] = fam_le.get(key, 0.0) + v

    def g(name: str) -> float:
        return fam.get(name, 0.0)

    def g_le(name: str, le: str) -> float:
        return fam_le.get((name, le), 0.0)

    try:
        probe = json.load(open(probe_path))
        probe_count = int(probe.get("count", 0))
        probe_last_ms = probe.get("last_e2e_ms")
    except Exception:
        probe_count = 0
        probe_last_ms = None

    cur = {
        "ttft_sum": g("vllm:time_to_first_token_seconds_sum"),
        "ttft_count": g("vllm:time_to_first_token_seconds_count"),
        "e2e_sum": g("vllm:e2e_request_latency_seconds_sum"),
        "e2e_count": g("vllm:e2e_request_latency_seconds_count"),
        "accepted": g("vllm:spec_decode_num_accepted_tokens_total"),
        "draft": g("vllm:spec_decode_num_draft_tokens_total"),
        "preempt": g("vllm:num_preemptions_total"),
        "probe_count": probe_count,
        "prefix_hits": g("vllm:prefix_cache_hits_total"),
        "prefix_queries": g("vllm:prefix_cache_queries_total"),
        "prefill_count": g("vllm:request_prefill_time_seconds_count"),
        "prefill_bucket_30": g_le("vllm:request_prefill_time_seconds_bucket", "30.0"),
    }
    waiting = g("vllm:num_requests_waiting")
    running = g("vllm:num_requests_running")
    kv_util = g("vllm:kv_cache_usage_perc")

    # presence on THIS scrape only — older/newer builds may lack these series
    # entirely; g()/g_le() default to 0.0, which would otherwise read as real data.
    prefix_hits_present = "vllm:prefix_cache_hits_total" in fam_seen
    prefix_queries_present = "vllm:prefix_cache_queries_total" in fam_seen
    prefill_count_present = "vllm:request_prefill_time_seconds_count" in fam_seen
    prefill_bucket_30_present = (
        "vllm:request_prefill_time_seconds_bucket",
        "30.0",
    ) in fam_le

    try:
        prev = json.load(open(state_path))
    except Exception:
        prev = None

    warns: list[str] = []
    notes: list[str] = []
    ttft_iv = e2e_iv = accept_iv = preempt_d = None
    ttft_user = e2e_user = ttft_probe = None
    prefix_hit_iv = long_prefills_iv = None
    canary_only = False
    d_probe = 0

    if prev:
        d_ttft_c = cur["ttft_count"] - prev.get("ttft_count", 0)
        d_ttft_s = cur["ttft_sum"] - prev.get("ttft_sum", 0)
        d_e2e_c = cur["e2e_count"] - prev.get("e2e_count", 0)
        d_e2e_s = cur["e2e_sum"] - prev.get("e2e_sum", 0)
        d_acc = cur["accepted"] - prev.get("accepted", 0)
        d_draft = cur["draft"] - prev.get("draft", 0)
        d_probe = probe_count - int(prev.get("probe_count", 0))
        preempt_d = cur["preempt"] - prev.get("preempt", 0)
        if d_ttft_c > 0:
            ttft_iv = d_ttft_s / d_ttft_c * 1000.0
        if d_e2e_c > 0:
            e2e_iv = d_e2e_s / d_e2e_c * 1000.0
        if d_draft >= MIN_DRAFT:
            accept_iv = d_acc / d_draft if d_draft > 0 else None

        # prefix-cache interval hit rate (vLLM #38182: MTP x prefix-cache hit-rate
        # interaction) — LOG-ONLY trend, no WARN.
        if prefix_hits_present and prefix_queries_present:
            d_prefix_hits = cur["prefix_hits"] - prev.get("prefix_hits", 0)
            d_prefix_queries = cur["prefix_queries"] - prev.get("prefix_queries", 0)
            if d_prefix_queries > 0:
                prefix_hit_iv = d_prefix_hits / d_prefix_queries

        # long prefills (>30s) this interval = count delta minus le=30.0 bucket delta.
        # Floor negative deltas (counter reset on restart) to 0 rather than reporting
        # a nonsensical negative count.
        if prefill_count_present and prefill_bucket_30_present:
            d_prefill_count = cur["prefill_count"] - prev.get("prefill_count", 0)
            d_prefill_bucket_30 = cur["prefill_bucket_30"] - prev.get("prefill_bucket_30", 0)
            lp = d_prefill_count - d_prefill_bucket_30
            long_prefills_iv = lp if lp > 0 else 0.0

        # Canary exclusion: watchdog records each probe to .watchdog-probe.state.
        # If every completion this interval is accounted for by canary probes,
        # this is standby — not user inference. vLLM TTFT has no user label on
        # this image, so the side-channel is the only clean split.
        if d_ttft_c > 0 and d_probe >= d_ttft_c and d_probe > 0:
            canary_only = True
            ttft_probe = ttft_iv
            ttft_user = None
            e2e_user = None
        else:
            ttft_user = ttft_iv
            e2e_user = e2e_iv
            if d_probe > 0 and d_ttft_c > d_probe:
                notes.append(
                    f"note: mixed interval user+canary "
                    f"(reqs={d_ttft_c:.0f} probes={d_probe}) "
                    f"— TTFT mean includes canary (~110 ms); slightly optimistic"
                )

    # TTFT alert policy (USER traffic only — never canary-only standby):
    # - waiting=0 + high TTFT → long-prefill compute (expected). Log note only.
    # - waiting>0 + high TTFT → queueing / short-req starvation. Alert candidate.
    # - hol_ok=0 already WARNed in shell; still treat high TTFT as critical context.
    if canary_only:
        notes.append(
            f"note: canary-only interval (watchdog 1-token probe; "
            f"ttft_probe_ms={ttft_probe:.0f} last_probe_e2e_ms={probe_last_ms}) "
            f"— standby, not user traffic"
        )
    elif ttft_user is not None and ttft_user > TTFT_WARN_MS:
        if waiting > 0 or not hol_ok:
            if ttft_user >= TTFT_HOL_MS:
                print(f"TTFT_CONTEND hol ttft_iv_ms={ttft_user:.0f} waiting={waiting:.0f}")
            else:
                print(f"TTFT_CONTEND elev ttft_iv_ms={ttft_user:.0f} waiting={waiting:.0f}")
        else:
            notes.append(
                f"note: ttft elevated {ttft_user:.0f} ms with waiting=0 "
                f"(long-prefill compute or post-boot warm request — not HoL; "
                f"HoL would queue short reqs). hol_ok={int(hol_ok)}"
            )

    if waiting > WAIT_WARN:
        warns.append(
            f"WARN queue: {waiting:.0f} requests waiting > {WAIT_WARN:.0f} "
            f"(saturation / head-of-line)"
        )
    if kv_util > KVUTIL_WARN:
        warns.append(
            f"WARN kv: kv_cache_usage_perc {kv_util:.3f} > {KVUTIL_WARN:.2f} (OOM risk)"
        )
    if accept_iv is not None and accept_iv < ACCEPT_WARN:
        warns.append(
            f"WARN spec-decode: interval acceptance {accept_iv:.3f} "
            f"< {ACCEPT_WARN:.2f} (MTP health)"
        )
    if preempt_d is not None and preempt_d > 0:
        warns.append(
            f"WARN preempt: {preempt_d:.0f} preemptions this interval "
            f"(concurrency thrash at 1M ctx)"
        )
    if long_prefills_iv is not None and long_prefills_iv >= LONG_PREFILL_NOTE_THRESHOLD:
        notes.append(
            f"note capacity-b: {long_prefills_iv:.0f} long prefills (>30s) overlapped "
            f"this interval — HoL Caveat B window (>=2 concurrent long prefills can "
            f"re-starve short requests)"
        )

    def f(x, nd=1):
        return "n/a" if x is None else f"{x:.{nd}f}"

    # ttft_iv_ms / e2e_iv_ms = USER SLI only (n/a on pure canary standby).
    # ttft_probe_ms = canary latency when this interval was probe-only.
    summary = (
        f"SUMMARY ttft_iv_ms={f(ttft_user, 0)} e2e_iv_ms={f(e2e_user, 0)} "
        f"ttft_probe_ms={f(ttft_probe, 0)} probes={d_probe} "
        f"accept_iv={f(accept_iv, 3)} waiting={waiting:.0f} running={running:.0f} "
        f"kv_util={kv_util:.3f} preempt_d={f(preempt_d, 0)} "
        f"prefix_hit_iv={f(prefix_hit_iv, 3)} long_prefills_iv={f(long_prefills_iv, 0)}"
    )

    for n in notes:
        print(n)
    for w in warns:
        print(w)
    print(summary)

    try:
        json.dump(cur, open(state_path, "w"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
