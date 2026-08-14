#!/usr/bin/env python3
"""Start/stop fail-closed 1 Hz NVIDIA power sampling on both serving nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import time


RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def run(command: list[str], *, timeout: int = 30) -> str:
    proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout


def ssh(user: str, host: str, command: str, *, timeout: int = 30) -> str:
    return run(["ssh", f"{user}@{host}", command], timeout=timeout)


def prefix(run_id: str) -> str:
    return f"/tmp/dgx-power-{run_id}"


def unit_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:20]
    return f"dgx-power-{digest}.service"


def cleanup_host(user: str, host: str, remote: str, unit: str) -> None:
    command = (
        f"touch {shlex.quote(remote)}.stop; "
        f"systemctl --user stop {shlex.quote(unit)} >/dev/null 2>&1 || true; "
        f"systemctl --user reset-failed {shlex.quote(unit)} >/dev/null 2>&1 || true; "
        f"rm -f {shlex.quote(remote)}.stop {shlex.quote(remote)}.csv "
        f"{shlex.quote(remote)}.pid"
    )
    try:
        ssh(user, host, command, timeout=10)
    except Exception:
        pass


def start(user: str, hosts: list[str], run_id: str) -> dict[str, object]:
    remote = prefix(run_id)
    unit = unit_name(run_id)
    loop = (
        "trap 'exit 0' TERM INT; "
        "while [ ! -e \"$1.stop\" ]; do "
        "ts=$(date +%s.%N); "
        "p=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null "
        "| head -n 1 | tr -d '[:space:]'); "
        "printf '%s,%s\\n' \"$ts\" \"$p\" >> \"$1.csv\"; "
        "sleep 1; done"
    )
    attempts = 0
    last_error: Exception | None = None
    for attempts in range(1, 4):
        try:
            for host in hosts:
                command = (
                    f"systemctl --user stop {shlex.quote(unit)} >/dev/null 2>&1 || true; "
                    f"systemctl --user reset-failed {shlex.quote(unit)} >/dev/null 2>&1 || true; "
                    f"rm -f {shlex.quote(remote)}.stop {shlex.quote(remote)}.csv "
                    f"{shlex.quote(remote)}.pid; "
                    f"systemd-run --user --unit={shlex.quote(unit.removesuffix('.service'))} "
                    f"--collect --quiet /bin/sh -c {shlex.quote(loop)} sh "
                    f"{shlex.quote(remote)}; "
                    f"systemctl --user is-active --quiet {shlex.quote(unit)}; "
                    f"p=$(systemctl --user show -p MainPID --value {shlex.quote(unit)}); "
                    'case "$p" in (*[!0-9]*|\'\'|0) exit 1;; esac; '
                    'kill -0 "$p" 2>/dev/null; '
                    f"printf '%s\\n' \"$p\" > {shlex.quote(remote)}.pid"
                )
                ssh(user, host, command)
            # PID creation alone does not prove nvidia-smi is still producing data.
            # Refuse before launching a benchmark unless every node has a live
            # sampler and at least two real, distinct-timestamp samples.
            time.sleep(2.2)
            for host in hosts:
                samples = ssh(
                    user,
                    host,
                    f"systemctl --user is-active --quiet {shlex.quote(unit)}; "
                    f"test -s {shlex.quote(remote)}.pid; "
                    f"p=$(cat {shlex.quote(remote)}.pid); "
                    'case "$p" in (*[!0-9]*|\'\') exit 1;; esac; '
                    'kill -0 "$p" 2>/dev/null; '
                    f"tail -n 3 {shlex.quote(remote)}.csv 2>/dev/null",
                )
                parse_samples(samples, host)
            break
        except Exception as exc:
            last_error = exc
            # The combined remote start+verify command can launch a unit and then
            # fail before returning. Clean every target, including the failing host.
            for host in hosts:
                cleanup_host(user, host, remote, unit)
            if attempts == 3:
                raise RuntimeError(
                    f"two-node power sampler failed after {attempts} attempts: {last_error}"
                ) from last_error
            time.sleep(0.5)
    return {
        "run_id": run_id,
        "hosts": hosts,
        "sample_hz": 1,
        "sampler_unit": unit,
        "startup_attempts": attempts,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def parse_samples(raw: str, host: str) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in raw.replace("\r", "").splitlines():
        left, sep, right = line.strip().partition(",")
        if not sep:
            continue
        try:
            ts, watts = float(left), float(right)
        except ValueError:
            continue
        if math.isfinite(ts) and math.isfinite(watts) and watts >= 0:
            rows.append((ts, watts))
    rows.sort()
    if len(rows) < 2:
        raise RuntimeError(f"{host}: power sampler returned only {len(rows)} valid samples")
    if len({timestamp for timestamp, _ in rows}) < 2:
        raise RuntimeError(f"{host}: power sampler timestamps did not advance")
    return rows


def summarize(samples: list[tuple[float, float]]) -> dict[str, float | int]:
    joules = 0.0
    gaps: list[float] = []
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt > 0:
            gaps.append(dt)
            joules += dt * (p0 + p1) / 2
    duration = samples[-1][0] - samples[0][0]
    if duration <= 0 or joules <= 0:
        raise RuntimeError("power samples do not span a positive duration/energy interval")
    max_gap = max(gaps, default=float("inf"))
    if max_gap > 3.0:
        raise RuntimeError(f"power sample gap {max_gap:.3f}s exceeds the 3s fail-closed limit")
    return {
        "samples": len(samples),
        "duration_s": round(duration, 6),
        "mean_watts": round(joules / duration, 6),
        "joules": round(joules, 6),
        "max_gap_s": round(max_gap, 6),
    }


def stop(
    user: str, hosts: list[str], run_id: str, completion_tokens: int
) -> dict[str, object]:
    remote = prefix(run_id)
    unit = unit_name(run_id)
    node_results: dict[str, object] = {}
    failures: list[str] = []
    for host in hosts:
        command = (
            f"touch {shlex.quote(remote)}.stop; "
            "i=0; "
            f"while systemctl --user is-active --quiet {shlex.quote(unit)} "
            "&& [ $i -lt 50 ]; do "
            "sleep 0.1; i=$((i+1)); done; "
            f"systemctl --user stop {shlex.quote(unit)} >/dev/null 2>&1 || true; "
            f"cat {shlex.quote(remote)}.csv 2>/dev/null; "
            f"systemctl --user reset-failed {shlex.quote(unit)} >/dev/null 2>&1 || true; "
            f"rm -f {shlex.quote(remote)}.stop {shlex.quote(remote)}.csv "
            f"{shlex.quote(remote)}.pid"
        )
        try:
            node_results[host] = summarize(
                parse_samples(ssh(user, host, command, timeout=20), host)
            )
        except Exception as exc:  # collect both nodes before failing
            failures.append(f"{host}: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))
    total_joules = sum(float(row["joules"]) for row in node_results.values())
    result = {
        "run_id": run_id,
        "sample_hz": 1,
        "sampler_unit": unit,
        "nodes": node_results,
        "cluster_joules": round(total_joules, 6),
        "completion_tokens": completion_tokens,
        "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if completion_tokens > 0:
        result["joules_per_output_token"] = round(total_joules / completion_tokens, 9)
    else:
        result["joules_per_output_token"] = None
    return result


def cleanup(user: str, hosts: list[str], run_id: str) -> None:
    remote = prefix(run_id)
    unit = unit_name(run_id)
    for host in hosts:
        cleanup_host(user, host, remote, unit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "cleanup"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--user", default="nvidia")
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--completion-tokens", type=int, default=0)
    args = parser.parse_args()
    if not RUN_ID.fullmatch(args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore, and dash")
    if len(set(args.hosts)) != len(args.hosts) or len(args.hosts) < 1:
        parser.error("--hosts must be a non-empty unique list")
    if args.action == "start":
        result = start(args.user, args.hosts, args.run_id)
    elif args.action == "stop":
        result = stop(
            args.user, args.hosts, args.run_id, args.completion_tokens
        )
    else:
        cleanup(args.user, args.hosts, args.run_id)
        result = {"run_id": args.run_id, "cleaned": True}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
