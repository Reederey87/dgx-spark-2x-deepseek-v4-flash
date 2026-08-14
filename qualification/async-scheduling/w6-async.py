#!/usr/bin/env python3
"""Deterministic asynchronous serving workload for scheduler A/B qualification."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import aiohttp


FILLER = (
    "Reliable distributed inference requires bounded queues, deterministic evidence, "
    "careful cache accounting, and observable request lifecycles. "
)
SUFFIX = "\nWrite a compact technical explanation."
SCHEMA = "w6-async-v2-c8"
METRIC_RE = re.compile(r"^([^\s{]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$")


def prompt_for(recipe: dict[str, Any]) -> str:
    return f"[nonce:{recipe['nonce']}] " + FILLER * int(recipe["copies"]) + SUFFIX


def prompt_hash(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(prompt_for(recipe).encode()).hexdigest()


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def one(pct: int) -> float:
        rank = (len(ordered) - 1) * pct / 100
        low, high = math.floor(rank), math.ceil(rank)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - rank) + ordered[high] * (rank - low)

    return {f"p{pct}": one(pct) for pct in (50, 95, 99)}


def parse_metrics(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line.strip())
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        if math.isfinite(value):
            totals[match.group(1)] = totals.get(match.group(1), 0.0) + value
    return totals


async def tokenize(
    session: aiohttp.ClientSession, base_url: str, model: str, prompt: str
) -> int:
    async with session.post(
        f"{base_url}/tokenize",
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=600,
    ) as response:
        payload = await response.json(content_type=None)
        if response.status != 200:
            raise RuntimeError(f"/tokenize returned {response.status}: {payload}")
    if isinstance(payload.get("count"), int):
        return int(payload["count"])
    tokens = payload.get("tokens") or payload.get("token_ids")
    if isinstance(tokens, list):
        return len(tokens)
    raise RuntimeError("/tokenize response lacks count/tokens")


async def exact_recipe(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    target: int,
    nonce: str,
    output_tokens: int,
    offset_s: float,
) -> dict[str, Any]:
    low, high = 0, max(1, target // 18)
    cache: dict[int, int] = {}

    async def count(copies: int) -> int:
        if copies not in cache:
            recipe = {"nonce": nonce, "copies": copies}
            cache[copies] = await tokenize(session, base_url, model, prompt_for(recipe))
        return cache[copies]

    while await count(high) < target:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if await count(middle) <= target:
            low = middle
        else:
            high = middle
    actual = await count(low)
    recipe = {
        "nonce": nonce,
        "copies": low,
        "target_input_tokens": target,
        "input_tokens": actual,
        "output_tokens": output_tokens,
        "offset_s": round(offset_s, 6),
    }
    recipe["prompt_sha256"] = prompt_hash(recipe)
    return recipe


def arrival_offsets(count: int, rate: float, seed: int) -> list[float]:
    if rate <= 0:
        raise ValueError("request rate must be positive")
    rng = random.Random(seed)
    gaps = [rng.expovariate(rate) for _ in range(max(0, count - 1))]
    if gaps:
        target_duration = count / rate
        scale = target_duration / sum(gaps)
        gaps = [gap * scale for gap in gaps]
    offsets = [0.0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    return offsets


async def create_manifest(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    profile: str,
    seed: int,
    request_rate: float,
) -> dict[str, Any]:
    if profile == "decode-heavy":
        specs = [(512, 2048)] * 48
        offsets = [0.0] * len(specs)
        concurrency = 8
        rate: float | None = None
    else:
        specs = [(512, 256)] * 24 + [(8192, 512)] * 12 + [(65536, 1024)] * 4
        rng = random.Random(seed)
        rng.shuffle(specs)
        offsets = arrival_offsets(len(specs), request_rate, seed)
        concurrency = 8
        rate = request_rate
    recipes = []
    for index, ((input_tokens, output_tokens), offset) in enumerate(zip(specs, offsets)):
        nonce = f"w6-{profile}-{seed:08x}-{index:03d}"
        recipes.append(
            await exact_recipe(
                session,
                base_url,
                model,
                input_tokens,
                nonce,
                output_tokens,
                offset,
            )
        )
    return {
        "schema": SCHEMA,
        "profile": profile,
        "model": model,
        "seed": seed,
        "request_rate": rate,
        "burstiness": 1.0 if rate is not None else None,
        "max_concurrency": concurrency,
        "recipes": recipes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def validate_manifest(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    profile: str,
    manifest: dict[str, Any],
    *,
    verify_tokens: bool = True,
) -> None:
    expected_count = 48 if profile == "decode-heavy" else 40
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("model") != model
        or manifest.get("profile") != profile
        or len(manifest.get("recipes") or []) != expected_count
    ):
        raise RuntimeError("W6 manifest identity/count mismatch")
    for recipe in manifest["recipes"]:
        if prompt_hash(recipe) != recipe.get("prompt_sha256"):
            raise RuntimeError("W6 prompt SHA mismatch")
        if verify_tokens:
            actual = await tokenize(session, base_url, model, prompt_for(recipe))
            if actual != recipe.get("input_tokens"):
                raise RuntimeError(
                    f"W6 prompt token mismatch {actual} != {recipe.get('input_tokens')}"
                )


async def fetch_metrics(session: aiohttp.ClientSession, base_url: str) -> dict[str, Any]:
    async with session.get(f"{base_url}/metrics", timeout=10) as response:
        text = await response.text()
        response.raise_for_status()
    totals = parse_metrics(text)
    return {
        "ts": time.monotonic(),
        "running": totals.get("vllm:num_requests_running", 0),
        "waiting": totals.get("vllm:num_requests_waiting", 0),
        "kv": totals.get("vllm:kv_cache_usage_perc", 0),
        "preemptions": totals.get("vllm:num_preemptions_total", 0),
        "draft": totals.get("vllm:spec_decode_num_draft_tokens_total", 0),
        "accepted": totals.get("vllm:spec_decode_num_accepted_tokens_total", 0),
        "generation": totals.get("vllm:generation_tokens_total", 0),
    }


async def run_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    model: str,
    recipe: dict[str, Any],
    run_id: str,
    index: int,
    started: float,
    thinking: bool,
) -> dict[str, Any]:
    await asyncio.sleep(max(0.0, float(recipe["offset_s"]) - (time.monotonic() - started)))
    arrived = time.monotonic()
    async with semaphore:
        admitted = time.monotonic()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_for(recipe)}],
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": int(recipe["output_tokens"]),
            "min_tokens": int(recipe["output_tokens"]),
            "ignore_eos": True,
            "chat_template_kwargs": {"thinking": thinking},
        }
        first: float | None = None
        previous: float | None = None
        itls: list[float] = []
        completion_tokens: int | None = None
        status: int | None = None
        error: str | None = None
        try:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"X-Request-Id": f"{run_id}-{index:03d}"},
                timeout=aiohttp.ClientTimeout(total=3600, sock_read=1200),
            ) as response:
                status = response.status
                if status != 200:
                    error = (await response.text())[:500]
                else:
                    async for raw in response.content:
                        for line in raw.decode(errors="replace").splitlines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            now = time.monotonic()
                            if first is None:
                                first = now
                            if previous is not None:
                                itls.append(now - previous)
                            previous = now
                            try:
                                event = json.loads(line[6:])
                                usage = event.get("usage") or {}
                                if usage.get("completion_tokens") is not None:
                                    completion_tokens = int(usage["completion_tokens"])
                            except Exception:
                                pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ended = time.monotonic()
    output_tokens = completion_tokens or int(recipe["output_tokens"] if status == 200 else 0)
    return {
        "index": index,
        "http_code": status,
        "error": error,
        "arrival_s": arrived - started,
        "client_queue_ms": (admitted - arrived) * 1000,
        "ttft_ms": None if first is None else (first - admitted) * 1000,
        "e2e_ms": (ended - admitted) * 1000,
        "itl_ms": [value * 1000 for value in itls],
        "input_tokens": recipe["input_tokens"],
        "output_tokens": output_tokens,
    }


async def async_main(args: argparse.Namespace) -> int:
    connector = aiohttp.TCPConnector(limit=32, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        if args.manifest.exists():
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        else:
            manifest = await create_manifest(
                session,
                args.base_url,
                args.model,
                args.profile,
                args.seed,
                args.request_rate,
            )
            tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
            tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, args.manifest)
        await validate_manifest(
            session,
            args.base_url,
            args.model,
            args.profile,
            manifest,
            verify_tokens=not args.skip_tokenize_validation,
        )
        manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if args.prepare_only:
            result = {
                "schema": SCHEMA,
                "profile": args.profile,
                "manifest_sha256": manifest_sha,
                "recipes": len(manifest["recipes"]),
                "tokens_verified": not args.skip_tokenize_validation,
                "passed": True,
                "verdict": "prepared",
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result))
            return 0
        before = await fetch_metrics(session, args.base_url)
        samples: list[dict[str, Any]] = []
        stop_metrics = asyncio.Event()

        async def monitor() -> None:
            while not stop_metrics.is_set():
                samples.append(await fetch_metrics(session, args.base_url))
                try:
                    await asyncio.wait_for(stop_metrics.wait(), timeout=1)
                except TimeoutError:
                    pass

        monitor_task = asyncio.create_task(monitor())
        started = time.monotonic()
        semaphore = asyncio.Semaphore(int(manifest["max_concurrency"]))
        tasks = [
            asyncio.create_task(
                run_request(
                    session,
                    semaphore,
                    args.base_url,
                    args.model,
                    recipe,
                    args.run_id,
                    index,
                    started,
                    args.thinking,
                )
            )
            for index, recipe in enumerate(manifest["recipes"])
        ]
        requests = await asyncio.gather(*tasks)
        duration = time.monotonic() - started
        stop_metrics.set()
        await monitor_task
        after = await fetch_metrics(session, args.base_url)

    good = [row for row in requests if row["http_code"] == 200 and not row["error"]]
    all_itl = [value for row in good for value in row["itl_ms"]]
    total_output = sum(int(row["output_tokens"]) for row in good)
    total_input = sum(int(row["input_tokens"]) for row in good)
    draft_delta = after["draft"] - before["draft"]
    accepted_delta = after["accepted"] - before["accepted"]
    result = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "profile": args.profile,
        "thinking": args.thinking,
        "manifest_sha256": manifest_sha,
        "request_rate": manifest["request_rate"],
        "burstiness": manifest["burstiness"],
        "max_concurrency": manifest["max_concurrency"],
        "duration_s": duration,
        "completed": len(good),
        "failed": len(requests) - len(good),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(good) / duration if duration else 0,
        "output_throughput": total_output / duration if duration else 0,
        "total_token_throughput": (total_input + total_output) / duration if duration else 0,
        "latency_ms": {
            "ttft": percentiles([row["ttft_ms"] for row in good if row["ttft_ms"] is not None]),
            "itl": percentiles(all_itl),
            "e2e": percentiles([row["e2e_ms"] for row in good]),
            "client_queue": percentiles([row["client_queue_ms"] for row in requests]),
        },
        "scheduler": {
            "peak_running": max((row["running"] for row in samples), default=0),
            "peak_waiting": max((row["waiting"] for row in samples), default=0),
            "peak_kv": max((row["kv"] for row in samples), default=0),
            "preemptions_delta": after["preemptions"] - before["preemptions"],
        },
        "spec_decode": {
            "draft_tokens": draft_delta,
            "accepted_tokens": accepted_delta,
            "acceptance_rate": accepted_delta / draft_delta if draft_delta > 0 else None,
        },
        "metrics": samples,
        "requests": requests,
        "passed": len(good) == len(requests),
    }
    result["verdict"] = "pass" if result["passed"] else "fail"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("decode-heavy", "production-mix"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--request-rate", type=float, default=0.15)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-tokenize-validation", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("invalid --run-id")
    try:
        return asyncio.run(async_main(args))
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "run_id": args.run_id,
            "profile": args.profile,
            "passed": False,
            "verdict": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }
        args.output.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
