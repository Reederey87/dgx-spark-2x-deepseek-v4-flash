#!/usr/bin/env python3
"""Decode-rate-vs-context-depth benchmark for the 2x DGX Spark DeepSeek-V4-Flash lane.

Closes the decode-at-depth blind spot flagged by MiaAI-Lab issue #22 (on a different lane,
nvfp4_ds_mla decode was reported collapsing to ~1 tok/s at ~630K context): every published
C1/C8 number for this kit is short-context, and decode rate at 500K+ is unmeasured here.
This script measures TTFT, prefill tok/s, and steady-state decode tok/s at exact prompt
depths up to ~2/3 of the 1M window.

Run it ON the head node against the loopback API (default http://127.0.0.1:8000/v1).
It puts REAL load on a serving cluster — a 655K-token case is many minutes of prefill per
rep — so run it only when the lane may carry bench load (see
docs/07-observability-and-warmup.md). The fp8_ds_mla comparison arm is a separate config
lane (KV_CACHE_DTYPE=fp8_ds_mla in cluster.env), not a flag here.

Stdlib only. Methodology adapted from MiaAI-Lab's scripts/benchmark-0731.py (tokenize-loop
prompt sizing, usage-based prefill rate, incremental JSON writes).
"""
import argparse
import json
import os
import secrets
import statistics
import sys
import time
import urllib.error
import urllib.request

FILLER_UNIT = "benchmark context datum "  # ~3 tokens; uniform so char-proportional trims converge


def post_json(url, body, timeout, retries=4):
    for attempt in range(retries):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def tokenize_url(base_url):
    return base_url.removesuffix("/v1") + "/tokenize"


def tokenize_count(base_url, model, text, timeout):
    return post_json(tokenize_url(base_url), {"model": model, "prompt": text}, timeout)["count"]


def fit_prompt(base_url, model, target, nonce, timeout):
    """Build a prompt of exactly `target` tokens. The nonce goes FIRST (it must land in the
    first cache block so prefix caching can't leak across requests/cases); all trims cut the
    tail only. Returns (text, actual_token_count)."""
    text = f"unique request {nonce} " + FILLER_UNIT * max(1, target // 3)
    count = tokenize_count(base_url, model, text, timeout)
    for _ in range(40):
        if count == target:
            break
        if count < target:
            text += FILLER_UNIT * max(1, (target - count) // 3)
        else:
            text = text[: max(1, int(len(text) * target / count) - 4)]
        count = tokenize_count(base_url, model, text, timeout)
    if count != target:
        print(f"  warn: prompt fit gave {count} tokens, target {target} — recording actual",
              file=sys.stderr)
    return text, count


def stream_case(base_url, model, prompt, max_tokens, timeout):
    """One streaming chat completion, thinking off. Returns timing/usage metrics;
    decode tok/s is steady-state (inter-chunk deltas, first token excluded)."""
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": prompt + f"\nReturn exactly {max_tokens} numbered lowercase English words, then stop."}],
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False},
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{base_url}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    deltas = []
    usage = None
    finish_reason = None
    n_chunks = 0
    saw_done = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            now = time.perf_counter()
            if now - t0 > timeout:
                raise TimeoutError(f"request exceeded the {timeout}s wall-clock cap")
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if event.get("error"):
                raise RuntimeError(f"stream error event: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                n_chunks += 1
                if first is None:
                    first = now
                else:
                    deltas.append(now - last)
                last = now
    if not saw_done:
        raise RuntimeError("stream ended before [DONE] — truncated/aborted response, rep discarded")
    done = time.perf_counter()
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    ttft = (first or done) - t0
    decode_window = (last - first) if (first is not None and last is not None) else 0.0
    # Spec decode (DSpark MTP n=2) commits multiple tokens per SSE chunk. The first
    # chunk's tokens land AT `first`, outside the [first, last] window, so subtract the
    # mean per-chunk share rather than assuming the first chunk carried exactly 1 token
    # (that assumption biases decode tok/s high on this lane).
    decode_tokens = completion_tokens * (1 - 1 / n_chunks) if n_chunks > 1 else 0
    decode_tok_s = (decode_tokens / decode_window
                    if decode_tokens > 0 and decode_window > 0 else None)
    return {"ttft_s": round(ttft, 2),
            "elapsed_s": round(done - t0, 2),
            "prompt_tokens": prompt_tokens,
            "prefill_tok_s": round(prompt_tokens / ttft, 1) if ttft > 0 else None,
            "completion_tokens": completion_tokens,
            "stream_chunks": n_chunks,
            "decode_tok_s": round(decode_tok_s, 2) if decode_tok_s else None,
            "median_interchunk_ms": round(statistics.median(deltas) * 1000, 1) if deltas else None,
            "finish_reason": finish_reason}


def summarize(reps):
    def stats(key):
        vals = [r[key] for r in reps if r.get(key) is not None]
        if not vals:
            return {"median": None, "min": None, "max": None}
        return {"median": round(statistics.median(vals), 2),
                "min": round(min(vals), 2), "max": round(max(vals), 2)}
    return {"decode_tok_s": stats("decode_tok_s"),
            "ttft_s": stats("ttft_s"),
            "prefill_tok_s": stats("prefill_tok_s")}


def write_report(path, report):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)  # crash-safe: a kill mid-run never leaves a torn JSON


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI base URL of the head API (loopback on the head node)")
    p.add_argument("--model", default="deepseek-v4-flash-dspark")
    p.add_argument("--depths", default="8192,131072,262144,409600,655360",
                   help="comma-separated target prompt sizes in tokens")
    p.add_argument("--reps", type=int, default=2, help="repetitions per depth")
    p.add_argument("--max-tokens", type=int, default=128, help="decode length per request")
    p.add_argument("--request-timeout", type=int, default=3600,
                   help="hard wall-clock cap per request, seconds (near-1M prefill is slow)")
    p.add_argument("--out", default=None,
                   help="results JSON path (default: results/decode-depth-<timestamp>.json under cwd)")
    args = p.parse_args()
    if args.reps < 1:
        p.error("--reps must be >= 1")

    out = args.out or os.path.join("results", f"decode-depth-{time.strftime('%Y%m%dT%H%M%S')}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    depths = [int(d) for d in args.depths.split(",")]
    report = {"meta": {"url": args.url, "model": args.model, "depths": depths,
                       "reps": args.reps, "max_tokens": args.max_tokens,
                       "request_timeout_s": args.request_timeout,
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "note": "decode-at-depth probe for the MiaAI #22 blind spot; fp8_ds_mla arm needs a separate config lane"},
              "cases": []}
    write_report(out, report)
    print(f"bench-decode-depth: {len(depths)} depths x {args.reps} reps -> {out}", file=sys.stderr)

    for depth in depths:
        case = {"target_prompt_tokens": depth, "reps": []}
        report["cases"].append(case)  # mutable ref: every write_report sees live progress
        for rep in range(args.reps):
            nonce = secrets.token_hex(8)  # first cache block — prefix cache can't leak across requests
            label = f"depth={depth} rep={rep + 1}/{args.reps}"
            try:
                t0 = time.perf_counter()
                prompt, actual = fit_prompt(args.url, args.model, depth, nonce,
                                            args.request_timeout)
                print(f"{label}: prompt fit at {actual} tokens ({time.perf_counter() - t0:.1f}s), streaming…",
                      file=sys.stderr)
                result = stream_case(args.url, args.model, prompt, args.max_tokens,
                                     args.request_timeout)
                result["nonce"] = nonce
                # fit_prompt's raw-text token count (usage prompt_tokens additionally
                # includes the chat template + appended instruction) — keeps a
                # non-converged fit identifiable from the results file alone.
                result["fitted_prompt_tokens"] = actual
            except Exception as exc:  # keep later depths running; the error is in the JSON
                result = {"nonce": nonce, "error": f"{type(exc).__name__}: {exc}"}
            case["reps"].append(result)
            case["summary"] = summarize(case["reps"])
            if "error" in result:
                print(f"{label}: ERROR {result['error']}", file=sys.stderr)
            else:
                print(f"{label}: ttft={result['ttft_s']}s prefill={result['prefill_tok_s']} tok/s "
                      f"decode={result['decode_tok_s']} tok/s ({result['finish_reason']})",
                      file=sys.stderr)
            write_report(out, report)
        s = case["summary"]["decode_tok_s"]
        print(f"depth={depth}: decode tok/s median={s['median']} min={s['min']} max={s['max']}",
              file=sys.stderr)

    failures = sum(1 for c in report["cases"] for r in c["reps"] if "error" in r)
    if failures:
        print(f"bench-decode-depth: done with {failures} failed rep(s) -> {out}", file=sys.stderr)
        sys.exit(1)
    print(f"bench-decode-depth: done -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
