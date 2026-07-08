# Reasoning mode (thinking) — how to turn it on and read it

DeepSeek-V4-Flash is a hybrid reason/chat model. This kit ships **non-think by default** (fast,
greedy, `temp 0`) because the DSpark spec-decode recipe was tuned there and that is the garble-safe
production profile. Reasoning is fully available — it is one flag away — but there are two sharp edges
that cost a full day to trace, so they are documented here in detail.

## TL;DR

- **The chain-of-thought comes back in `message.reasoning`, NOT `message.reasoning_content`.** On the
  vLLM build this kit uses (`0.21.1rc1.dev*`), `reasoning_content` is always `null`. If you probe the
  wrong field you will conclude — wrongly — that "reasoning doesn't work." It works.
- **Two ways to turn it on:** flip the whole server to thinking-by-default (`DSPARK_REASONING=on`), or
  leave the server non-think and opt in **per request** with `chat_template_kwargs`.
- **Budget your `max_tokens`.** Reasoning is spent *before* the answer; a small cap truncates the model
  mid-think and leaves `content` empty.

## Activate it

### Option A — per request (no restart, server stays non-think)

Send `chat_template_kwargs` on the request. Either `thinking` or `enable_thinking` works (the
`deepseek_v4` tokenizer treats them as `thinking or enable_thinking`):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash-dspark",
  "messages": [{"role": "user", "content": "Is 91 prime? Show your reasoning."}],
  "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"},
  "max_tokens": 1024
}' | jq '{reasoning: .choices[0].message.reasoning, content: .choices[0].message.content}'
```

`reasoning` holds the full CoT; `content` holds the clean final answer. Omit `chat_template_kwargs`
(or send `{"thinking": false}`) for a direct non-think answer.

### Option B — server default (whole cluster reasons)

Set the toggle in `cluster.env` and restart the head:

```bash
# cluster.env
DSPARK_REASONING=on            # off (default) = non-think greedy
DSPARK_REASONING_EFFORT=high   # high | max
```

The compose command then serves `--default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}'`
and flips `--override-generation-config` to the reasoning sampling profile (below). Every request reasons
unless it opts out per-request with `{"thinking": false}`. Roll back by setting `off` and restarting.

## Sampling profile — thinking runs at temp 1.0, not greedy

The official V4-Flash reasoning sampling is **`temperature = 1.0`, `top_p = 1.0`** (NOT R1's 0.6/0.95).
Probabilistic DSpark drafting is *designed* for `temp 1.0`, so thinking is more in-regime there than the
greedy `temp 0` used for non-think. The `DSPARK_REASONING=on` path sets this for you. Because it moves
sampling off the greedy point the recipe was tuned at, **re-run `eval-cluster.sh` after switching** — the
garble gate is the accept criterion.

`reasoning_effort`: `high` (default) or `max`. `max` prepends a maximize-depth instruction and needs
`--max-model-len >= 393216` (the 1M default clears it) — reserve it for low-concurrency calls.

## The `max_tokens` trap

In thinking mode the model emits `reasoning … </think> … answer`, and it spends the *early* tokens on the
reasoning block. A `max_tokens` sized for a short answer truncates mid-think:

| `max_tokens` | `finish_reason` | `content` | Recall |
|---|---|---|---|
| `64` | `length` | **empty** | answer stranded in `reasoning` |
| `1024` | `stop` | the answer | correct |

Always give thinking calls generous headroom. (This is why the long-context needle in `eval-cluster.sh`
uses `max_tokens: 1024`, not the 64 that suffices for non-think.)

## Tool calling and multi-turn

DeepSeek's **hosted** API rejects thinking-mode tool loops unless the prior turn's reasoning is echoed
back (`400 The reasoning_content in the thinking mode must be passed back to the API`; see Factory issue
[#1018](https://github.com/Factory-AI/factory/issues/1018)). **This local vLLM does not enforce that** —
a tool round-trip with thinking on and no reasoning echoed back returns `200`. So agent tool loops do not
hard-break here. They only lose prior-turn chain-of-thought on replay, because vLLM's OpenAI-compat
endpoint does not consume incoming `reasoning_content`. Clients that want interleaved thinking preserved
across tool turns should re-embed the prior `<think>…</think>` inline in `content` on replay.

## How it works (for the curious)

1. **Prompt.** The `deepseek_v4` tokenizer (`--tokenizer-mode deepseek_v4`) maps `thinking`/`enable_thinking`
   to `thinking_mode` and appends `<think>` to the prompt in thinking mode (or `</think>` in chat mode, so
   the model answers directly). There is no Jinja `chat_template` — it is custom trust-remote-code encoding.
2. **Generation.** The model emits `reasoning … </think> … answer`.
3. **Parse.** `--reasoning-parser deepseek_v4` splits on `</think>` into `message.reasoning` / `message.content`.

## Client integration notes

- **Raw OpenAI-compatible clients:** read `message.reasoning`. Fall back to `message.reasoning_content`
  only for forward-compatibility with other builds.
- **Factory droid (BYOK `generic-chat-completion-api`):** reasoning-effort flags are not yet supported for
  custom models, and the generic provider does not render the reasoning field. Pin thinking via
  `extraArgs` on the model entry: `"extraArgs": {"chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}}`.
  Give it a large `maxOutputTokens` (e.g. 32768) to clear the reasoning block.
- **Agents generally:** the model reasons and answers correctly with thinking on; the extra cost is
  reasoning tokens (latency) on every call, so weigh it against your latency budget.

## Verify

```bash
# non-empty .reasoning ⇒ thinking is active
curl -s :8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4-flash-dspark",
  "messages":[{"role":"user","content":"Is 91 prime?"}],
  "chat_template_kwargs":{"thinking":true},"max_tokens":600}' \
 | jq '.choices[0].message.reasoning'

bash runtime/eval-cluster.sh   # garble gate — must stay green after enabling DSPARK_REASONING=on
```
