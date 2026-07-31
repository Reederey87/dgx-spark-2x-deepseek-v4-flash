# Hermes Agent on DeepSeek-V4-Flash

Everything needed to run [Hermes Agent](https://hermes-agent.nousresearch.com)
against this deployment — a ready config profile and a safe switch script.

Unlike the Inkling kit, **no proxy is required**: the server's `deepseek_v4`
tool parser handles Hermes' streaming requests correctly, so Hermes talks to
the head node's API directly.

## Setup

Prereqs: the cluster serving (`bash runtime/start-cluster.sh` → `:8000`
healthy) and Hermes Agent installed on the head node (`hermes` on PATH,
`hermes-gateway` as a systemd user unit). The API is **loopback-only**, so
Hermes must run on the head node itself (or reach `127.0.0.1:8000` through an
SSH tunnel).

```bash
# install the bundled profile + verify
bash hermes/switch-hermes.sh
```

The script preflights the server, backs up the live config to
`~/.hermes/config.yaml.bak-switch-<ts>`, installs the profile, restarts
`hermes-gateway`, and verifies with a one-shot. Rollback is
`switch-hermes.sh ~/.hermes/config.yaml.bak-switch-<ts>` (run from the node)
or re-running with any other profile file.

Config: the script sources `runtime/cluster.env` when present (`HEAD_HOST`,
`API_PORT`). Env overrides: `HEAD_HOST`, `HERMES_SSH` (the SSH target that
lands on the account Hermes runs under — default `$HEAD_HOST`), `API_PORT`.

## What the profile sets

- **Main model** → the local cluster `http://127.0.0.1:8000/v1`,
  `deepseek-v4-flash-dspark`, 1M ctx. Header comments in `config.yaml`
  document every choice.
- **Thinking = exactly one knob** — `agent.reasoning_effort: high`. The
  cluster already serves thinking by default (`DSPARK_REASONING=on`); this
  sets the effort and is steerable mid-session with `/reasoning`. CoT arrives
  in `message.reasoning`, **not** `reasoning_content`.
- **Auxiliaries** — `vision` / `web_extract` / `title_generation` default to an
  optional cloud model (needs an OpenAI-compatible key; V4-Flash is text-only,
  so there is no local vision swap — `web_extract`/`title_generation` can move
  to `provider: main` for fully-local). `compression` stays local
  (`provider: main`) with thinking off via
  `chat_template_kwargs.thinking: false` — the correct off-spelling for this
  model.
- **Compaction sized for the 1M window** — `threshold_tokens: 260000`
  (binding trigger), prune 48000 / reclaim 20000. Long-context prefill runs
  ~855 tok/s on a 2× GB10 pair, so compacting early beats re-prefilling a
  huge window.
- **Optional fallback rung** — one cloud model, used only when the local
  server is unreachable. Delete the `fallback_providers` block to run
  local-only.

## Gotchas (all verified against this deployment)

- **base_url fields must match** — `model.base_url` and
  `providers.custom.base_url` are both `:8000`; a mismatch silently detaches
  the provider block.
- **Never `reasoning_effort: none`** on the main model — it emits an
  Ollama-style `think=false` this server rejects with 422. To run without
  thinking, either send `chat_template_kwargs.thinking: false` per request or
  serve with `DSPARK_REASONING=off` (see `docs/06`).
- **Thinking eats the output budget** — small `max_tokens` yields empty
  content with `finish_reason=length` (the CoT consumed it). The profile sets
  `max_tokens: 65536`; this is the model, not a bug.
- **Long-form asks over-deliberate** — on prompts with numeric length
  constraints or very long single-shot asks, the model word-counts and
  re-drafts inside thinking until the cap cuts it off (coherent, not garble).
  Irrelevant for tool-use traffic; chunk long-form asks. See `docs/12`.
- **Compaction is local** (`provider: main`), so `idle_compact_after_seconds`
  must stay 0 — otherwise session resume blocks on a local compaction. Don't
  lower the prune knobs either — pruning below the trigger breaks the prefix
  cache.
- **Keep `~/.hermes/.env` for secrets** (0600) — API keys for the optional
  cloud auxiliaries/fallback live there, never in the profile. If Hermes runs
  under systemd, note that unit env (`HERMES_HOME`/`PATH`) comes from the
  unit file, not `~/.bashrc`.
