# Contributing

Thanks for improving this kit. It's a small, focused set of shell scripts,
systemd units, and docs for deploying one specific model on one specific pair of
machines — so contributions that keep it **simple, reproducible, and honest about
risk** are the most valuable.

## Ground rules

- **No secrets, ever.** No real hostnames, IPs beyond the RFC1918 fabric defaults,
  SSH keys, API/HF/Telegram tokens, or personal paths in committed files.
  Site-specific values live only in your local `runtime/cluster.env` and
  `runtime/notify.env` (both git-ignored). Before you push, run:
  ```bash
  grep -rniE 'ssh-ed25519 AAAA|BEGIN .*PRIVATE KEY|api[_-]?key|hf_[A-Za-z0-9]{20}|[0-9]{8,}:[A-Za-z0-9_-]{30,}' .
  ```
  (the last alternative catches a Telegram bot token) and confirm it's clean, and
  that no real machine names or LAN IPs slipped into a comment.
- **The layout is `bringup/` + `runtime/` + `docs/`.** `bringup/` holds the one-time,
  control-host-driven setup (numbered scripts + `install-services.sh`); `runtime/`
  holds everything a node runs — `cluster.env`, `render-env.sh`,
  `docker-compose.dspark.yml`, the `*.service`/`*.timer` units, and the lifecycle +
  ops scripts. The whole tree is rsynced to `KIT_DIR` on each node *preserving this
  structure*, and the units reference `%h/dgx-cluster/runtime/…`. If you move a
  runtime file, update the unit paths and `bringup/install-services.sh` to match.
- **One source of truth.** Every tunable belongs in `runtime/cluster.env.example`
  with a comment. Don't hardcode a value in a script if it could be a variable —
  and `render-env.sh` bakes each into `.env.dspark`, so a var it emits MUST exist in
  `cluster.env.example` (or `render-env.sh` aborts under `set -u`).
- **Pin upstreams.** New image/recipe dependencies must be pinned (digest or SHA)
  and attributed in `NOTICE`.
- **Shell hygiene.** `set -euo pipefail`, `bash -n` clean, and prefer bounded
  waits (see `preflight.sh`'s `wait_for`) over unbounded loops.

## Testing a change

Run the offline structural checks first:

```bash
bash tests/validate-runtime.sh
```

There is no hardware CI — this touches real machines. At minimum:

1. Keep `tests/validate-runtime.sh` green (it includes `bash -n` for every shipped shell script).
2. From `runtime/`, `docker compose --env-file .env.dspark -f docker-compose.dspark.yml config`
   parses after a `render-env.sh` run.
3. If you changed the serve path or a tunable, re-run `09-smoke-serve.sh` and
   `eval-cluster.sh` on a real 2× Spark pair and paste the numbers in the PR.

## Scope

This kit deliberately does **not** include downstream integrations (agent
frameworks, gateways, app config). If you built one, link it from your fork —
keep this repo a clean cluster-and-serve foundation others can build on.

By contributing you agree your contribution is licensed under Apache-2.0 (LICENSE).
