# Contributing

Thanks for improving this kit. It's a small, focused set of shell scripts,
systemd units, and docs for deploying one specific model on one specific pair of
machines — so contributions that keep it **simple, reproducible, and honest about
risk** are the most valuable.

## Ground rules

- **No secrets, ever.** No hostnames, IPs beyond the RFC1918 defaults, SSH keys,
  API/HF tokens, or personal paths in committed files. Site-specific values live
  only in your local `cluster.env` (git-ignored). Before you push, run:
  ```bash
  grep -rniE 'ssh-ed25519 AAAA|BEGIN .*PRIVATE KEY|api[_-]?key|hf_[A-Za-z0-9]{20}' .
  ```
  and confirm it's clean.
- **Keep the runtime kit flat.** The scripts, `cluster.env`, `render-env.sh`,
  `docker-compose.dspark.yml`, and the `*.service` units are rsynced *together* to
  `KIT_DIR` on each node and run from there. Don't reorganize them into
  subdirectories — it breaks the deploy model. (Human docs live in `docs/`.)
- **One source of truth.** Every tunable belongs in `cluster.env.example` with a
  comment. Don't hardcode a value in a script if it could be a variable.
- **Pin upstreams.** New image/recipe dependencies must be pinned (digest or SHA)
  and credited in `CREDITS.md` / `NOTICE`.
- **Shell hygiene.** `set -euo pipefail`, `bash -n` clean, and prefer bounded
  waits (see `preflight.sh`'s `wait_for`) over unbounded loops.

## Testing a change

There is no CI — this touches real hardware. At minimum:

1. `bash -n *.sh` on every script you touched.
2. `docker compose --env-file .env.dspark -f docker-compose.dspark.yml config`
   parses after a `render-env.sh` run.
3. If you changed the serve path or a tunable, re-run `09-smoke-serve.sh` and
   `eval-cluster.sh` on a real 2× Spark pair and paste the numbers in the PR.

## Scope

This kit deliberately does **not** include downstream integrations (agent
frameworks, gateways, app config). If you built one, link it from your fork —
keep this repo a clean cluster-and-serve foundation others can build on.

By contributing you agree your contribution is licensed under Apache-2.0 (LICENSE).
