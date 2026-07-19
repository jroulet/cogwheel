#!/usr/bin/env bash
# .claude/sdk/run_py.sh — run a python script/module under the pipeline's
# conda env (SDK_CONDA_ENV / .env chain, default cogwheel_310).
#
# Usage: run_py.sh <script.py|-c 'code'> [args...]
# Exists so the driver can execute vetted python (render_fragments,
# validation probes) through the Bash allowlist's .claude/sdk exception
# when the serena shell layer is unavailable (2026-07-18 session).
set -u
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
if [[ -f "$REPO_ROOT/.env" ]] && [[ -z "${SDK_CONDA_ENV:-}" ]]; then
  SDK_CONDA_ENV="$(grep -E '^SDK_CONDA_ENV=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
ENV_NAME="${SDK_CONDA_ENV:-cogwheel_310}"
PYBIN="$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "ERROR: $PYBIN not found; check SDK_CONDA_ENV" >&2
  exit 1
fi
cd "$REPO_ROOT"
exec "$PYBIN" "$@"
