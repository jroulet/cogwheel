#!/bin/sh
# conda-python.sh — shared conda-Python resolver, sourced by the git hooks
# (pre-commit, post-commit, post-merge) so the interpreter is resolved in ONE
# place via the durable .env idiom instead of a hardcoded absolute path.
#
# Precedence (mirrors gw_detection_ias and python-dotenv's override=False):
#   shell SDK_CONDA_ENV  >  repo-root .env  >  the cogwheel_310 default.
# Resolution NEVER hard-fails on a missing .env — it simply falls through to
# the default env name, then to python3 if that env has no interpreter.
#
# Contract: the caller MUST have REPO_ROOT set before sourcing. On return,
# $PYTHON points at the resolved interpreter and $ENV_NAME names the conda env.
if [ -z "${SDK_CONDA_ENV:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    SDK_CONDA_ENV="$(grep -E '^SDK_CONDA_ENV=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
ENV_NAME="${SDK_CONDA_ENV:-cogwheel_310}"
CONDA_PREFIX_LOCAL="$(conda info --base 2>/dev/null)/envs/$ENV_NAME"
if [ -x "$CONDA_PREFIX_LOCAL/bin/python" ]; then
    PYTHON="$CONDA_PREFIX_LOCAL/bin/python"
else
    PYTHON="python3"
fi
