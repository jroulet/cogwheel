#!/usr/bin/env bash
# Mark the current Codex process ready after Serena initialization.

set -eu

[[ "${AGENT_DISABLE_SERENA:-}" == "1" ]] && exit 0

ready_key="${CODEX_SERENA_READY_KEY:-}"
if [[ -z "$ready_key" && -z "${CODEX_SERENA_URL:-}" ]]; then
  ready_key="${CODEX_THREAD_ID:-}"
fi
[[ -z "$ready_key" ]] && exit 0
ready_key="${ready_key//[^A-Za-z0-9._-]/_}"
state_dir="${CODEX_SERENA_READY_DIR:-/tmp/codex-serena-ready}"

umask 077
mkdir -p "$state_dir"
touch "$state_dir/$ready_key.ready"
