#!/usr/bin/env bash
# Codex-compatible form of the Claude settings hook that routes bare
# python/pip shell commands through the configured project conda environment.

input="$(cat)"
tool_name="$(jq -r '.tool_name // ""' <<< "$input")"
case "$tool_name" in
  Bash|mcp__serena__execute_shell_command|mcp__serena_build__execute_shell_command) ;;
  *) exit 0 ;;
esac

command="$(jq -r '.tool_input.command // ""' <<< "$input")"
if ! grep -Eq '(^|[[:space:]])(python3?|pip3?)([[:space:]]|$)' <<< "$command"; then
  exit 0
fi
if grep -Eq '(^|[[:space:]])conda([[:space:]]+run)?([[:space:]]|$)' <<< "$command"; then
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
# .env is authoritative — always read it (a stale shell export may be wrong).
env_name="${SDK_CONDA_ENV:-}"
if [[ -f "$repo_root/.env" ]]; then
  _env_val="$(grep -E '^SDK_CONDA_ENV=' "$repo_root/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
  [[ -n "$_env_val" ]] && env_name="$_env_val"
fi
env_name="${env_name:-cogwheel_310}"

jq -n --arg command "conda run -n $env_name $command" '{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "updatedInput": {"command": $command}
  }
}'
