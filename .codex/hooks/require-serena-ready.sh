#!/usr/bin/env bash
# Block native Codex project tools until this process has initialized Serena.

set -u

[[ "${AGENT_DISABLE_SERENA:-}" == "1" ]] && exit 0

ready_key="${CODEX_SERENA_READY_KEY:-}"
serena_tool="mcp__serena_build__initial_instructions"
if [[ -z "$ready_key" && -z "${CODEX_SERENA_URL:-}" ]]; then
  ready_key="${CODEX_THREAD_ID:-}"
  serena_tool="mcp__serena__initial_instructions"
fi
[[ -z "$ready_key" ]] && exit 0
ready_key="${ready_key//[^A-Za-z0-9._-]/_}"
state_dir="${CODEX_SERENA_READY_DIR:-/tmp/codex-serena-ready}"
ready_file="$state_dir/$ready_key.ready"

[[ -f "$ready_file" ]] && exit 0

jq -n --arg serena_tool "$serena_tool" '{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Serena is not initialized for this Codex process. Use tool_search to discover \($serena_tool), then call that tool before using native project tools."
  }
}'
