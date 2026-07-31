#!/usr/bin/env bash
# Behavioural probe for the per-process Serena readiness gate.

set -eu

repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
state_dir="$(mktemp -d)"
trap 'rm -rf "$state_dir"' EXIT

native_matcher="Read|Grep|Glob|Edit|Write|Bash"
for matcher in \
  "$(jq -r '.hooks.PreToolUse[0].matcher' "$repo_root/.codex/hooks.json")" \
  "$(jq -r '.hooks.PreToolUse[2].matcher' "$repo_root/.codex/hooks.json")"; do
  if [[ "$matcher" != *"$native_matcher"* ]]; then
    echo "FAIL: native Codex tool matcher does not cover $native_matcher" >&2
    exit 1
  fi
done

export CODEX_SERENA_READY_DIR="$state_dir"
export CODEX_THREAD_ID="serena-ready-probe"

before="$("$repo_root/.codex/hooks/require-serena-ready.sh")"
if ! grep -q '"permissionDecision": "deny"' <<< "$before"; then
  echo "FAIL: native project tools were allowed before Serena initialization" >&2
  exit 1
fi

"$repo_root/.codex/hooks/mark-serena-ready.sh"
after="$("$repo_root/.codex/hooks/require-serena-ready.sh")"
if [[ -n "$after" ]]; then
  echo "FAIL: native project tools remained blocked after Serena initialization" >&2
  exit 1
fi

unset CODEX_THREAD_ID
export CODEX_SERENA_URL="http://localhost:8324/mcp"
export CODEX_SERENA_READY_KEY="serena-build-ready-probe"
build_before="$("$repo_root/.codex/hooks/require-serena-ready.sh")"
if ! grep -q 'mcp__serena_build__initial_instructions' <<< "$build_before"; then
  echo "FAIL: build native project tools were allowed before build Serena initialization" >&2
  exit 1
fi

"$repo_root/.codex/hooks/mark-serena-ready.sh"
build_after="$("$repo_root/.codex/hooks/require-serena-ready.sh")"
if [[ -n "$build_after" ]]; then
  echo "FAIL: build native project tools remained blocked after build Serena initialization" >&2
  exit 1
fi

disabled="$(AGENT_DISABLE_SERENA=1 "$repo_root/.codex/hooks/require-serena-ready.sh")"
if [[ -n "$disabled" ]]; then
  echo "FAIL: explicit Serena opt-out remained blocked" >&2
  exit 1
fi

echo "PASS: Serena readiness gate blocks every interactive thread and build role until initialization"
