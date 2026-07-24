#!/usr/bin/env bash
# Supply the Codex-specific half of the shared project orientation.

jq -n '{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "Codex project integration is active. AGENTS.md is the canonical shared instruction file; CLAUDE.md is its compatibility symlink. Interactive Serena is configured in .codex/config.toml with the codex context. A .codex/build run starts one separate shared Serena SSE server for all build roles; .claude/build intentionally remains Claude-default."
  }
}'
