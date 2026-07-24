#!/usr/bin/env bash
# PostToolUse hook: auto-create read markers when the Professor writes to a
# topic memory.
#
# Fires after `mcp__serena__write_memory` calls. If the memory_name matches
# `professor/<topic>` (excluding `professor/read.d/...` itself), extract any
# arxiv IDs from the memory content and create marker files at
# `.serena/memories/professor/read.d/<arxiv_id>`.
#
# Idempotent: marker files are touched, not appended.
#
# This is the authoritative read-tracking mechanism. The --mark-read CLI in
# sync_professor_papers.py is a convenience for the "Professor read but
# decided nothing novel to add" case (where no synthesis write fires).

input=$(cat)
tool_name=$(jq -r '.tool_name // ""' <<< "$input")

# Only fire on Serena's write_memory tool.
[[ "$tool_name" == "mcp__serena__write_memory"
   || "$tool_name" == "mcp__serena_build__write_memory" ]] || exit 0

# Only fire on professor topic memory writes.
memory_name=$(jq -r '.tool_input.memory_name // ""' <<< "$input")
case "$memory_name" in
  professor/read.d/*) exit 0 ;;       # Don't recurse on marker writes
  professor/*) ;;                       # Topic memory — proceed
  *) exit 0 ;;                          # Not a professor topic memory — skip
esac

# Get the content that was written.
content=$(jq -r '.tool_input.content // ""' <<< "$input")
[[ -z "$content" ]] && exit 0

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)" || exit 0
READ_DIR="$REPO_ROOT/.serena/memories/professor/read.d"
mkdir -p "$READ_DIR"

# Extract arxiv IDs (YYYY.NNNNN format, 4-5 digit suffix).
ids=$(echo "$content" | grep -Eo '[0-9]{4}\.[0-9]{4,5}' | sort -u)
[[ -z "$ids" ]] && exit 0

# Touch a marker file for each ID (no-op if already exists).
for id in $ids; do
  marker="$READ_DIR/$id"
  [[ -f "$marker" ]] || touch "$marker"
done

# Silent success — hooks shouldn't be chatty. The marker file is the signal.
exit 0
