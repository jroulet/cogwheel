#!/usr/bin/env bash
# PreToolUse hook: redirect native Read/Grep/Glob/Edit/Write/Bash to Serena equivalents
# for project files. Non-project files, images, PDFs, and notebooks pass through.
#
# Serena toolkit quick reference:
#
# === Reading & Navigation ===
# mcp__serena__read_file                    (relative_path, start_line, end_line)
# mcp__serena__get_symbols_overview         (relative_path, depth)
# mcp__serena__find_symbol                  (name_path_pattern, include_body, include_info, depth, substring_matching)
# mcp__serena__find_referencing_symbols     (name_path, relative_path)
# mcp__serena__list_dir                     (relative_path, recursive)
# mcp__serena__find_file                    (file_mask, relative_path)
# mcp__serena__search_for_pattern           (substring_pattern, relative_path, paths_include_glob, context_lines_before/after)
#
# === Editing ===
# mcp__serena__replace_content              (relative_path, needle, repl, mode='regex'|'literal')
# mcp__serena__replace_symbol_body          (name_path, relative_path, body)
# mcp__serena__replace_lines                (relative_path, start_line, end_line, content)
# mcp__serena__delete_lines                 (relative_path, start_line, end_line)
# mcp__serena__insert_at_line               (relative_path, line, content)
# mcp__serena__insert_after_symbol          (name_path, relative_path, body)
# mcp__serena__insert_before_symbol         (name_path, relative_path, body)
# mcp__serena__rename_symbol                (name_path, relative_path, new_name)
# mcp__serena__create_text_file             (relative_path, content)
#
# === Shell ===
# mcp__serena__execute_shell_command        (command, cwd)

input=$(cat)
tool_name=$(jq -r '.tool_name' <<< "$input")

# Lazy-evaluate PROJECT only when needed (git rev-parse is slow on NFS)
_project=""
get_project() {
  if [[ -z "$_project" ]]; then
    _project="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
  fi
  echo "$_project"
}

is_project_file() {
  local PROJECT; PROJECT="$(get_project)"
  local fp="$1"
  # Resolve relative paths against PROJECT so spawned agents that pass
  # e.g. "src/foo.py" are caught by the gate. Without this the relative
  # form never matches "$PROJECT/*" and native Read/Edit/Write passed
  # through silently (2026-04-23 finding).
  if [[ "$fp" != /* ]]; then
    fp="$PROJECT/$fp"
  fi
  [[ "$fp" == "$PROJECT"/* ]] && \
  [[ "$fp" != "$PROJECT/.claude"/* ]]
}

# Gitignored files (CLAUDE.local.md, .env, etc.) can't be read by Serena
# because it refuses ignored paths. Allow native tools through for these.
is_gitignored() {
  local fp="$1"
  local PROJECT; PROJECT="$(get_project)"
  local rel="${fp#$PROJECT/}"
  git -C "$PROJECT" check-ignore -q "$rel" 2>/dev/null
}

deny() {
  local reason="$1"
  if [[ "${AGENT_PROVIDER:-}" == "codex" && -n "${CODEX_SERENA_URL:-}" ]]; then
    reason="${reason//mcp__serena__/mcp__serena_build__}"
  fi
  jq -n --arg reason "$reason" '{
    "hookSpecificOutput": {
      "hookEventName": "PreToolUse",
      "permissionDecision": "deny",
      "permissionDecisionReason": $reason
    }
  }'
}

case "$tool_name" in
  Read)
    fp=$(jq -r '.tool_input.file_path // ""' <<< "$input")
    # Allow images, PDFs, notebooks through native Read (Serena can't handle these)
    if [[ "$fp" =~ \.(png|jpg|jpeg|gif|svg|pdf|ipynb)$ ]]; then
      exit 0
    fi
    # Allow gitignored files through (Serena refuses them)
    if is_gitignored "$fp"; then
      exit 0
    fi
    if is_project_file "$fp"; then
      deny "USE SERENA for project files. Pick the right tool:
- mcp__serena__read_file (relative_path) for full file reads
- mcp__serena__get_symbols_overview (relative_path, depth) for code structure overview
- mcp__serena__find_symbol (name_path_pattern, include_body=true) to read a specific symbol's source
- mcp__serena__find_symbol (name_path_pattern, include_info=true) for docstring/signature
- mcp__serena__find_referencing_symbols (name_path, relative_path) to find all references
All use relative paths from project root."
    fi
    ;;
  Grep)
    gp=$(jq -r '.tool_input.path // ""' <<< "$input")
    if [[ -z "$gp" ]] || [[ "$gp" == "$(get_project)"* ]]; then
      deny "USE SERENA for project searches. Pick the right tool:
- mcp__serena__search_for_pattern (substring_pattern, relative_path, paths_include_glob, context_lines_before/after) for regex text search
- mcp__serena__find_symbol (name_path_pattern, substring_matching=true) for symbol name search
- mcp__serena__find_referencing_symbols (name_path, relative_path) to find all usages of a symbol"
    fi
    ;;
  Glob)
    gp=$(jq -r '.tool_input.path // ""' <<< "$input")
    if [[ -z "$gp" ]] || [[ "$gp" == "$(get_project)"* ]]; then
      deny "USE SERENA for project file discovery. Pick the right tool:
- mcp__serena__find_file (file_mask, relative_path='.') to find files by pattern
- mcp__serena__list_dir (relative_path='.', recursive=true) to list directory contents
- mcp__serena__get_symbols_overview (relative_path) to understand a file's structure"
    fi
    ;;
  Edit)
    fp=$(jq -r '.tool_input.file_path // ""' <<< "$input")
    if is_gitignored "$fp"; then exit 0; fi
    if is_project_file "$fp"; then
      deny "USE SERENA for project edits. Pick the right tool:
- mcp__serena__replace_content (relative_path, needle, repl, mode='regex'|'literal') for targeted find-and-replace
- mcp__serena__replace_symbol_body (name_path, relative_path, body) to replace an entire symbol definition
- mcp__serena__replace_lines (relative_path, start_line, end_line, content) to replace a range of lines
- mcp__serena__delete_lines (relative_path, start_line, end_line) to delete lines
- mcp__serena__insert_at_line (relative_path, line, content) to insert at a specific line
- mcp__serena__insert_after_symbol / mcp__serena__insert_before_symbol for symbol-relative insertion
- mcp__serena__rename_symbol (name_path, relative_path, new_name) for LSP-powered rename
All use relative paths from project root."
    fi
    ;;
  Write)
    fp=$(jq -r '.tool_input.file_path // ""' <<< "$input")
    if is_gitignored "$fp"; then exit 0; fi
    if is_project_file "$fp"; then
      deny "USE SERENA for project file creation. Pick the right tool:
- mcp__serena__create_text_file (relative_path, content) to create or overwrite a file
- mcp__serena__insert_after_symbol / mcp__serena__insert_before_symbol if adding near a known symbol
- mcp__serena__insert_at_line (relative_path, line, content) to insert at a specific line
All use relative paths from project root."
    fi
    ;;
  Bash)
    cmd=$(jq -r '.tool_input.command // ""' <<< "$input")
    # Allow git, gh, conda, brew, and common non-destructive system commands
    # through native Bash. Read-only inspectors (ls, stat, wc, etc.) have no
    # Serena equivalent or don't benefit from LSP; routing them through Serena
    # just costs a deny+retry round-trip.
    #
    # Normalize command-substitutions ($(...) and `...`) to a no-space
    # placeholder so they don't break the VAR=value stripper below. The value
    # of `LOG=.claude/sdk/logs/build_$(date +%Y%m%d_%H%M%S).log` contains a
    # space inside the $(...), and the stripper needs the value to be a
    # single unbroken token. The placeholder keeps the safety judgment local
    # to the command boundary, not the substitution contents.
    stripped="$cmd"
    while [[ "$stripped" =~ \$\([^\(\)]*\) ]]; do
      stripped="${stripped/${BASH_REMATCH[0]}/X}"
    done
    while [[ "$stripped" =~ \`[^\`]*\` ]]; do
      stripped="${stripped/${BASH_REMATCH[0]}/X}"
    done
    # Strip leading VAR=value assignments (bash's standard per-command env
    # prefix) so templates like `LOG=/tmp/x conda run ...` are judged on the
    # actual command, not the variable assignment.
    while [[ "$stripped" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+ ]]; do
      stripped="${stripped#${BASH_REMATCH[0]}}"
    done
    # Python project: the default passthrough set below is sufficient (python/pip
    # are wrapped by the conda hook, not bypassed here). Add tools here if needed.
    if [[ "$stripped" =~ ^(git|gh|conda|brew|npm|npx|which|chmod|mkdir|ls|stat|wc|pwd|date|env|printenv|df|du|file|ps|pgrep|diff|kill|pkill)([[:space:]]|$) ]]; then
      exit 0
    fi
    # Allow project-owned shell scripts under .claude/sdk/ and .claude/hooks/
    # (these are our own code; they already route through the safety model).
    if [[ "$stripped" =~ ^\.claude/(sdk|hooks)/[A-Za-z0-9_.-]+\.sh([[:space:]]|$) ]] \
       || [[ "$stripped" =~ ^\.codex/build([[:space:]]|$) ]] \
       || [[ "$stripped" =~ ^\.codex/hooks/[A-Za-z0-9_.-]+\.sh([[:space:]]|$) ]]; then
      exit 0
    fi
    deny "USE SERENA for shell commands: mcp__serena__execute_shell_command (command, cwd).
Exception: git, gh, conda, brew, and common read-only system commands
(ls, stat, wc, pwd, date, env, df, du, file, ps, pgrep, diff, kill, pkill)
and project scripts under .claude/sdk/ or .claude/hooks/ may use Bash directly.
Leading VAR=value env assignments are stripped before matching.
SDK builds: do NOT hand-roll the launch — use
  .claude/sdk/launch_build.sh <task_slug> <prompt_file> [stale_seconds]
then arm the Monitor printed in the log header (health = log mtime, not pgrep)."
    ;;

  # ── Within-Serena shell command hygiene ──────────────────────────────
  # Block shell commands inside execute_shell_command that have dedicated
  # Serena symbolic tools (cat->read_file, grep->search_for_pattern, etc.)
  mcp__serena__execute_shell_command|mcp__serena_build__execute_shell_command)
    cmd=$(jq -r '.tool_input.command // ""' <<< "$input")
    # Strip leading cd ... && or whitespace
    clean_cmd=$(echo "$cmd" | sed 's/^cd [^&]*&& *//' | sed 's/^ *//')
    # Fast exit: python/conda commands don't need redirection to Serena tools.
    # These are arbitrary code execution, not file-manipulation primitives.
    if echo "$clean_cmd" | grep -qE '^(python3?|conda|pip|pytest) '; then
      exit 0
    fi
    # Allow commands targeting .claude/ — Serena can't see gitignored files
    if echo "$cmd" | grep -q '\.claude'; then
      exit 0
    fi
    # Allow shell commands targeting absolute paths outside the project
    # (e.g., /data/shared/, /home/user/) — Serena can't see those.
    # Two checks: (1) command is a read-only tool, (2) any argument is
    # an absolute path (space followed by /). Doesn't try to parse flags.
    if echo "$clean_cmd" | grep -qE '^(ls|cat|head|tail|find|wc)\b' && echo "$clean_cmd" | grep -q ' /'; then
      exit 0
    fi
    # cat / head / tail -> read_file
    if echo "$clean_cmd" | grep -qE '^(cat|head|tail) '; then
      deny "USE mcp__serena__read_file (relative_path, start_line, end_line) instead of cat/head/tail.
For symbol-level reads: mcp__serena__find_symbol (name_path_pattern, include_body=true)."
    fi
    # grep / rg -> search_for_pattern
    if echo "$clean_cmd" | grep -qE '^(grep|rg|egrep|fgrep) '; then
      deny "USE mcp__serena__search_for_pattern (substring_pattern, relative_path, context_lines_before/after) instead of grep.
For symbol searches: mcp__serena__find_symbol (name_path_pattern, substring_matching=true).
For reference tracking: mcp__serena__find_referencing_symbols (name_path, relative_path)."
    fi
    # find -> find_file / list_dir
    if echo "$clean_cmd" | grep -qE '^find '; then
      deny "USE mcp__serena__find_file (file_mask, relative_path) instead of find.
For directory listing: mcp__serena__list_dir (relative_path, recursive=true)."
    fi
    # ls -> list_dir
    if echo "$clean_cmd" | grep -qE '^ls '; then
      deny "USE mcp__serena__list_dir (relative_path, recursive=false) instead of ls."
    fi
    # (wc -l was previously denied here, redirecting to get_symbols_overview
    # — wrong tool for a line-count question. Drop the deny: wc is in the
    # top-level Bash whitelist now, and inside Serena's shell layer it's
    # just a read-only inspector with no Serena equivalent.)
    ;;
esac
