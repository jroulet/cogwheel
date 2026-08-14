# Foreman-Lite Short-Term Observations

- INS-1-002 (2026-08-14): fixed by a plain `git add scripts/measure_saddle_eta_floor.py`
  in the actual repo worktree `/home/tejaswi/Work/cogwheel-claude-dev` (NOT
  `/home/tejaswi/Work/cogwheel`, which is a separate, unrelated checkout with no
  saddle_eta files). The Bash sandbox blocks `cd`+chained commands and `grep`
  outright ("USE SERENA for shell commands") — git/ls/stat/etc. must be invoked
  directly with `-C <path>` (no leading cd), and content search must go through
  `mcp__serena__search_for_pattern` instead of grep. Syntax-checked the newly
  staged script via `ast.parse` through `mcp__serena__execute_shell_command`
  with explicit `cwd` (the earlier top-level Bash python call was also blocked
  by the same sandbox exception).
