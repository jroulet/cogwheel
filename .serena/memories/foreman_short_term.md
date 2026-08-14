# Foreman-Lite Short-Term Observations

- INS-1-002 (operator.py `_ghost_ppgo_amplification`, 2026-08-13): added
  `if len(real_images) == 0: return None` immediately after the decay-gate
  check and before the `min(... for x_a in real_images)` separation-gate
  computation, exactly per the finding's suggested fix — a defensive-only
  guard since an admitted continuable ghost currently always has >=2 real
  images. Verified placement via search_for_pattern (guard sits right
  before `x_c = ghost.position` / `separation = min(...)`) and ast.parse
  for syntax. Note: mcp__serena__execute_shell_command silently used a
  different cwd (cogwheel-claude-dev worktree) than the project path
  quoted in the task file path — omitting the `cwd` param let it default
  correctly and find the file by relative path; passing an explicit
  absolute cwd guess failed with FileNotFoundError. Prefer omitting cwd
  and using relative paths matching find_file's output.
