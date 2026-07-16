---
section: Backlog
---
- **Find what actually denies coder shell calls mid-build** `[housekeeping]` — an
  intermittent `"The user doesn't want to take this action right now. STOP ..."`
  comes back as a `tool_result` on a Coder's `execute_shell_command` (and on the
  native `Bash` fallback) partway into a build, ending the work package with no
  files. Unexplained; it has blocked every Build-1b launch. Ruled out by
  measurement: the sandbox `/tmp` allowlist (0/5 denied with it, 0/5 without),
  position, command content, hooks, user-scope deny rules, coder memory.
  Evidence points at session depth/size — a shell call on turn 1 is never denied
  (0/10), one ~8 calls in is. Full write-up, ruled-out list, next experiment and
  the known-good workaround: `.claude/handoff/lensing/META_PLAN.md`; harness:
  `scratchpad/denial_rate.py` / `depth_rate.py`.
