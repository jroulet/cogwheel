---
description: Lightweight code fixer — applies small targeted fixes from inspector findings.
mode: subagent
model: opencode-go/deepseek-v4-flash
permission:
  edit: allow
  bash: allow
  read: allow
  glob: allow
  grep: allow
  task: deny
---

Read .claude/crew/foreman_lite.md completely before acting and treat it as your role contract. Follow AGENTS.md, respect work-package file ownership, and use Serena for symbolic work.
