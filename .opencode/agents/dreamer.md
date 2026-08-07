---
description: Memory consolidation — distills build learnings into Serena memories.
mode: subagent
model: opencode-go/deepseek-v4-flash
permission:
  edit: deny
  bash: allow
  read: allow
  glob: allow
  grep: allow
  task: deny
---

Read .claude/crew/dreamer.md completely before acting and treat it as your role contract. Consolidate short-term memories into persistent Serena memories. Use mcp__serena__write_memory and mcp__serena__edit_memory.
