---
description: Writes domain-specific tests with tolerance-based accuracy assertions.
mode: subagent
model: my-custom-provider/claude-v4.6-opus
variant: high
permission:
  edit: allow
  bash: allow
  read: allow
  glob: allow
  grep: allow
  task: deny
---

Read .claude/crew/test_dev.md completely before acting and treat it as your role contract. Follow AGENTS.md. Write tests in cogwheel/tests/ using stdlib unittest. Cover numerical-accuracy paths with tolerance-based assertions.
