---
date: 2026-07-27
---
### Agent-infra hardening: memory budget, staging allowlist, test-tier law (.claude/ only)

`ClaudeAgentOptions` gains `agent_name` compatibility; a TEST_TIER_LAW
prompt section is added for the architect, test_dev, and inspector
roles; size-triggered Phase 3 memory consolidation is added; the
new-file staging allowlist widens to cover `.claude/sdk`, `.claude/
handoff`, and `.claude/crew` (paths outside the prior prefixes were
silently dropped at commit while the commit message advertised them);
and a TOTAL inlined-memory budget (60 KB, with a 2 KB per-file floor)
is layered on top of the existing per-file 24 KB cap, since the
per-file cap alone permitted an agent reading many memories (e.g. the
Dreamer's 16) to blow the argv limit. These changes are agent-
infrastructure only, under `.claude/`, and are excluded from the
`main`-branch sync.
