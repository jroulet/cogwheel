---
section: Backlog
---
- **Coder shell-call denials: ROOT-CAUSED; verify the fix empirically, then
  propagate** `[housekeeping]` — the intermittent bare denial (`"The user
  doesn't want to take this action right now. STOP..."`) was the auto-mode
  permission classifier FAILING CLOSED on its own transient errors: its prompt
  embeds the agent's whole transcript, so deep agents fail more (measured: 106
  denials / 59 sessions; zero in any session's first two calls, median at call
  14; 38% were the agent's final act). SDK agents never hit the allowlist
  fast-path because they load `setting_sources=["user"]` (deliberate — keeps
  project hooks out of the serena-crash fallback) and the user scope had no
  permissions block. FIX (4a6e310): `.claude/settings.agents.json` — an
  agents-only allowlist (38 portable rules, no hooks, no MCP keys) passed via
  `ClaudeAgentOptions(settings=...)`; allowlisted tools bypass the classifier.
  REMAINING: (1) confirm empirically — next full build should show ~zero bare
  denials on allowlisted tools (compare against the 106/59 baseline via the
  transcript grep in META_PLAN); (2) after cogwheel proves it, propagate to
  teja-force + gw with the rest of the validated batch; (3) optional
  defense-in-depth: a single delayed orchestrator retry on the bare-denial
  signature (the classifier's own reason text says "usually transient —
  retrying often succeeds") for tools that stay classifier-exposed. Full
  diagnosis with the decompiled fail-closed path: META_PLAN + the deep-dive
  report (2026-07-16).
