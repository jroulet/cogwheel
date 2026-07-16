# Architect Long-Term Knowledge

- When the source task/prompt is unreadable in-session (sandbox confines
  Serena file tools to the project root, built-in Read/Bash disabled,
  `execute_shell_command` absent, orchestrator log gitignored/blocked),
  do NOT fabricate a task from guesswork. Emit a plan with empty
  `work_packages` and a summary flagging the blocker for a human re-run
  (stage the prompt inside the project root, or enable Read/Bash for
  Architect). Confirm via more than one independent attempt before
  concluding it's a real blocker, not a one-off.
- Before planning a docs-only task, check whether the target already
  substantially satisfies the goal (e.g. the module already has a
  docstring, functions already state the convention). Plan the minimal
  diff (append/expand the actual gap), not a rewrite-from-scratch — a
  full rewrite risks drifting from adjacent, already-correct docstrings.
- Simplifier verdict pattern: prefer the leanest correct response over
  speculative extra work — don't add a no-op verification work package
  "just in case," and don't guess at unstated requirements when the real
  gap can be scoped precisely.
