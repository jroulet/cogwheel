# Foreman Short-Term Notes

- INS-4-002 (test_lensing_likelihood.py): fixed two stale docstring
  references to `kernel_subsamples default 8` → `default 2`, matching
  production `_DEFAULT_KERNEL_SUBSAMPLES` in cogwheel/lensing/likelihood.py.
  Only touched the two exact spots named in the finding (class docstring
  ~l.352, test method docstring ~l.378) — did NOT rewrite the surrounding
  narrative that still frames `kernel_subsamples=2` as an "aliasing"
  edge-secant pathology, even though production now defaults to 2 (which
  the module docstring in likelihood.py says is now accurate, not
  pathological). That deeper narrative contradiction looks like it
  belongs to the broader INS-4-001 rebase, not this trivial docstring fix
  — flagging for whoever owns INS-4-001.
- ast.parse syntax-check via mcp__serena__execute_shell_command was
  denied twice (bare "user doesn't want to take this action" signature,
  no reason given) — per task instructions, retried once, then stopped
  and verified visually via read_file instead. Edits were pure
  docstring-string replacements (replace_content, literal mode), low
  risk, no structural change.
