# Tidy Long-Term Knowledge

- When shell/autoflake/`ast.parse` tools are denied or absent, fall back
  to manual inspection (import ordering, blank-line rules, unused-import
  cross-check by hand against usages) and state the fallback explicitly
  rather than skipping the pass.
- Style checklist for this codebase: 2 blank lines between top-level
  def/class, 1 blank line between methods, zero 3+ blank-line runs, zero
  whitespace-only lines; import order stdlib -> third-party -> local
  with blank lines between groups; `from __future__ import annotations`
  first when present.
- Check mechanically, not visually: line length via `awk 'length()>79'`
  line-by-line, and wrapped-import continuation alignment by measuring
  the indent against the opening-paren column. Correct import alignment
  does not imply body lines are within the column limit.
- `__init__.py` re-export lines need a blank line separating the module
  docstring from the first `from .module import Name` line, matching
  sibling packages.
- Don't normalize style patterns the rubric doesn't mandate (e.g. blank
  line between module docstring and __future__ import — the codebase is
  split); note the inconsistency for a dedicated pass, leave as-is.
- Known codebase split (do NOT normalize): __init__.py files mix absolute
  (`from cogwheel.lensing.x import`) and relative (`from .x import`)
  re-exports; the rubric mandates neither, so flag for a dedicated pass.
- Within the local-import group the codebase orders by dependency/layer,
  NOT alphabetically (verified against cogwheel/posterior.py and
  likelihood/__init__.py) — only the 4 broad groups are mandatory; don't
  over-reorder within a group. Within-package sibling imports
  (cogwheel.pkg.module) sort AFTER the broader cogwheel.* layer imports.
- A newly-written module can legitimately require zero edits — verify
  the checklist per file rather than assuming something must be wrong.
- If a file outside your lane (e.g. a test file) is named in the task's
  file list, spot-check it read-only but leave edits to the owning agent
  (Test Dev) — never touch test files even when explicitly listed.
