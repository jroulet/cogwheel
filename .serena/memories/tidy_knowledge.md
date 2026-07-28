# Tidy Long-Term Knowledge

- When shell/autoflake/`ast.parse` tools are denied or absent (confirmed
  absent in `cogwheel-newlal`: pylint, pyflakes, autoflake), fall back to
  manual inspection (import ordering, blank-line rules, unused-import
  cross-check by hand) plus a per-line length/whitespace scan, and state
  the fallback explicitly rather than skipping the pass.
- Style checklist for this codebase: 2 blank lines between top-level
  def/class, 1 blank line between methods, zero 3+ blank-line runs, zero
  whitespace-only lines; import order stdlib -> third-party -> local
  with blank lines between groups; `from __future__ import annotations`
  first when present. House line limit is 79 and is genuinely universal —
  measure the WHOLE tree first (not just the target dir) before deciding
  whether an offending pattern is house style or local drift.
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
  NOT alphabetically — only the 4 broad groups are mandatory; don't
  over-reorder within a group. Within-package sibling imports sort AFTER
  the broader cogwheel.* layer imports.
- A newly-written module can legitimately require zero edits — verify
  the checklist per file rather than assuming something must be wrong.
- If a file outside your lane (e.g. a test file) is named in the task's
  file list, spot-check it read-only but leave edits to the owning agent.
- Prose rewrap: preserve two-spaces-after-period sentence spacing — a
  naive `' '.join(words)` rewrap collapses it and triples the diff. Safe
  recipe: refill only the overflowing paragraph tail, keep (word,
  separator) pairs, stop at blank lines/bullet markers/banner rules/a
  lone closing `"""`; verify via whitespace-normalized word-stream diff
  against `git show HEAD:<file>` (catches off-by-one slices eating an
  adjacent code line). Bullet continuation lines need the DEEPER
  hanging-indent prefix detected from the next line, only when the
  target line is itself a bullet.
- Some 80-81 char lines (banner rules, section headers) are unfixable
  without content change — wrapping merges the header into prose or
  strands a fragment; leave and report, a future pass re-flagging it is
  correct, not a defect. A `\n\n# banner \n\ndef` pattern is correct PEP 8
  (2 blanks precede the banner), not a spacing violation. An AST-based
  unused-import checker always flags `from __future__ import annotations`
  as unused (the name is never referenced) — never strip it.
