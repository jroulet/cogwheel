# Tidy Long-Term Knowledge

- When shell/autoflake/`ast.parse` tools are denied by the permission
  system, fall back to manual inspection (import ordering, blank-line
  rules, unused-import cross-check by hand against usages) and state the
  fallback explicitly rather than skipping the pass.
- Style checklist for this codebase: 2 blank lines between top-level
  def/class, 1 blank line between methods within a class, zero 3+
  blank-line runs, zero whitespace-only lines; import order is stdlib ->
  third-party -> local with a blank line between groups, and
  `from __future__ import annotations` first when present.
- `__init__.py` re-export lines need a blank line separating the module
  docstring from the first `from .module import Name` line, matching
  sibling packages' convention.
- A newly-written module can legitimately require zero edits — verify
  the checklist per file rather than assuming something must be wrong.
