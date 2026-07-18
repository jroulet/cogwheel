# Tidy Short-Term Observations

- `cogwheel/lensing/likelihood.py` (2026-07-18 pass): already fully
  rubric-compliant (import grouping/alignment, blank-line spacing,
  no unused imports, valid syntax) — zero edits made. Confirmed via
  manual mechanical checks (python line-length/indent/blank-run
  scripts) since `autoflake` was not installed in the environment;
  noting the fallback here per policy.

- `cogwheel/lensing/{__init__.py,posterior.py,prior.py}` (2026-07-18
  pass): also fully rubric-compliant already — zero edits made after
  mechanical checks (blank-line runs, whitespace-only lines, 2/1
  blank-line spacing, import-group boundaries, manual unused-import
  cross-check; autoflake absent, same fallback). Tried reordering
  `posterior.py`'s local imports (`cogwheel.posterior` vs
  `cogwheel.lensing.chang_refsdal.*`) into alphabetical order but
  reverted: cross-checked against `cogwheel/posterior.py`,
  `cogwheel/likelihood/__init__.py`, and
  `cogwheel/likelihood/marginalization/__init__.py` and found the
  codebase does NOT enforce alphabetical sub-ordering within the
  "local" import group — real files order by dependency/layer, not
  alphabetically. Only the 4 broad groups (stdlib/third-party/
  local/relative) and the explicit example lines in the rubric are
  mandatory; don't over-reorder within a group.
