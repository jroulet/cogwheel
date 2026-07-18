# Tidy Short-Term Observations

- CORRECTION (2026-07-18, later pass): the earlier "likelihood.py is
  fully zero-edit-compliant" note below was wrong — a re-check found
  one 81-char line in `_evaluate_envelope`/`_envelope_loo_nodes`
  (`scale = max(float(np.max(np.abs(ftot_nodes))), _ENVELOPE_SCALE_FLOOR)`)
  over the 79-col limit; wrapped it. Lesson: line-length must be
  checked line-by-line (awk length()), not just visually skimmed —
  import alignment being correct doesn't imply body lines are too.
- cogwheel/lensing/likelihood.py (checked 2026-07-18): multi-line
  `from cogwheel.lensing.* import (...)` continuations align under the
  opening paren correctly (verified by measuring indent vs paren
  column, don't trust eyeballing wrapped imports), no unused imports,
  no blank-line-run violations. See correction above re: line length.
- Codebase-wide check (2026-07-18): 13 of 14 lensing modules
  (incl. channels.py, likelihood.py) have NO blank line between the
  module docstring and `from __future__ import annotations`; only
  _gauge.py and test_lensing_channels.py insert one. This is not a
  rubric violation either way (rubric doesn't mandate it) — left as-is,
  flag for a future dedicated consistency pass if ever wanted.
- channels.py and _gauge.py (checked 2026-07-18): fully rubric-compliant
  as-is — spacing, import layering/grouping, no unused imports, all
  multi-line import continuations correctly aligned to the opening
  paren column. Zero edits needed.
- autoflake is not installed in this environment (`python3 -m autoflake`
  -> ModuleNotFoundError); fell back to manual grep-count cross-check of
  each imported name's usages instead.
