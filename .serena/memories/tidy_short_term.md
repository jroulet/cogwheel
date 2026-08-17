# Tidy short-term observations (2026-08-17, beat-free build production files)

- The 109 advisory long lines in `surrogate_training.py` / `surrogate.py` /
  `surrogate_census.py` were ~95% prose (76+2 comment lines, 28 docstring
  lines) only 1-5 columns over; just two were code (`surrogate_training.py`
  span-continuation, now parenthesized instead of backslash-continued;
  `surrogate_census.py` trailing comment moved above the statement) and one a
  concatenated string literal (re-balanced across pieces, byte-identical
  value).
- Safe bulk procedure that worked: classify every reported line with
  `tokenize` (COMMENT / STRING / CODE) before touching it, spill-forward
  refill only true COMMENT lines, then prove harmlessness two ways:
  token-stream equality ignoring COMMENT/NL, and `ast.dump` equality with
  string constants whitespace-normalized (catches docstring edits changing
  wording).  Both checks passed on all three files.
- Watch the refill seams: greedy word-wrap will split inline math/expressions
  across comment lines (`u = d**(2/3)`, `xi ~ (w * eta^{3/2})^{2/3}`,
  `parity != 1`, ```` ``si = 0`` ````).  Four such seams needed manual
  polish; grep the diff for `=$`, `!=$`, `(w *$`-style line endings after any
  automated rewrap.
- Import audit: no unused imports.  `LobeExteriorChart` (training) and
  `_SaddleLobeAdmission` (surrogate, TYPE_CHECKING-only by design) show zero
  Name-node uses but live in quoted annotations -- an AST Name counter alone
  under `from __future__ import annotations` is not an unused-import oracle.
- The other ~58 advisory files were untouched by recent builds and were
  skipped on driver instruction (diff noise).
