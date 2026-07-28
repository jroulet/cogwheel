# Tidy Short-Term Observations

## 2026-07-27 — post-commit advisory (d0dc6da), 8-day backlog

- **House line limit is really 79 and really universal.** All 47 non-lensing,
  non-test `cogwheel/*.py` files have ZERO lines over 79 (`max-line-length = 79`
  in `pyproject.toml`). `cogwheel/lensing/**` was the only offender: 153 lines
  over, mostly at exactly 80 (one char). Measuring the rest of the tree first is
  what turned "is this house style?" into a decidable question — do that before
  reflowing anyone's prose.
- **Prose style to preserve when reflowing:** two spaces after a sentence-ending
  period (measured 903 two-space vs 75 one-space intra-line in
  `cogwheel/lensing/`). A naive `' '.join(words)` rewrap silently collapses them
  and triples the diff.
- **Safe rewrap recipe** (used here, worked): re-fill only the tail of the
  paragraph starting at the overlong line; keep `(word, separator)` pairs rather
  than bare words; stop the paragraph at blank lines, bullet markers
  (`* `/`- `/`N. `), banner rules, `--`-prefixed section headers, and a lone
  closing `"""`. Verify by comparing the whitespace-normalized, comment-marker-
  stripped word stream of the whole file against `git show HEAD:<file>` — that
  catches an off-by-one slice eating an adjacent code line (it did).
- **Bullet items need a hanging-indent continuation prefix.** A `* ...` item at
  indent 4 with continuation lines at indent 6: filling with the item's own
  indent strands the overflow word at indent 4 and breaks the block. Detect the
  deeper prefix from the next line, but only when the target line is itself a
  bullet (otherwise numpydoc `name : type` blocks get merged into their
  descriptions).
- **Section banners are unfixable without content change.** Two comment headers
  in `surrogate_training.py` (`# -- ... --` and `# --- Saddle ...`) sit at 80-81
  chars; wrapping either merges the header into the following prose or strands a
  fragment. Left them and reported. A future run will re-flag them — that is the
  correct outcome, not a defect.
- **Section-banner blank-line pattern is NOT a spacing defect.** In
  `surrogate_census.py` / `surrogate_training.py`, top-level defs preceded by
  `\n\n# ---- banner ----\n\n def` read as "1 blank line before def" to a naive
  checker. The two blank lines precede the banner; this is correct PEP 8.
- **`from __future__ import annotations` always looks unused** to an AST-based
  import checker (the name `annotations` is never referenced). Never strip it.
- **Tooling gap:** neither `pylint`, `pyflakes`, nor `autoflake` is installed in
  the local env (`cogwheel-newlal`). The rubric's `autoflake` step cannot run as
  written; an AST-based unused-import check plus a per-line length/whitespace
  scan is the working substitute.
