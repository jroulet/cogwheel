# Tidy Short-Term Observations

- cogwheel/lensing/prior.py: within-package sibling imports
  (`cogwheel.lensing.likelihood`, `cogwheel.lensing.marginalized_likelihood`)
  should sort AFTER the broader `cogwheel.gw_prior` / `cogwheel.prior`
  layer imports, not interleaved between them — mirrors the pattern
  already used in marginalized_likelihood.py (cogwheel.likelihood before
  cogwheel.lensing.likelihood). Reordered accordingly.
- cogwheel/lensing/__init__.py uses absolute imports
  (`from cogwheel.lensing.prior import ...`) while every other
  __init__.py in the repo (likelihood, gw_prior, lensing/chang_refsdal)
  uses relative dot-imports (`from .module import ...`). Left as-is
  since import ordering/spacing rubric doesn't mandate relative-vs-
  absolute style changes; flagging for a dedicated normalization pass.
- cogwheel/lensing/marginalized_likelihood.py required zero edits
  (spacing, import order, and unused-import checks all already
  rubric-compliant) — confirms "zero edits can be correct" per prior
  long-term note.
- autoflake unavailable in this environment (not on PATH); unused-import
  check for these 3 files done manually by cross-referencing every
  imported name against its usages in-file.
