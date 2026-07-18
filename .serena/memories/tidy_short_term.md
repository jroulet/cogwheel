# Tidy Short-Term Observations

- cogwheel/lensing/likelihood.py (checked 2026-07-18): already fully
  rubric-compliant — multi-line `from cogwheel.lensing.* import (...)`
  continuations align under the opening paren correctly (verified by
  measuring indent vs paren column, don't trust eyeballing wrapped
  imports), no unused imports, no spacing violations. Zero-edit result
  confirmed correct, not a missed check.
- autoflake is not installed in this environment (`python3 -m autoflake`
  -> ModuleNotFoundError); fell back to manual grep-count cross-check of
  each imported name's usages instead.
