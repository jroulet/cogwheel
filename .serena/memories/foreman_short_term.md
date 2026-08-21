# Foreman-Lite Short-Term Observations

## 2026-08-21 (INS-3-002, dead import `_born_factors` in likelihood.py)

- DEAD-IMPORT REMOVAL (single-line): `_born_factors` was imported at
  likelihood.py:110 but its only caller (old quotient-chart serve's
  `sqrt_mu_full = _born_factors(...)[0]`) was replaced by the new
  `_low_w_shell_chart_serve` composing `mass_sheet_phase * f_serve / lam`.
  Before editing, `search_for_pattern` the symbol name across the WHOLE file
  to confirm the import line is the only occurrence — other files
  (`_diffractive.py`, `_born.py`, tests) import `_born_factors` through their
  OWN imports and are untouched. One `replace_content` fixed it; verification
  = ast.parse + live import + `hasattr` absent/present asserts. No pytest
  needed (import-only change, zero behavior).
- NOTE: a redundant second replace_content with the SAME needle returns a
  ValueError (no matches) after the first already applied — don't read that
  as an edit failure; re-read the file to confirm the first edit landed.
- SPEC-staleness scan: SPEC.md mentions `_born_factors` only in the
  born_carrier_omitted_term context (a live quantity still computed) — no
  staleness from this removal.
