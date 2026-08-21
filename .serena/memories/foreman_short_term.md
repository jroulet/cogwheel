# Foreman-Lite Short-Term Observations

## 2026-08-21 (INS-2-001/002, _pearcey_cusp.py dead-field + dead-accessor removal)

- DEAD-FIELD/ACCESSOR REMOVAL in a file carrying parallel uncommitted work: the
  targeted source-string asserts (dataclass fields list, hasattr absent, __all__
  membership) are the reliable verification — git diff shows the whole parallel
  refactor block as `+` (DIFF TRAP) so diff isolation is unusable.
- YAGNI choice for INS-2-002: DROPPED the unused public scalar
  `cusp_uniform_reference` + `_cusp_uniform_form` wrapper (zero prod callers, zero
  tests, exact duplicate of `cusp_uniform_reference_grid([w])[0]`) instead of
  adding a pin test — the Inspector finding explicitly sanctioned the drop, and
  the scalar was a leftover from the earlier grid-refactor (INS-1-003) that moved
  the only production consumer to the grid form.
- Deleting a mid-file function via regex `\ndef NAME\(.*?<last-body-line>\n` with
  an EMPTY repl leaves the correct two-blank-line spacing; a repl of a single
  newline leaves THREE blank lines (needs a follow-up cleanup pass).
- Docstring sweep after dropping `_cusp_uniform_form`: 5 sites referenced the
  dead wrapper (`_CuspUniformGeometry`, `_cusp_uniform_geometry`,
  `_cusp_uniform_at_w` docstrings + `cusp_amplification` comment +
  `cusp_uniform_reference_grid` docstring) — reworded to the surviving
  `_cusp_uniform_geometry` / `_cusp_uniform_at_w` / `_CuspUniformForm`.
- Keep computing branch/vertex/phi_ssr LOCALLY in `_cusp_uniform_geometry` (they
  feed vertex/curvature/c4) — only the dataclass STORAGE is dead; removing the
  local computations would break the derived fields.
- Verification: ast.parse + import + fields() list + attr-absence asserts, then
  behavior via existing tests (CuspFref*/FoldCuspContinuity in
  test_lensing_low_w_diffractive_chart.py, 60 cusp/Pearcey tests in
  test_lensing_airy_fold.py, 4 pre-existing train-tier skips) — no new test
  needed since the code removed had none.
- SPEC.md scan: only hit for `phi_ssr` is the F074 physics formula in the
  serving-ladder paragraph — the mathematical quantity still lives in the local
  computation, so no spec staleness from this removal.
