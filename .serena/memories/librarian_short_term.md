# Librarian Short-Term Observations

## 2026-08-07 ExteriorPolarChart partial build post-commit sync

**Scope**: One pending commit from sync_issues.json:
- `4d59a6d` — `feat(lensing): partial ExteriorPolarChart + polar training wiring (build stranded)`

**Changed files in commit** (doc-relevant subset):
- `cogwheel/lensing/surrogate.py` — major: added `ExteriorPolarChart` class (rho, theta_c,
  axis-schema 'exterior_polar_rho_theta_c'), wired into `select_chart` and `LensAmplificationSurrogate`;
  `FarFieldChart` (s, d, tag 'farfield_arclength_s_perp_d_framewinv') retained as backward-compat.
- `cogwheel/lensing/surrogate_training.py` — training wired to ExteriorPolarChart
- Test files — skip entirely

**Stale surfaces found and fixed**:

1. **SPEC.md lines 60-66** (Key abstractions section): Described `FarFieldChart` with `(s, d)`
   as the positive-parity exterior chart and `(rho, theta_c)` as "retained only for tile proposal".
   Now describes `ExteriorPolarChart` with `(rho, theta_c)` and tag `'exterior_polar_rho_theta_c'`
   as the active exterior chart, with `FarFieldChart` noted as backward-compat.
   → Created `spec_changelog.d/2026-08-07_exterior_polar_chart.md` (bump: patch)

2. **DATA_CONTRACTS.yaml** (lens_amplification_surrogate description): The paragraph "Each
   FarFieldChart record (exterior far-field only...)" described FarFieldChart with (s,d) as the
   exterior chart and said it was "replacing the retired caustic-fixed (rho, theta_c) axes".
   This was doubly stale: the coordinate role is now inverted (rho, theta_c) is active, (s,d)
   is compat. Replaced with ExteriorPolarChart description.
   → Created `contracts_changelog.d/2026-08-07_exterior_polar_chart.md` (bump: patch)

**Skipped**:
- SPEC.md main table rows (lines 54, 56): NAMING HAZARD mentions FarFieldChart (still exists in
  code as a class — not wrong, just incomplete). No pipeline-step or module-attribution errors;
  the table row is a description of the engine, not the exterior chart contract.
- docs/source/: No mentions of FarFieldChart or ExteriorPolarChart anywhere — no Sphinx updates needed.
- overview.rst, crash_course.rst: Neither references these chart classes directly.

**Pattern**: Class renames in surrogate.py go stale in TWO places: SPEC.md "Key abstractions"
section (which names the active exterior chart class) AND DATA_CONTRACTS.yaml (which describes
the serialized format per chart type). A commit that adds a new chart type and retires the old
role but doesn't delete the old class creates a documentation state where BOTH classes must be
described, with the new one as active and old as compat — don't just replace one mention.

**sync_derived_docs.py**: lens_amplification_surrogate test-only-caller warning recurred again
(7th+ time). Existing escalation TODO fragment in place. No diff from script.

**Fragile cross-reference watch**: The (s,d) arc_map description in DATA_CONTRACTS.yaml (the
sentences after the new ExteriorPolarChart paragraph) still describes the arc_map fields
(arc_gamma_nodes, arc_theta_fine, arc_s_table) — these belong to FarFieldChart which still
exists. They should be accurate as long as FarFieldChart persists. If a future commit deletes
FarFieldChart entirely, those sentences become dead description and should be removed.
