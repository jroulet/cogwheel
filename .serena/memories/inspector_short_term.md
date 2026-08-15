# Inspector Short-Term Observations

## 2026-08-15 — Build saddle_tube_fundamental_training (F081) pass-2 — PASS

Scope: uncommitted tree, /home/tejaswi/Work/cogwheel-claude-dev. Re-review of the
three test-file findings raised in pass-1 (INS-1-001/002/003). Production code
(surrogate_training.py +115/-, tiling_census.py, scripts) UNCHANGED since pass-1
where I already confirmed it correct (D2 orbit-partition `_tube_training_arcs`,
min_eta_max lobe-edge shell, field `max_tube_arcs` removed everywhere).

### All three pass-1 findings RESOLVED (re-derived, not just deleted):
- INS-1-001: test_lensing_surrogate_training.py — `max_tube_arcs=4` class-body
  literal removed. Collection now succeeds: 279 tests collected across the 4
  suites, no TypeError. Zero `max_tube_arcs` refs in the file.
- INS-1-002: test_lensing_tiling_census.py — ArcCensusQ1TestCase (2 tests) PASS
  in 12s. `test_detected_and_trained_arc_counts_match_fundamental_domain` now
  derives `expected_saddle_trained = len(st._tube_training_arcs(saddle_ctxs[0]
  .structure, -1))` with strict `1 <= trained < _SADDLE_DETECTED_ARCS(6)` teeth.
  Retired widening test replaced by `test_saddle_folds_strictly_below_detected_
  while_astroid_pins_one` (uses `n_orbits = len(_tube_training_arcs(structure,-1))`).
- INS-1-003: test_lensing_caustic_cusps.py — both `test_margin_removal_is_a_safe_
  superset` and `test_inflated_margin_changes_admission` PASS in 12s; both now use
  `st._tube_training_arcs(structure, 1)` (astroid single pi/4 arc) instead of the
  removed `structure.arcs[:config.max_tube_arcs]` slice.

### Verification runs (all green)
- collect-only over 4 suites: 279 collected, no error.
- ArcCensusQ1TestCase: 2 passed.
- caustic_cusps -k "margin_removal_is_a_safe_superset or inflated_margin_changes_
  admission": 2 passed.
- tube_d2_fold + tiling_census -k D2/orbit/fold/saddle: 28 passed.
- Tree-wide grep: only 3 residual `max_tube_arcs` hits, ALL docstrings/comments
  naming the "retired" knob (test_lensing_tube_d2_fold.py L49/583, tiling_census
  test L550). No live refs.

### Lesson reinforced
The pass-1 "field removal is a signature change — sweep ALL tests same build"
finding was correctly actioned: the deferral rationale ("owned by other runs")
was invalid and the follow-up build fixed all three files by RE-DERIVING the
expectation from the production selector, which is the right fix (a green test
keyed on the new D2-orbit semantics), not a delete.

### Carry-forward -> Librarian (doc staleness, NOT this build's code)
- Region vocabulary (lobe_exterior/lobe_interior/wedge_interior) still absent
  from SPEC.md / DATA_CONTRACTS.yaml.
- exterior_polar_rho_log_carrier_v1 "ONLY known tag" stale since V5 2D carrier.
