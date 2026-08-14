# Inspector Short-Term Observations

## 2026-08-14 — tube_d2_fold RE-REVIEW (VERDICT: PASS; INS-2-001 + INS-2-002 RESOLVED)

Re-reviewed the working tree after the Coder applied the two open findings.
Scope: scripts/census_dry_run.py, scripts/train_surrogate_production.py,
cogwheel/lensing/surrogate.py (docstring), surrogate_training.py (config comment).

### INS-2-001 — CONFIRMED RESOLVED (re-verified, not on trust)
census_dry_run.py:127 now computes `arc_r_min` over
`_st._tube_training_arcs(structure, _SADDLE_PARITY)` (parity=-1 => `list(structure.arcs)`,
the FULL saddle arc set) instead of `structure.arcs[:cfg.max_tube_arcs]`. This is a
faithful mirror of production's `_train_band_charts` saddle path (served == counted).
train_surrogate_production.py dropped both the `max_tube_arcs=20` assignment and its
banner print line. TrainingConfig.max_tube_arcs (surrogate_training.py:309) now carries
a 3-line comment: "No longer governs production tube training... Retained only for tests
that set it explicitly." Test-local slices in test_lensing_caustic_cusps.py:1301,1660
and the degenerate-band CONFIG (test_lensing_surrogate_training.py:6329 max_tube_arcs=4)
correctly left untouched per prior instruction. Scripts parse (ast.parse ok).

### INS-2-002 — CONFIRMED RESOLVED
surrogate.py `_evaluate_chart` docstring (~:3326-3340) reworded: y1_eig/y2_eig are now
documented as consumed by the tube branch to fold theta into the D2 fundamental domain
via `_fold_caustic_theta` before the frame/arc-length mapping (same fold as the
`_tube_serves` gate). No logic change.

### Verification
- test_lensing_tube_d2_fold.py: 30 passed / 4.88s (fast tier, green).
- No new findings. Diff is exactly the two requested fixes + config comment; nothing
  else in the serve path changed since the prior review (which had already cleared the
  WP1 core and INS-1-001).

### STILL OPEN -> Librarian (doc-sync, NOT code defects; carried from prior review)
- todo.d/lensing_saddle_tube_fundamental_training.md close + SPEC.md serving-regime text.
- INS-1-002/003 lineage: exterior_polar V5 tag staleness; region vocabulary
  (lobe_exterior/interior/wedge) absent from SPEC/CONTRACTS; saddle exterior raw-theta text.
