# Inspector Short-Term Observations

## 2026-07-30 — Build 1d WP1 (delete _WEDGE_EPS / analytic _tube_normal / false-prose fix)

Scope: uncommitted working-tree diff. Production: cogwheel/lensing/surrogate_training.py
(deleted `_WEDGE_EPS=1e-3`, 6 edge-sample sites now at true wedge edge; `_tube_normal`
rewritten to use analytic `geometry.caustic_derivatives` y'/|y'| instead of 1e-6 finite
diff) + geometry.py `caustic_derivatives` docstring only.

VERDICT: PASS — no new findings.

Why correct:
- `_tube_normal`: y'=d(source)/dtheta, same beta=0/kappa=0 parametrization as swept
  `critical_point`. y' and old forward-diff both point along +theta => (-t_y,t_x)
  rotation preserves orientation => inward_sign unchanged. Confirmed by passing golden
  inward_sign gate (Gate 4b) + census served/reconstructed consistency.
- All 6 true-edge sites safe: critical_point succeeds at edge; caustic_speed/
  caustic_derivatives wrapped in per-theta LensDomainError skips (_branch_speed_profile,
  _lobe_*). Tube-serve thetas stay in arc interior (edge window 0.08 + _ARC_MARGIN_FRAC),
  so `_tube_source` never hits the edge divergence. `_make_arc` catches LensDomainError.
- Docstrings now truthful (this build's stated goal): F044 wedge-edge=regular-point;
  `_winding_number` IS applied to saddle lobes via `_SaddleLobeAdmission.admits` L2252
  (old "never applied" was the false prose); `_lobe_winding_loop` bit-exact 0.0 closure
  asserted by passing Gate 2 (disc clamps <=0 at true edge, both branches coincide).

Tests: test_lensing_saddle_geometry + caustic_cusps = 58 passed/2 skip; surrogate_training
= 20 passed/36 skip (all COGWHEEL_TRAIN_TIER post-build, no setup errors); census sibling
consumer 14 passed/13 skip. No stale `_WEDGE_EPS` import (test-local `_LEGACY_WEDGE_EPS`
is a deliberate frozen PRE-WP1 comparison).

Non-blocking note (not a finding): `test_normal_is_unit_and_perpendicular_to_analytic_
tangent` is mildly circular (caustic_derivatives on both sides) but paired with an AST
no-theta-BinOp guard + the load-bearing golden inward_sign table, so orientation is
independently pinned.

DATA_CONTRACTS: none needed — training-path helper change, `lens_amplification_surrogate`
schema/format unchanged (only sample placement, an accuracy improvement).

Open issues carried forward: none new.
