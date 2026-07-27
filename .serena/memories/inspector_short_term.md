# Inspector Short-Term Observations

## 2026-07-27 — Build 8h-b4 follow-up review (INS-4-001 saddle-axis formula fix)

Scope: uncommitted worktree cogwheel-claude-dev, branch claude-dev.
ONLY production change: cogwheel/lensing/surrogate_training.py (`_train_band_charts`,
~L3206 and ~L3242). No test files touched; surrogate.py NOT touched this build.

### Verdict: ISSUES (1 resolved: INS-4-001; INS-2-001 + INS-3-001 STILL OPEN)

### INS-4-001 — RESOLVED
Both `exclusion_rho` and `rho_outer_region` in `_train_band_charts` now use the
additive form `1.0 + <physical> - coordinate_radius_min` for BOTH parities
(the multiplicative `/ coordinate_radius_min` else-arm is gone).
- parity==1: BYTE-IDENTICAL to HEAD (parity==1 branch already used exactly this
  additive expression; removing the conditional is a no-op for it). Confirmed
  via diff + test_lensing_exterior_admission.py 23 passed (141s).
- parity!=1 (saddle): now exact inverse of `_from_caustic_fixed` saddle arm
  `y_mag = _caustic_reach(gamma) + rho - 1` => `rho = 1 + |y| - _caustic_reach`.
  `coordinate_radius_min` for parity!=1 = `min` over band edges/midpoint of
  `_scalar_caustic_reach` (which is `_caustic_reach` imported-as-alias, L59),
  i.e. band-minimum scalar reach — a conservative inner-edge bound, mirroring
  parity==1's min-over-angles semantics. rho=1 <=> |y|=_caustic_reach, drho/d|y|=1.
Module ast.parse OK. Saddle exterior charts still not trained (dormant path), so
zero live behavior change; fix is correctness-consistency for when S2-2 wiring lands.

### INS-2-001 — STILL OPEN (blocker vs acceptance (d))
Far-field battery NOT ported. test_lensing_surrogate.py::BetaEliminationTestCase
::test_eigenframe_envelope_is_beta_invariant still RED: "the anchor beta=0 source
is out of domain" (served_0=False), stale (y1,y2)->(rho,theta_c) API. -x run
1 failed + 1 error in 28s. Acceptance (d) requires this + ppgo_bandsplit +
surrogate_census + exterior_windows GREEN — unmet. WP3 port not done this build.

### INS-3-001 — STILL OPEN (flag to Librarian)
SPEC.md + DATA_CONTRACTS.yaml unchanged. Saddle exterior axis prose (multiplicative
-> additive) likely stale. DATA_CONTRACTS `rho=|y|/caustic_reach` is the ppGO
annulus (scalar-reach, intentionally unchanged) — NOT a divergence. Surrogate .npz
offline/unshipped — no contract entry owed.

### Carry-forward
- Re-run FULL far-field battery once WP3 port lands; watch on-axis ghost xfail +
  RB delta_t_max edge-margin fixtures under caustic-fixed larger-separation configs.
- INS-1-001 (ghost_kernel double _ghost_delay) not re-examined; presumed open,
  non-blocking micro-DRY.
