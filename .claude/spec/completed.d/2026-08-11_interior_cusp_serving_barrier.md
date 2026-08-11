---
date: 2026-08-11
section: Backlog
---
## Interior cusp serving barrier — resolved (calibration bypass + fold-band gate)

Implemented in build `interior_cusp_serving_barrier` (code in working tree,
`cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`, `cusp_amplification`).

- **Calibration bypass for interior cusp sources**: the per-image delay
  calibration certificate (`_calibration_certified`) is now guarded by
  `if len(stationary_values) != 3:` — 3-stationary (interior, `rho < 1`)
  sources skip the certificate. Safe because the uniform-error gate
  (`radius >= radius_min`) already bounds the answer to the
  `envelope_bar` tolerance, and `P/P_asymp` is self-calibrating to leading
  order (both evaluated at the same `(x, y)`, so a control miscalibration
  cancels at first order). Exterior (1-stationary) sources still certify
  delay-to-image alignment.
- **ppGO fold-band gate**: the high-w ppGO fast rung now requires
  `nearest.distance >= _airy_fold._ETA_MAX_FOLD` — refuses inside the fold
  arm's serving band, where the fold arm is the designated rung; serving
  there with the cusp arm double-served the corner with a different answer
  (measured 44% disagreement). Restores the serving-ladder partition
  between the fold and cusp arms.
- **Tests**: `InteriorCuspServingTestCase` (3 tests) in
  `test_lensing_airy_fold.py` pins the interior regime's serve and the
  fold-band refusal; stale-test updates (O(1)→bounded-constant geometry
  docstring, `_EXTERIOR_VERTEX_CONFIGS`, wedge-edge carve-out rewrite,
  float-noise tolerance).
- **Test-infrastructure**: ladder-node frequencies moved above the Schwinger
  QD ceiling (`w = 150`) so the exact engine hard-refuses instantly instead
  of entering the unbounded adaptive `mp.quad` band (`_CUSP_NODE_W` 80→160,
  `_GEOMETRIC_NODE` w 100→200, `FOP_REFUSALS`/supra 63→160,
  `LEVER5_ABOVE_CEILING_W` 62→160). Retires
  `lensing_fast_tier_hangs_in_mpmath` and 10/12 items of
  `lensing_serving_ladder_guards_are_red`; the 2 remaining reds are
  genuine production issues tracked in
  `lensing_mpmath_band_fixed_panel_rule.md` (deferred production fix).

ACCEPTANCE (all met): interior cusp sources (3 comparable images) are
served by a fast path (Pearcey arm, not the exact engine); the refusal
barrier is understood and fixed (calibration certificate was the barrier);
no live quadrature in the hot path; exterior cusp sources continue via ppGO
(fold-band gate only tightens their serving partition).
