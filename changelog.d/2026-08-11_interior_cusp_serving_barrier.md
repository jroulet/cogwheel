---
date: 2026-08-11
---
### Interior cusp sources now serve via the Pearcey arm; cross-arm ppGO fold-band gate

The last refusing cusp regime is served: interior cusp sources (inside the
caustic near a cusp vertex, 3 comparable images, `rho < 1`) no longer fall
to the exact engine. Two changes in `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`:

- **Calibration bypass for the 3-stationary interior regime**: the per-image
  delay calibration certificate (`_calibration_certified`) now applies only
  when `len(stationary_values) != 3` — interior sources skip it. This is
  safe because the uniform-error gate (`radius >= radius_min`) already
  bounds the answer to the `envelope_bar` tolerance, and the ratio
  `P/P_asymp` is self-calibrating to leading order (both evaluated at the
  same `(x, y)`, so a control miscalibration cancels at first order).
  Exterior (1-stationary) sources still validate delay-to-image alignment.
- **ppGO fold-band gate**: the high-w ppGO fast rung now requires
  `nearest.distance >= _airy_fold._ETA_MAX_FOLD` — it refuses inside the
  fold arm's serving band, where the fold arm is the designated rung and
  serving there with the cusp arm double-served the corner with a different
  answer (measured 44% disagreement). Restores the serving-ladder partition
  between the fold and cusp arms.

Also retires the fast-tier mpmath-hang cluster at the TEST level (parameter
choice): ladder-node frequencies moved above the Schwinger QD ceiling
(`w = 150`) so the exact engine hard-refuses instantly instead of entering
the unbounded adaptive `mp.quad` band — `_CUSP_NODE_W` 80→160 and
`_GEOMETRIC_NODE` w 100→200 (`test_lensing_airy_fold.py`), `FOP_REFUSALS`
and supra grids 63→160 (`test_lensing_fast_path.py`), and
`LEVER5_ABOVE_CEILING_W` 62→160 (`test_lensing_levers.py`). New
`InteriorCuspServingTestCase` (3 tests) pins the interior regime's serve
and the fold-band refusal. The production fix (bounded fixed-panel mpmath
rule) is deferred, tracked in `lensing_mpmath_band_fixed_panel_rule.md`.
